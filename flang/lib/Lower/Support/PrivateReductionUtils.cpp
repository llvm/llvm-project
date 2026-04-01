//===-- PrivateReductionUtils.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Support/PrivateReductionUtils.h"

#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/CUDA.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Semantics/symbol.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "aiir/IR/Location.h"
#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<bool> enableGPUHeapAlloc(
    "enable-gpu-heap-alloc",
    llvm::cl::desc(
        "Allow the use of heap allocation for dynamically sized arrays on GPU"),
    llvm::cl::init(false));

static bool hasFinalization(const Fortran::semantics::Symbol &sym) {
  if (sym.has<Fortran::semantics::ObjectEntityDetails>())
    if (const Fortran::semantics::DeclTypeSpec *declTypeSpec = sym.GetType())
      if (const Fortran::semantics::DerivedTypeSpec *derivedTypeSpec =
              declTypeSpec->AsDerived())
        return Fortran::semantics::IsFinalizable(*derivedTypeSpec);
  return false;
}

static void createCleanupRegion(Fortran::lower::AbstractConverter &converter,
                                aiir::Location loc, aiir::Type argType,
                                aiir::Region &cleanupRegion,
                                const Fortran::semantics::Symbol *sym,
                                bool isDoConcurrent) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  assert(cleanupRegion.empty());
  aiir::Block *block = builder.createBlock(&cleanupRegion, cleanupRegion.end(),
                                           {argType}, {loc});
  builder.setInsertionPointToEnd(block);

  auto typeError = [loc]() {
    fir::emitFatalError(loc,
                        "Attempt to create an omp cleanup region "
                        "for a type that wasn't allocated",
                        /*genCrashDiag=*/true);
  };

  aiir::Type valTy = fir::unwrapRefType(argType);
  const bool argIsVolatile = fir::isa_volatile_type(argType);
  if (auto boxTy = aiir::dyn_cast_or_null<fir::BaseBoxType>(valTy)) {
    // TODO: what about undoing init of unboxed derived types?
    if (auto recTy = aiir::dyn_cast<fir::RecordType>(
            fir::unwrapSequenceType(fir::dyn_cast_ptrOrBoxEleTy(boxTy)))) {
      aiir::Type eleTy = boxTy.getEleTy();
      if (aiir::isa<fir::PointerType, fir::HeapType>(eleTy)) {
        aiir::Type mutableBoxTy =
            fir::ReferenceType::get(fir::BoxType::get(eleTy), argIsVolatile);
        aiir::Value converted =
            builder.createConvert(loc, mutableBoxTy, block->getArgument(0));
        if (recTy.getNumLenParams() > 0)
          TODO(loc, "Deallocate box with length parameters");
        fir::MutableBoxValue mutableBox{converted, /*lenParameters=*/{},
                                        /*mutableProperties=*/{}};
        Fortran::lower::genDeallocateIfAllocated(converter, mutableBox, loc);
        if (isDoConcurrent)
          fir::YieldOp::create(builder, loc);
        else
          aiir::omp::YieldOp::create(builder, loc);
        return;
      }
    }

    // TODO: just replace this whole body with
    // Fortran::lower::genDeallocateIfAllocated (not done now to avoid test
    // churn)

    aiir::Value arg = builder.loadIfRef(loc, block->getArgument(0));
    assert(aiir::isa<fir::BaseBoxType>(arg.getType()));

    // Extract address from the box for deallocation.
    // The FIR type system doesn't necessarily know that this is a mutable
    // box if we allocated the thread local array on the heap to avoid looped
    // stack allocations.
    aiir::Value addr =
        hlfir::genVariableRawAddress(loc, builder, hlfir::Entity{arg});

    // Deallocate if allocated
    aiir::Value isAllocated = builder.genIsNotNullAddr(loc, addr);
    fir::IfOp ifOp =
        fir::IfOp::create(builder, loc, isAllocated, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    aiir::Value cast = builder.createConvert(
        loc, fir::HeapType::get(fir::dyn_cast_ptrEleTy(addr.getType())), addr);
    fir::FreeMemOp::create(builder, loc, cast);

    builder.setInsertionPointAfter(ifOp);
    // Free the managed descriptor if this is a CUDA device allocatable.
    if (sym) {
      unsigned idx = Fortran::lower::getAllocatorIdx(sym->GetUltimate());
      if (idx != kDefaultAllocator) {
        cuf::DataAttributeAttr dataAttr =
            Fortran::lower::translateSymbolCUFDataAttribute(
                builder.getContext(), sym->GetUltimate());
        cuf::FreeOp::create(builder, loc, block->getArgument(0), dataAttr);
      }
    }
    if (isDoConcurrent)
      fir::YieldOp::create(builder, loc);
    else
      aiir::omp::YieldOp::create(builder, loc);
    return;
  }

  // Handle !fir.boxchar (passed by VALUE for runtime-length characters).
  // Note: This is distinct from !fir.box<!fir.char<>> which is handled above.
  // BoxChar is a special tuple type (addr, len) used when character length
  // is only known at runtime.
  if (auto boxCharTy = aiir::dyn_cast<fir::BoxCharType>(argType)) {
    auto [addr, len] =
        fir::factory::CharacterExprHelper{builder, loc}.createUnboxChar(
            block->getArgument(0));

    // convert addr to a heap type so it can be used with fir::FreeMemOp
    auto refTy = aiir::cast<fir::ReferenceType>(addr.getType());
    auto heapTy = fir::HeapType::get(refTy.getEleTy());
    addr = builder.createConvert(loc, heapTy, addr);

    fir::FreeMemOp::create(builder, loc, addr);
    if (isDoConcurrent)
      fir::YieldOp::create(builder, loc);
    else
      aiir::omp::YieldOp::create(builder, loc);

    return;
  }

  typeError();
}

fir::ShapeShiftOp Fortran::lower::getShapeShift(
    fir::FirOpBuilder &builder, aiir::Location loc, aiir::Value box,
    bool cannotHaveNonDefaultLowerBounds, bool useDefaultLowerBounds) {
  fir::SequenceType sequenceType = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(box.getType()));
  const unsigned rank = sequenceType.getDimension();

  llvm::SmallVector<aiir::Value> lbAndExtents;
  lbAndExtents.reserve(rank * 2);
  aiir::Type idxTy = builder.getIndexType();

  aiir::Value oneVal;
  auto one = [&] {
    if (!oneVal)
      oneVal = builder.createIntegerConstant(loc, idxTy, 1);
    return oneVal;
  };

  if ((cannotHaveNonDefaultLowerBounds || useDefaultLowerBounds) &&
      !sequenceType.hasDynamicExtents()) {
    // We don't need fir::BoxDimsOp if all of the extents are statically known
    // and we can assume default lower bounds. This helps avoids reads from the
    // mold arg.
    // We may also want to use default lower bounds to iterate through array
    // elements without having to adjust each index.
    for (int64_t extent : sequenceType.getShape()) {
      assert(extent != sequenceType.getUnknownExtent());
      lbAndExtents.push_back(one());
      aiir::Value extentVal = builder.createIntegerConstant(loc, idxTy, extent);
      lbAndExtents.push_back(extentVal);
    }
  } else {
    for (unsigned i = 0; i < rank; ++i) {
      // TODO: ideally we want to hoist box reads out of the critical section.
      // We could do this by having box dimensions in block arguments like
      // OpenACC does
      aiir::Value dim = builder.createIntegerConstant(loc, idxTy, i);
      auto dimInfo =
          fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy, box, dim);
      lbAndExtents.push_back(useDefaultLowerBounds ? one()
                                                   : dimInfo.getLowerBound());
      lbAndExtents.push_back(dimInfo.getExtent());
    }
  }

  auto shapeShiftTy = fir::ShapeShiftType::get(builder.getContext(), rank);
  auto shapeShift =
      fir::ShapeShiftOp::create(builder, loc, shapeShiftTy, lbAndExtents);
  return shapeShift;
}

// Initialize box newBox using moldBox. These should both have the same type and
// be boxes containing derived types e.g.
// fir.box<!fir.type<>>
// fir.box<!fir.heap<!fir.type<>>
// fir.box<!fir.heap<!fir.array<fir.type<>>>
// fir.class<...<!fir.type<>>>
// If the type doesn't match , this does nothing
static void initializeIfDerivedTypeBox(fir::FirOpBuilder &builder,
                                       aiir::Location loc, aiir::Value newBox,
                                       aiir::Value moldBox, bool hasInitializer,
                                       bool isFirstPrivate) {
  assert(moldBox.getType() == newBox.getType());
  fir::BoxType boxTy = aiir::dyn_cast<fir::BoxType>(newBox.getType());
  fir::ClassType classTy = aiir::dyn_cast<fir::ClassType>(newBox.getType());
  if (!boxTy && !classTy)
    return;

  // remove pointer and array types in the middle
  aiir::Type eleTy = boxTy ? boxTy.getElementType() : classTy.getEleTy();
  aiir::Type derivedTy = fir::unwrapRefType(eleTy);
  if (auto array = aiir::dyn_cast<fir::SequenceType>(derivedTy))
    derivedTy = array.getElementType();

  if (!fir::isa_derived(derivedTy))
    return;

  if (hasInitializer)
    fir::runtime::genDerivedTypeInitialize(builder, loc, newBox);

  if (hlfir::mayHaveAllocatableComponent(derivedTy) && !isFirstPrivate)
    fir::runtime::genDerivedTypeInitializeClone(builder, loc, newBox, moldBox);
}

static void getLengthParameters(fir::FirOpBuilder &builder, aiir::Location loc,
                                aiir::Value moldArg,
                                llvm::SmallVectorImpl<aiir::Value> &lenParams) {
  // We pass derived types unboxed and so are not self-contained entities.
  // Assume that unboxed derived types won't need length paramters.
  if (!hlfir::isFortranEntity(moldArg))
    return;

  hlfir::genLengthParameters(loc, builder, hlfir::Entity{moldArg}, lenParams);
  if (lenParams.empty())
    return;

  // The verifier for EmboxOp doesn't allow length parameters when the the
  // character already has static LEN. genLengthParameters may still return them
  // in this case.
  auto strTy = aiir::dyn_cast<fir::CharacterType>(
      fir::getFortranElementType(moldArg.getType()));

  if (strTy && strTy.hasConstantLen())
    lenParams.resize(0);
}

static bool
isDerivedTypeNeedingInitialization(const Fortran::semantics::Symbol &sym) {
  // Fortran::lower::hasDefaultInitialization returns false for ALLOCATABLE, so
  // re-implement here.
  // ignorePointer=true because either the pointer points to the same target as
  // the original variable, or it is uninitialized.
  if (const Fortran::semantics::DeclTypeSpec *declTypeSpec = sym.GetType())
    if (const Fortran::semantics::DerivedTypeSpec *derivedTypeSpec =
            declTypeSpec->AsDerived())
      return derivedTypeSpec->HasDefaultInitialization(
          /*ignoreAllocatable=*/false, /*ignorePointer=*/true);
  return false;
}

static aiir::Value generateZeroShapeForRank(fir::FirOpBuilder &builder,
                                            aiir::Location loc,
                                            aiir::Value moldArg) {
  aiir::Type moldType = fir::unwrapRefType(moldArg.getType());
  aiir::Type eleType = fir::dyn_cast_ptrOrBoxEleTy(moldType);
  fir::SequenceType seqTy =
      aiir::dyn_cast_if_present<fir::SequenceType>(eleType);
  if (!seqTy)
    return aiir::Value{};

  unsigned rank = seqTy.getShape().size();
  aiir::Value zero =
      builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  aiir::SmallVector<aiir::Value> dims;
  dims.resize(rank, zero);
  aiir::Type shapeTy = fir::ShapeType::get(builder.getContext(), rank);
  return fir::ShapeOp::create(builder, loc, shapeTy, dims);
}

namespace {
using namespace Fortran::lower;
/// Class to store shared data so we don't have to maintain so many function
/// arguments
class PopulateInitAndCleanupRegionsHelper {
public:
  PopulateInitAndCleanupRegionsHelper(
      Fortran::lower::AbstractConverter &converter, aiir::Location loc,
      aiir::Type argType, aiir::Value scalarInitValue,
      aiir::Value allocatedPrivVarArg, aiir::Value moldArg,
      aiir::Block *initBlock, aiir::Region &cleanupRegion,
      DeclOperationKind kind, const Fortran::semantics::Symbol *sym,
      bool cannotHaveLowerBounds, bool isDoConcurrent)
      : converter{converter}, builder{converter.getFirOpBuilder()}, loc{loc},
        argType{argType}, scalarInitValue{scalarInitValue},
        allocatedPrivVarArg{allocatedPrivVarArg}, moldArg{moldArg},
        initBlock{initBlock}, cleanupRegion{cleanupRegion}, kind{kind},
        sym{sym}, cannotHaveNonDefaultLowerBounds{cannotHaveLowerBounds},
        isDoConcurrent{isDoConcurrent} {
    valType = fir::unwrapRefType(argType);
  }

  void populateByRefInitAndCleanupRegions();

private:
  Fortran::lower::AbstractConverter &converter;
  fir::FirOpBuilder &builder;

  aiir::Location loc;

  /// The type of the block arguments passed into the init and cleanup regions
  aiir::Type argType;

  /// argType stripped of any references
  aiir::Type valType;

  /// sclarInitValue:      The value scalars should be initialized to (only
  ///                      valid for reductions).
  /// allocatedPrivVarArg: The allocation for the private
  ///                      variable.
  /// moldArg:             The original variable.
  /// loadedMoldArg:       The original variable, loaded. Access via
  ///                      getLoadedMoldArg().
  aiir::Value scalarInitValue, allocatedPrivVarArg, moldArg, loadedMoldArg;

  /// The first block in the init region.
  aiir::Block *initBlock;

  /// The region to insert clanup code into.
  aiir::Region &cleanupRegion;

  /// The kind of operation we are generating init/cleanup regions for.
  DeclOperationKind kind;

  /// (optional) The symbol being privatized.
  const Fortran::semantics::Symbol *sym;

  /// Any length parameters which have been fetched for the type
  aiir::SmallVector<aiir::Value> lenParams;

  /// If the source variable being privatized definitely can't have non-default
  /// lower bounds then we don't need to generate code to read them.
  bool cannotHaveNonDefaultLowerBounds;

  bool isDoConcurrent;

  void createYield(aiir::Value ret) {
    if (isDoConcurrent)
      fir::YieldOp::create(builder, loc, ret);
    else
      aiir::omp::YieldOp::create(builder, loc, ret);
  }

  void initTrivialType() {
    builder.setInsertionPointToEnd(initBlock);
    if (scalarInitValue)
      builder.createStoreWithConvert(loc, scalarInitValue, allocatedPrivVarArg);
    createYield(allocatedPrivVarArg);
  }

  void initBoxedPrivatePointer(fir::BaseBoxType boxTy);

  /// e.g. !fir.box<!fir.heap<i32>>, !fir.box<!fir.type<....>>,
  /// !fir.box<!fir.char<...>>
  void initAndCleanupBoxedScalar(fir::BaseBoxType boxTy,
                                 bool needsInitialization);

  void initAndCleanupBoxedArray(fir::BaseBoxType boxTy,
                                bool needsInitialization);

  void initAndCleanupBoxchar(fir::BoxCharType boxCharTy);

  void initAndCleanupUnboxedDerivedType(bool needsInitialization);

  fir::IfOp handleNullAllocatable();

  // Do this lazily so that we don't load it when it is not used.
  inline aiir::Value getLoadedMoldArg() {
    if (loadedMoldArg)
      return loadedMoldArg;
    loadedMoldArg = builder.loadIfRef(loc, moldArg);
    return loadedMoldArg;
  }

  bool shouldAllocateTempOnStack(fir::BaseBoxType boxTy) const;
};

} // namespace

/// The initial state of a private pointer is undefined so we don't need to
/// match the mold argument (OpenMP 5.2 end of page 106).
void PopulateInitAndCleanupRegionsHelper::initBoxedPrivatePointer(
    fir::BaseBoxType boxTy) {
  assert(isPrivatization(kind));
  // we need a shape with the right rank so that the embox op is lowered
  // to an llvm struct of the right type. This returns nullptr if the types
  // aren't right.
  aiir::Value shape = generateZeroShapeForRank(builder, loc, moldArg);
  // Just incase, do initialize the box with a null value
  aiir::Value null = builder.createNullConstant(loc, boxTy.getEleTy());
  aiir::Value nullBox;
  nullBox = fir::EmboxOp::create(builder, loc, boxTy, null, shape,
                                 /*slice=*/aiir::Value{}, lenParams);
  fir::StoreOp::create(builder, loc, nullBox, allocatedPrivVarArg);
  createYield(allocatedPrivVarArg);
}
/// Check if an allocatable box is unallocated. If so, initialize the boxAlloca
/// to be unallocated e.g.
/// %box_alloca = fir.alloca !fir.box<!fir.heap<...>>
/// %addr = fir.box_addr %box
/// if (%addr == 0) {
///   %nullbox = fir.embox %addr
///   fir.store %nullbox to %box_alloca
/// } else {
///   // ...
///   fir.store %something to %box_alloca
/// }
/// omp.yield %box_alloca
fir::IfOp PopulateInitAndCleanupRegionsHelper::handleNullAllocatable() {
  aiir::Value addr = fir::BoxAddrOp::create(builder, loc, getLoadedMoldArg());
  aiir::Value isNotAllocated = builder.genIsNullAddr(loc, addr);
  fir::IfOp ifOp = fir::IfOp::create(builder, loc, isNotAllocated,
                                     /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  // Just embox the null address and return.
  // We have to give the embox a shape so that the LLVM box structure has the
  // right rank. This returns an empty value if the types don't match.
  aiir::Value shape = generateZeroShapeForRank(builder, loc, moldArg);

  auto nullBox = fir::EmboxOp::create(builder, loc, valType, addr, shape,
                                      /*slice=*/aiir::Value{}, lenParams);
  if (sym) {
    unsigned idx = Fortran::lower::getAllocatorIdx(sym->GetUltimate());
    if (idx != kDefaultAllocator)
      nullBox.setAllocatorIdx(idx);
  }
  fir::StoreOp::create(builder, loc, nullBox, allocatedPrivVarArg);
  return ifOp;
}

void PopulateInitAndCleanupRegionsHelper::initAndCleanupBoxedScalar(
    fir::BaseBoxType boxTy, bool needsInitialization) {
  bool isAllocatableOrPointer =
      aiir::isa<fir::HeapType, fir::PointerType>(boxTy.getEleTy());
  aiir::Type innerTy = fir::unwrapRefType(boxTy.getEleTy());
  fir::IfOp ifUnallocated{nullptr};
  if (isAllocatableOrPointer) {
    ifUnallocated = handleNullAllocatable();
    builder.setInsertionPointToStart(&ifUnallocated.getElseRegion().front());
  }

  bool shouldAllocateOnStack = shouldAllocateTempOnStack(boxTy);
  aiir::Value valAlloc =
      (shouldAllocateOnStack)
          ? builder.createTemporary(loc, innerTy, /*name=*/{},
                                    /*shape=*/{}, lenParams)
          : builder.createHeapTemporary(loc, innerTy, /*name=*/{},
                                        /*shape=*/{}, lenParams);

  if (scalarInitValue)
    builder.createStoreWithConvert(loc, scalarInitValue, valAlloc);
  aiir::Value box = fir::EmboxOp::create(builder, loc, valType, valAlloc,
                                         /*shape=*/aiir::Value{},
                                         /*slice=*/aiir::Value{}, lenParams);
  initializeIfDerivedTypeBox(
      builder, loc, box, getLoadedMoldArg(), needsInitialization,
      /*isFirstPrivate=*/kind == DeclOperationKind::FirstPrivateOrLocalInit);
  fir::StoreOp lastOp =
      fir::StoreOp::create(builder, loc, box, allocatedPrivVarArg);

  if (!shouldAllocateOnStack)
    createCleanupRegion(converter, loc, argType, cleanupRegion, sym,
                        isDoConcurrent);

  if (ifUnallocated)
    builder.setInsertionPointAfter(ifUnallocated);
  else
    builder.setInsertionPointAfter(lastOp);

  createYield(allocatedPrivVarArg);
}

bool PopulateInitAndCleanupRegionsHelper::shouldAllocateTempOnStack(
    fir::BaseBoxType boxTy) const {
  auto offloadMod =
      llvm::dyn_cast<aiir::omp::OffloadModuleInterface>(*builder.getModule());
  // On the GPU, always allocate on the stack unless the user explicitly
  // specifies otherwise since heap allocatins are very expensive.
  bool isGPU = offloadMod && offloadMod.getIsGPU();
  if (isGPU && enableGPUHeapAlloc) {
    // Check if it is adjustable array
    if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(boxTy.getEleTy())) {
      if (seqTy.hasUnknownShape() || seqTy.hasDynamicExtents()) {
        return false;
      }
    }
  }
  return isGPU;
}

void PopulateInitAndCleanupRegionsHelper::initAndCleanupBoxedArray(
    fir::BaseBoxType boxTy, bool needsInitialization) {
  bool isAllocatableOrPointer =
      aiir::isa<fir::HeapType, fir::PointerType>(boxTy.getEleTy());
  getLengthParameters(builder, loc, getLoadedMoldArg(), lenParams);

  fir::IfOp ifUnallocated{nullptr};
  if (isAllocatableOrPointer) {
    ifUnallocated = handleNullAllocatable();
    builder.setInsertionPointToStart(&ifUnallocated.getElseRegion().front());
  }

  // Create the private copy from the initial fir.box:
  hlfir::Entity source = hlfir::Entity{getLoadedMoldArg()};

  // Special case for (possibly allocatable) arrays of polymorphic types
  // e.g. !fir.class<!fir.heap<!fir.array<?x!fir.type<>>>>
  if (source.isPolymorphic()) {
    fir::ShapeShiftOp shape =
        getShapeShift(builder, loc, source, cannotHaveNonDefaultLowerBounds);
    aiir::Type arrayType = source.getElementOrSequenceType();
    aiir::Value allocatedArray = fir::AllocMemOp::create(
        builder, loc, arrayType, /*typeparams=*/aiir::ValueRange{},
        shape.getExtents());
    aiir::Value firClass = fir::EmboxOp::create(builder, loc, source.getType(),
                                                allocatedArray, shape);
    initializeIfDerivedTypeBox(
        builder, loc, firClass, source, needsInitialization,
        /*isFirstprivate=*/kind == DeclOperationKind::FirstPrivateOrLocalInit);
    fir::StoreOp::create(builder, loc, firClass, allocatedPrivVarArg);
    if (ifUnallocated)
      builder.setInsertionPointAfter(ifUnallocated);
    createYield(allocatedPrivVarArg);
    aiir::OpBuilder::InsertionGuard guard(builder);
    createCleanupRegion(converter, loc, argType, cleanupRegion, sym,
                        isDoConcurrent);
    return;
  }

  // Allocating on the heap in case the whole reduction/privatization is nested
  // inside of a loop
  auto temp = [&]() {
    if (shouldAllocateTempOnStack(boxTy))
      return createStackTempFromMold(loc, builder, source);

    auto [temp, needsDealloc] = createTempFromMold(loc, builder, source);
    // if needsDealloc, add cleanup region. Always
    // do this for allocatable boxes because they might have been re-allocated
    // in the body of the loop/parallel region
    if (needsDealloc) {
      aiir::OpBuilder::InsertionGuard guard(builder);
      createCleanupRegion(converter, loc, argType, cleanupRegion, sym,
                          isDoConcurrent);
    } else {
      assert(!isAllocatableOrPointer &&
             "Pointer-like arrays must be heap allocated");
    }
    return temp;
  }();

  // Put the temporary inside of a box:
  // hlfir::genVariableBox doesn't handle non-default lower bounds
  aiir::Value box;
  fir::ShapeShiftOp shapeShift = getShapeShift(builder, loc, getLoadedMoldArg(),
                                               cannotHaveNonDefaultLowerBounds);
  aiir::Type boxType = getLoadedMoldArg().getType();
  if (aiir::isa<fir::BaseBoxType>(temp.getType()))
    // the box created by the declare form createTempFromMold is missing
    // lower bounds info
    box = fir::ReboxOp::create(builder, loc, boxType, temp, shapeShift,
                               /*shift=*/aiir::Value{});
  else
    box = fir::EmboxOp::create(builder, loc, boxType, temp, shapeShift,
                               /*slice=*/aiir::Value{},
                               /*typeParams=*/llvm::ArrayRef<aiir::Value>{});

  if (scalarInitValue)
    hlfir::AssignOp::create(builder, loc, scalarInitValue, box);

  initializeIfDerivedTypeBox(
      builder, loc, box, getLoadedMoldArg(), needsInitialization,
      /*isFirstPrivate=*/kind == DeclOperationKind::FirstPrivateOrLocalInit);

  fir::StoreOp::create(builder, loc, box, allocatedPrivVarArg);
  if (ifUnallocated)
    builder.setInsertionPointAfter(ifUnallocated);
  createYield(allocatedPrivVarArg);
}

void PopulateInitAndCleanupRegionsHelper::initAndCleanupBoxchar(
    fir::BoxCharType boxCharTy) {
  aiir::Type eleTy = boxCharTy.getEleTy();
  builder.setInsertionPointToStart(initBlock);
  fir::factory::CharacterExprHelper charExprHelper{builder, loc};
  auto [addr, len] = charExprHelper.createUnboxChar(moldArg);

  // Using heap temporary so that
  // 1) It is safe to use privatization inside of big loops.
  // 2) The lifetime can outlive the current stack frame for delayed task
  // execution.
  // We can't always allocate a boxchar implicitly as the type of the
  // omp.private because the allocation potentially needs the length
  // parameters fetched above.
  // TODO: this deviates from the intended design for delayed task
  // execution.
  aiir::Value privateAddr = builder.createHeapTemporary(
      loc, eleTy, /*name=*/{}, /*shape=*/{}, /*lenParams=*/len);
  aiir::Value boxChar = charExprHelper.createEmboxChar(privateAddr, len);

  createCleanupRegion(converter, loc, argType, cleanupRegion, sym,
                      isDoConcurrent);

  builder.setInsertionPointToEnd(initBlock);
  createYield(boxChar);
}

void PopulateInitAndCleanupRegionsHelper::initAndCleanupUnboxedDerivedType(
    bool needsInitialization) {
  builder.setInsertionPointToStart(initBlock);
  aiir::Type boxedTy = fir::BoxType::get(valType);
  aiir::Value newBox =
      fir::EmboxOp::create(builder, loc, boxedTy, allocatedPrivVarArg);
  aiir::Value moldBox = fir::EmboxOp::create(builder, loc, boxedTy, moldArg);
  initializeIfDerivedTypeBox(builder, loc, newBox, moldBox, needsInitialization,
                             /*isFirstPrivate=*/kind ==
                                 DeclOperationKind::FirstPrivateOrLocalInit);

  if (sym && hasFinalization(*sym))
    createCleanupRegion(converter, loc, argType, cleanupRegion, sym,
                        isDoConcurrent);

  builder.setInsertionPointToEnd(initBlock);
  createYield(allocatedPrivVarArg);
}

/// This is the main driver deciding how to initialize the private variable.
void PopulateInitAndCleanupRegionsHelper::populateByRefInitAndCleanupRegions() {
  if (isPrivatization(kind)) {
    assert(sym && "Symbol information is required to privatize derived types");
    assert(!scalarInitValue && "ScalarInitvalue is unused for privatization");
  }
  // Only check for assumed rank if moldArg is a valid Fortran entity.
  // Boxed types (like allocatable characters) may not be valid entities yet.
  if (hlfir::isFortranEntity(moldArg) && hlfir::Entity{moldArg}.isAssumedRank())
    TODO(loc, "Privatization of assumed rank variable");
  aiir::Type valTy = fir::unwrapRefType(argType);

  if (fir::isa_trivial(valTy)) {
    initTrivialType();
    return;
  }

  bool needsInitialization =
      sym ? isDerivedTypeNeedingInitialization(sym->GetUltimate()) : false;

  if (auto boxTy = aiir::dyn_cast_or_null<fir::BaseBoxType>(valTy)) {
    builder.setInsertionPointToEnd(initBlock);

    // For CUDA device allocatables, allocate the descriptor in managed
    // memory so that CUF kernels can access it from the GPU.
    if (sym && aiir::isa<fir::HeapType>(boxTy.getEleTy())) {
      unsigned idx = Fortran::lower::getAllocatorIdx(sym->GetUltimate());
      if (idx != kDefaultAllocator) {
        cuf::DataAttributeAttr dataAttr =
            Fortran::lower::translateSymbolCUFDataAttribute(
                builder.getContext(), sym->GetUltimate());
        allocatedPrivVarArg =
            cuf::AllocOp::create(builder, loc, valTy,
                                 /*uniq_name=*/llvm::StringRef{},
                                 /*bindc_name=*/llvm::StringRef{}, dataAttr,
                                 /*typeparams=*/aiir::ValueRange{},
                                 /*shape=*/aiir::ValueRange{})
                .getResult();
      }
    }

    // TODO: don't do this unless it is needed
    getLengthParameters(builder, loc, getLoadedMoldArg(), lenParams);

    if (isPrivatization(kind) &&
        aiir::isa<fir::PointerType>(boxTy.getEleTy())) {
      initBoxedPrivatePointer(boxTy);
      return;
    }

    aiir::Type innerTy = fir::unwrapRefType(boxTy.getEleTy());
    bool isDerived = fir::isa_derived(innerTy);
    bool isChar = fir::isa_char(innerTy);
    if (fir::isa_trivial(innerTy) || isDerived || isChar) {
      // boxed non-sequence value e.g. !fir.box<!fir.heap<i32>>
      // Character types in reductions are supported, but derived types are not
      // yet.
      if (isDerived && (isReduction(kind) || scalarInitValue))
        TODO(loc, "Reduction of an unsupported boxed derived type");
      initAndCleanupBoxedScalar(boxTy, needsInitialization);
      return;
    }

    innerTy = fir::extractSequenceType(boxTy);
    if (!innerTy || !aiir::isa<fir::SequenceType>(innerTy))
      TODO(loc, "Unsupported boxed type for reduction/privatization");
    initAndCleanupBoxedArray(boxTy, needsInitialization);
    return;
  }

  // Unboxed types:
  if (auto boxCharTy = aiir::dyn_cast<fir::BoxCharType>(valTy)) {
    initAndCleanupBoxchar(boxCharTy);
    return;
  }
  // Handle unboxed character types (e.g., !fir.char<1,1>).
  // For fixed-length character types, we just need to initialize the value.
  if (fir::isa_char(valTy)) {
    builder.setInsertionPointToEnd(initBlock);
    if (scalarInitValue)
      builder.createStoreWithConvert(loc, scalarInitValue, allocatedPrivVarArg);
    createYield(allocatedPrivVarArg);
    return;
  }
  if (fir::isa_derived(valType)) {
    initAndCleanupUnboxedDerivedType(needsInitialization);
    return;
  }

  TODO(loc,
       "creating reduction/privatization init region for unsupported type");
}

void Fortran::lower::populateByRefInitAndCleanupRegions(
    Fortran::lower::AbstractConverter &converter, aiir::Location loc,
    aiir::Type argType, aiir::Value scalarInitValue, aiir::Block *initBlock,
    aiir::Value allocatedPrivVarArg, aiir::Value moldArg,
    aiir::Region &cleanupRegion, DeclOperationKind kind,
    const Fortran::semantics::Symbol *sym, bool cannotHaveLowerBounds,
    bool isDoConcurrent) {
  PopulateInitAndCleanupRegionsHelper helper(
      converter, loc, argType, scalarInitValue, allocatedPrivVarArg, moldArg,
      initBlock, cleanupRegion, kind, sym, cannotHaveLowerBounds,
      isDoConcurrent);
  helper.populateByRefInitAndCleanupRegions();

  // Often we load moldArg to check something (e.g. length parameters, shape)
  // but then those answers can be gotten statically without accessing the
  // runtime value and so the only remaining use is a dead load. These loads can
  // force us to insert additional barriers and so should be avoided where
  // possible.
  if (moldArg.hasOneUse()) {
    aiir::Operation *user = *moldArg.getUsers().begin();
    if (auto load = aiir::dyn_cast<fir::LoadOp>(user))
      if (load.use_empty())
        load.erase();
  }
}
