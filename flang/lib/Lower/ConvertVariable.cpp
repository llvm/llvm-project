//===-- ConvertVariable.cpp -- bridge to lower to MLIR --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/BoxAnalyzer.h"
#include "flang/Lower/CUDA.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertConstant.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/ConvertProcedureDesignator.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Runtime/allocator-registry-consts.h"
#include "flang/Semantics/runtime-type-info.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <optional>

static llvm::cl::opt<bool>
    allowAssumedRank("allow-assumed-rank",
                     llvm::cl::desc("Enable assumed rank lowering"),
                     llvm::cl::init(true));

#define DEBUG_TYPE "flang-lower-variable"

/// Helper to lower a scalar expression using a specific symbol mapping.
static mlir::Value genScalarValue(Fortran::lower::AbstractConverter &converter,
                                  mlir::Location loc,
                                  const Fortran::lower::SomeExpr &expr,
                                  Fortran::lower::SymMap &symMap,
                                  Fortran::lower::StatementContext &context) {
  // This does not use the AbstractConverter member function to override the
  // symbol mapping to be used expression lowering.
  if (converter.getLoweringOptions().getLowerToHighLevelFIR()) {
    hlfir::EntityWithAttributes loweredExpr =
        Fortran::lower::convertExprToHLFIR(loc, converter, expr, symMap,
                                           context);
    return hlfir::loadTrivialScalar(loc, converter.getFirOpBuilder(),
                                    loweredExpr);
  }
  return fir::getBase(Fortran::lower::createSomeExtendedExpression(
      loc, converter, expr, symMap, context));
}

/// Does this variable have a default initialization?
bool Fortran::lower::hasDefaultInitialization(
    const Fortran::semantics::Symbol &sym) {
  if (sym.has<Fortran::semantics::ObjectEntityDetails>() && sym.size())
    if (!Fortran::semantics::IsAllocatableOrPointer(sym))
      if (const Fortran::semantics::DeclTypeSpec *declTypeSpec = sym.GetType())
        if (const Fortran::semantics::DerivedTypeSpec *derivedTypeSpec =
                declTypeSpec->AsDerived()) {
          // Pointer assignments in the runtime may hit undefined behaviors if
          // the RHS contains garbage. Pointer objects are always established by
          // lowering to NULL() (in Fortran::lower::createMutableBox). However,
          // pointer components need special care here so that local and global
          // derived type containing pointers are always initialized.
          // Intent(out), however, do not need to be initialized since the
          // related descriptor storage comes from a local or global that has
          // been initialized (it may not be NULL() anymore, but the rank, type,
          // and non deferred length parameters are still correct in a
          // conformant program, and that is what matters).
          const bool ignorePointer = Fortran::semantics::IsIntentOut(sym);
          return derivedTypeSpec->HasDefaultInitialization(
              /*ignoreAllocatable=*/false, ignorePointer);
        }
  return false;
}

// Does this variable have a finalization?
static bool hasFinalization(const Fortran::semantics::Symbol &sym) {
  if (sym.has<Fortran::semantics::ObjectEntityDetails>())
    if (const Fortran::semantics::DeclTypeSpec *declTypeSpec = sym.GetType())
      if (const Fortran::semantics::DerivedTypeSpec *derivedTypeSpec =
              declTypeSpec->AsDerived())
        return Fortran::semantics::IsFinalizable(*derivedTypeSpec);
  return false;
}

// Does this variable have an allocatable direct component?
static bool
hasAllocatableDirectComponent(const Fortran::semantics::Symbol &sym) {
  if (sym.has<Fortran::semantics::ObjectEntityDetails>())
    if (const Fortran::semantics::DeclTypeSpec *declTypeSpec = sym.GetType())
      if (const Fortran::semantics::DerivedTypeSpec *derivedTypeSpec =
              declTypeSpec->AsDerived())
        return Fortran::semantics::HasAllocatableDirectComponent(
            *derivedTypeSpec);
  return false;
}
//===----------------------------------------------------------------===//
// Global variables instantiation (not for alias and common)
//===----------------------------------------------------------------===//

/// Helper to generate expression value inside global initializer.
static fir::ExtendedValue
genInitializerExprValue(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc,
                        const Fortran::lower::SomeExpr &expr,
                        Fortran::lower::StatementContext &stmtCtx) {
  // Data initializer are constant value and should not depend on other symbols
  // given the front-end fold parameter references. In any case, the "current"
  // map of the converter should not be used since it holds mapping to
  // mlir::Value from another mlir region. If these value are used by accident
  // in the initializer, this will lead to segfaults in mlir code.
  Fortran::lower::SymMap emptyMap;
  return Fortran::lower::createSomeInitializerExpression(loc, converter, expr,
                                                         emptyMap, stmtCtx);
}

/// Can this symbol constant be placed in read-only memory?
static bool isConstant(const Fortran::semantics::Symbol &sym) {
  return sym.attrs().test(Fortran::semantics::Attr::PARAMETER) ||
         sym.test(Fortran::semantics::Symbol::Flag::ReadOnly);
}

/// Call \p genInit to generate code inside \p global initializer region.
static void
createGlobalInitialization(fir::FirOpBuilder &builder, fir::GlobalOp global,
                           std::function<void(fir::FirOpBuilder &)> genInit);

static mlir::Location genLocation(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::semantics::Symbol &sym) {
  // Compiler generated name cannot be used as source location, their name
  // is not pointing to the source files.
  if (!sym.test(Fortran::semantics::Symbol::Flag::CompilerCreated))
    return converter.genLocation(sym.name());
  return converter.getCurrentLocation();
}

/// Create the global op declaration without any initializer
static fir::GlobalOp declareGlobal(Fortran::lower::AbstractConverter &converter,
                                   const Fortran::lower::pft::Variable &var,
                                   llvm::StringRef globalName,
                                   mlir::StringAttr linkage) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  if (fir::GlobalOp global = builder.getNamedGlobal(globalName))
    return global;
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  cuf::DataAttributeAttr dataAttr =
      Fortran::lower::translateSymbolCUFDataAttribute(
          converter.getFirOpBuilder().getContext(), sym);
  // Always define linkonce data since it may be optimized out from the module
  // that actually owns the variable if it does not refers to it.
  if (linkage == builder.createLinkOnceODRLinkage() ||
      linkage == builder.createLinkOnceLinkage())
    return defineGlobal(converter, var, globalName, linkage, dataAttr);
  mlir::Location loc = genLocation(converter, sym);
  // Resolve potential host and module association before checking that this
  // symbol is an object of a function pointer.
  const Fortran::semantics::Symbol &ultimate = sym.GetUltimate();
  if (!ultimate.has<Fortran::semantics::ObjectEntityDetails>() &&
      !Fortran::semantics::IsProcedurePointer(ultimate))
    mlir::emitError(loc, "processing global declaration: symbol '")
        << toStringRef(sym.name()) << "' has unexpected details\n";
  return builder.createGlobal(loc, converter.genType(var), globalName, linkage,
                              mlir::Attribute{}, isConstant(ultimate),
                              var.isTarget(), dataAttr);
}

/// Temporary helper to catch todos in initial data target lowering.
static bool
hasDerivedTypeWithLengthParameters(const Fortran::semantics::Symbol &sym) {
  if (const Fortran::semantics::DeclTypeSpec *declTy = sym.GetType())
    if (const Fortran::semantics::DerivedTypeSpec *derived =
            declTy->AsDerived())
      return Fortran::semantics::CountLenParameters(*derived) > 0;
  return false;
}

fir::ExtendedValue Fortran::lower::genExtAddrInInitializer(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::lower::SomeExpr &addr) {
  Fortran::lower::SymMap globalOpSymMap;
  Fortran::lower::AggregateStoreMap storeMap;
  Fortran::lower::StatementContext stmtCtx;
  if (const Fortran::semantics::Symbol *sym =
          Fortran::evaluate::GetFirstSymbol(addr)) {
    // Length parameters processing will need care in global initializer
    // context.
    if (hasDerivedTypeWithLengthParameters(*sym))
      TODO(loc, "initial-data-target with derived type length parameters");

    auto var = Fortran::lower::pft::Variable(*sym, /*global=*/true);
    Fortran::lower::instantiateVariable(converter, var, globalOpSymMap,
                                        storeMap);
  }

  if (converter.getLoweringOptions().getLowerToHighLevelFIR())
    return Fortran::lower::convertExprToAddress(loc, converter, addr,
                                                globalOpSymMap, stmtCtx);
  return Fortran::lower::createInitializerAddress(loc, converter, addr,
                                                  globalOpSymMap, stmtCtx);
}

/// create initial-data-target fir.box in a global initializer region.
mlir::Value Fortran::lower::genInitialDataTarget(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Type boxType, const Fortran::lower::SomeExpr &initialTarget,
    bool couldBeInEquivalence) {
  Fortran::lower::SymMap globalOpSymMap;
  Fortran::lower::AggregateStoreMap storeMap;
  Fortran::lower::StatementContext stmtCtx;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
          initialTarget))
    return fir::factory::createUnallocatedBox(builder, loc, boxType,
                                              /*nonDeferredParams=*/{});
  // Pointer initial data target, and NULL(mold).
  for (const auto &sym : Fortran::evaluate::CollectSymbols(initialTarget)) {
    // Derived type component symbols should not be instantiated as objects
    // on their own.
    if (sym->owner().IsDerivedType())
      continue;
    // Length parameters processing will need care in global initializer
    // context.
    if (hasDerivedTypeWithLengthParameters(sym))
      TODO(loc, "initial-data-target with derived type length parameters");
    auto var = Fortran::lower::pft::Variable(sym, /*global=*/true);
    if (couldBeInEquivalence) {
      auto dependentVariableList =
          Fortran::lower::pft::getDependentVariableList(sym);
      for (Fortran::lower::pft::Variable var : dependentVariableList) {
        if (!var.isAggregateStore())
          break;
        instantiateVariable(converter, var, globalOpSymMap, storeMap);
      }
      var = dependentVariableList.back();
      assert(var.getSymbol().name() == sym->name() &&
             "missing symbol in dependence list");
    }
    Fortran::lower::instantiateVariable(converter, var, globalOpSymMap,
                                        storeMap);
  }

  // Handle NULL(mold) as a special case. Return an unallocated box of MOLD
  // type. The return box is correctly created as a fir.box<fir.ptr<T>> where
  // T is extracted from the MOLD argument.
  if (const Fortran::evaluate::ProcedureRef *procRef =
          Fortran::evaluate::UnwrapProcedureRef(initialTarget)) {
    const Fortran::evaluate::SpecificIntrinsic *intrinsic =
        procRef->proc().GetSpecificIntrinsic();
    if (intrinsic && intrinsic->name == "null") {
      assert(procRef->arguments().size() == 1 &&
             "Expecting mold argument for NULL intrinsic");
      const auto *argExpr = procRef->arguments()[0].value().UnwrapExpr();
      assert(argExpr);
      const Fortran::semantics::Symbol *sym =
          Fortran::evaluate::GetFirstSymbol(*argExpr);
      assert(sym && "MOLD must be a pointer or allocatable symbol");
      mlir::Type boxType = converter.genType(*sym);
      mlir::Value box =
          fir::factory::createUnallocatedBox(builder, loc, boxType, {});
      return box;
    }
  }

  mlir::Value targetBox;
  mlir::Value targetShift;
  if (converter.getLoweringOptions().getLowerToHighLevelFIR()) {
    auto target = Fortran::lower::convertExprToBox(
        loc, converter, initialTarget, globalOpSymMap, stmtCtx);
    targetBox = fir::getBase(target);
    targetShift = builder.createShape(loc, target);
  } else {
    if (initialTarget.Rank() > 0) {
      auto target = Fortran::lower::createSomeArrayBox(converter, initialTarget,
                                                       globalOpSymMap, stmtCtx);
      targetBox = fir::getBase(target);
      targetShift = builder.createShape(loc, target);
    } else {
      fir::ExtendedValue addr = Fortran::lower::createInitializerAddress(
          loc, converter, initialTarget, globalOpSymMap, stmtCtx);
      targetBox = builder.createBox(loc, addr);
      // Nothing to do for targetShift, the target is a scalar.
    }
  }
  // The targetBox is a fir.box<T>, not a fir.box<fir.ptr<T>> as it should for
  // pointers (this matters to get the POINTER attribute correctly inside the
  // initial value of the descriptor).
  // Create a fir.rebox to set the attribute correctly, and use targetShift
  // to preserve the target lower bounds if any.
  return fir::ReboxOp::create(builder, loc, boxType, targetBox, targetShift,
                              /*slice=*/mlir::Value{});
}

/// Generate default initial value for a derived type object \p sym with mlir
/// type \p symTy.
static mlir::Value genDefaultInitializerValue(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::semantics::Symbol &sym, mlir::Type symTy,
    Fortran::lower::StatementContext &stmtCtx);

/// Generate the initial value of a derived component \p component and insert
/// it into the derived type initial value \p insertInto of type \p recTy.
/// Return the new derived type initial value after the insertion.
static mlir::Value genComponentDefaultInit(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::semantics::Symbol &component, fir::RecordType recTy,
    mlir::Value insertInto, Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  std::string name = converter.getRecordTypeFieldName(component);
  mlir::Type componentTy = recTy.getType(name);
  assert(componentTy && "component not found in type");
  mlir::Value componentValue;
  if (const auto *object{
          component.detailsIf<Fortran::semantics::ObjectEntityDetails>()}) {
    if (const auto &init = object->init()) {
      // Component has explicit initialization.
      if (Fortran::semantics::IsPointer(component))
        // Initial data target.
        componentValue =
            genInitialDataTarget(converter, loc, componentTy, *init);
      else
        // Initial value.
        componentValue = fir::getBase(
            genInitializerExprValue(converter, loc, *init, stmtCtx));
    } else if (Fortran::semantics::IsAllocatableOrPointer(component)) {
      // Pointer or allocatable without initialization.
      // Create deallocated/disassociated value.
      // From a standard point of view, pointer without initialization do not
      // need to be disassociated, but for sanity and simplicity, do it in
      // global constructor since this has no runtime cost.
      componentValue =
          fir::factory::createUnallocatedBox(builder, loc, componentTy, {});
    } else if (Fortran::lower::hasDefaultInitialization(component)) {
      // Component type has default initialization.
      componentValue = genDefaultInitializerValue(converter, loc, component,
                                                  componentTy, stmtCtx);
    } else {
      // Component has no initial value. Set its bits to zero by extension
      // to match what is expected because other compilers are doing it.
      componentValue = fir::ZeroOp::create(builder, loc, componentTy);
    }
  } else if (const auto *proc{
                 component
                     .detailsIf<Fortran::semantics::ProcEntityDetails>()}) {
    if (proc->init().has_value()) {
      auto sym{*proc->init()};
      if (sym) // Has a procedure target.
        componentValue =
            Fortran::lower::convertProcedureDesignatorInitialTarget(converter,
                                                                    loc, *sym);
      else // Has NULL() target.
        componentValue =
            fir::factory::createNullBoxProc(builder, loc, componentTy);
    } else
      componentValue = fir::ZeroOp::create(builder, loc, componentTy);
  }
  assert(componentValue && "must have been computed");
  componentValue = builder.createConvert(loc, componentTy, componentValue);
  auto fieldTy = fir::FieldType::get(recTy.getContext());
  // FIXME: type parameters must come from the derived-type-spec
  auto field =
      fir::FieldIndexOp::create(builder, loc, fieldTy, name, recTy,
                                /*typeParams=*/mlir::ValueRange{} /*TODO*/);
  return fir::InsertValueOp::create(
      builder, loc, recTy, insertInto, componentValue,
      builder.getArrayAttr(field.getAttributes()));
}

static mlir::Value genDefaultInitializerValue(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::semantics::Symbol &sym, mlir::Type symTy,
    Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Type scalarType = symTy;
  fir::SequenceType sequenceType;
  if (auto ty = mlir::dyn_cast<fir::SequenceType>(symTy)) {
    sequenceType = ty;
    scalarType = ty.getEleTy();
  }
  // Build a scalar default value of the symbol type, looping through the
  // components to build each component initial value.
  auto recTy = mlir::cast<fir::RecordType>(scalarType);
  mlir::Value initialValue = fir::UndefOp::create(builder, loc, scalarType);
  const Fortran::semantics::DeclTypeSpec *declTy = sym.GetType();
  assert(declTy && "var with default initialization must have a type");

  if (converter.getLoweringOptions().getLowerToHighLevelFIR()) {
    // In HLFIR, the parent type is the first component, while in FIR there is
    // not parent component in the fir.type and the component of the parent are
    // "inlined" at the beginning of the fir.type.
    const Fortran::semantics::Symbol &typeSymbol =
        declTy->derivedTypeSpec().typeSymbol();
    const Fortran::semantics::Scope *derivedScope =
        declTy->derivedTypeSpec().GetScope();
    assert(derivedScope && "failed to retrieve derived type scope");
    for (const auto &componentName :
         typeSymbol.get<Fortran::semantics::DerivedTypeDetails>()
             .componentNames()) {
      auto scopeIter = derivedScope->find(componentName);
      assert(scopeIter != derivedScope->cend() &&
             "failed to find derived type component symbol");
      const Fortran::semantics::Symbol &component = scopeIter->second.get();
      initialValue = genComponentDefaultInit(converter, loc, component, recTy,
                                             initialValue, stmtCtx);
    }
  } else {
    Fortran::semantics::OrderedComponentIterator components(
        declTy->derivedTypeSpec());
    for (const auto &component : components) {
      // Skip parent components, the sub-components of parent types are part of
      // components and will be looped through right after.
      if (component.test(Fortran::semantics::Symbol::Flag::ParentComp))
        continue;
      initialValue = genComponentDefaultInit(converter, loc, component, recTy,
                                             initialValue, stmtCtx);
    }
  }

  if (sequenceType) {
    // For arrays, duplicate the scalar value to all elements with an
    // fir.insert_range covering the whole array.
    auto arrayInitialValue = fir::UndefOp::create(builder, loc, sequenceType);
    llvm::SmallVector<int64_t> rangeBounds;
    for (int64_t extent : sequenceType.getShape()) {
      if (extent == fir::SequenceType::getUnknownExtent())
        TODO(loc,
             "default initial value of array component with length parameters");
      rangeBounds.push_back(0);
      rangeBounds.push_back(extent - 1);
    }
    return fir::InsertOnRangeOp::create(
        builder, loc, sequenceType, arrayInitialValue, initialValue,
        builder.getIndexVectorAttr(rangeBounds));
  }
  return initialValue;
}

/// Does this global already have an initializer ?
static bool globalIsInitialized(fir::GlobalOp global) {
  return !global.getRegion().empty() || global.getInitVal();
}

/// Call \p genInit to generate code inside \p global initializer region.
static void
createGlobalInitialization(fir::FirOpBuilder &builder, fir::GlobalOp global,
                           std::function<void(fir::FirOpBuilder &)> genInit) {
  mlir::Region &region = global.getRegion();
  region.push_back(new mlir::Block);
  mlir::Block &block = region.back();
  auto insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&block);
  genInit(builder);
  builder.restoreInsertionPoint(insertPt);
}

static unsigned getAllocatorIdxFromDataAttr(cuf::DataAttributeAttr dataAttr) {
  if (dataAttr) {
    if (dataAttr.getValue() == cuf::DataAttribute::Pinned)
      return kPinnedAllocatorPos;
    if (dataAttr.getValue() == cuf::DataAttribute::Device)
      return kDeviceAllocatorPos;
    if (dataAttr.getValue() == cuf::DataAttribute::Managed)
      return kManagedAllocatorPos;
    if (dataAttr.getValue() == cuf::DataAttribute::Unified)
      return kUnifiedAllocatorPos;
  }
  return kDefaultAllocator;
}

/// Create the global op and its init if it has one
fir::GlobalOp Fortran::lower::defineGlobal(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable &var, llvm::StringRef globalName,
    mlir::StringAttr linkage, cuf::DataAttributeAttr dataAttr) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  mlir::Location loc = genLocation(converter, sym);
  bool isConst = isConstant(sym);
  fir::GlobalOp global = builder.getNamedGlobal(globalName);
  mlir::Type symTy = converter.genType(var);

  if (global && globalIsInitialized(global))
    return global;

  if (!converter.getLoweringOptions().getLowerToHighLevelFIR() &&
      Fortran::semantics::IsProcedurePointer(sym))
    TODO(loc, "procedure pointer globals");

  // If this is an array, check to see if we can use a dense attribute
  // with a tensor mlir type. This optimization currently only supports
  // Fortran arrays of integer, real, complex, or logical. The tensor
  // type does not support nested structures.
  if (mlir::isa<fir::SequenceType>(symTy) &&
      !Fortran::semantics::IsAllocatableOrPointer(sym)) {
    mlir::Type eleTy = mlir::cast<fir::SequenceType>(symTy).getElementType();
    if (mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::ComplexType,
                  fir::LogicalType>(eleTy)) {
      const auto *details =
          sym.detailsIf<Fortran::semantics::ObjectEntityDetails>();
      if (details->init()) {
        global = Fortran::lower::tryCreatingDenseGlobal(
            builder, loc, symTy, globalName, linkage, isConst,
            details->init().value(), dataAttr);
        if (global) {
          global.setVisibility(mlir::SymbolTable::Visibility::Public);
          return global;
        }
      }
    }
  }
  if (!global)
    global =
        builder.createGlobal(loc, symTy, globalName, linkage, mlir::Attribute{},
                             isConst, var.isTarget(), dataAttr);
  if (Fortran::semantics::IsAllocatableOrPointer(sym) &&
      !Fortran::semantics::IsProcedure(sym)) {
    const auto *details =
        sym.detailsIf<Fortran::semantics::ObjectEntityDetails>();
    if (details && details->init()) {
      auto expr = *details->init();
      createGlobalInitialization(builder, global, [&](fir::FirOpBuilder &b) {
        mlir::Value box =
            Fortran::lower::genInitialDataTarget(converter, loc, symTy, expr);
        fir::HasValueOp::create(b, loc, box);
      });
    } else {
      // Create unallocated/disassociated descriptor if no explicit init
      createGlobalInitialization(builder, global, [&](fir::FirOpBuilder &b) {
        mlir::Value box = fir::factory::createUnallocatedBox(
            b, loc, symTy,
            /*nonDeferredParams=*/{},
            /*typeSourceBox=*/{}, getAllocatorIdxFromDataAttr(dataAttr));
        fir::HasValueOp::create(b, loc, box);
      });
    }
  } else if (const auto *details =
                 sym.detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
    if (details->init()) {
      createGlobalInitialization(
          builder, global, [&](fir::FirOpBuilder &builder) {
            Fortran::lower::StatementContext stmtCtx(
                /*cleanupProhibited=*/true);
            fir::ExtendedValue initVal = genInitializerExprValue(
                converter, loc, details->init().value(), stmtCtx);
            mlir::Value castTo =
                builder.createConvert(loc, symTy, fir::getBase(initVal));
            fir::HasValueOp::create(builder, loc, castTo);
          });
    } else if (Fortran::lower::hasDefaultInitialization(sym)) {
      createGlobalInitialization(
          builder, global, [&](fir::FirOpBuilder &builder) {
            Fortran::lower::StatementContext stmtCtx(
                /*cleanupProhibited=*/true);
            mlir::Value initVal =
                genDefaultInitializerValue(converter, loc, sym, symTy, stmtCtx);
            mlir::Value castTo = builder.createConvert(loc, symTy, initVal);
            fir::HasValueOp::create(builder, loc, castTo);
          });
    }
  } else if (Fortran::semantics::IsProcedurePointer(sym)) {
    const auto *details{sym.detailsIf<Fortran::semantics::ProcEntityDetails>()};
    if (details && details->init()) {
      auto sym{*details->init()};
      if (sym) // Has a procedure target.
        createGlobalInitialization(
            builder, global, [&](fir::FirOpBuilder &b) {
              Fortran::lower::StatementContext stmtCtx(
                  /*cleanupProhibited=*/true);
              auto box{Fortran::lower::convertProcedureDesignatorInitialTarget(
                  converter, loc, *sym)};
              auto castTo{builder.createConvert(loc, symTy, box)};
              fir::HasValueOp::create(b, loc, castTo);
            });
      else { // Has NULL() target.
        createGlobalInitialization(builder, global, [&](fir::FirOpBuilder &b) {
          auto box{fir::factory::createNullBoxProc(b, loc, symTy)};
          fir::HasValueOp::create(b, loc, box);
        });
      }
    } else {
      // No initialization.
      createGlobalInitialization(builder, global, [&](fir::FirOpBuilder &b) {
        auto box{fir::factory::createNullBoxProc(b, loc, symTy)};
        fir::HasValueOp::create(b, loc, box);
      });
    }
  } else if (sym.has<Fortran::semantics::CommonBlockDetails>()) {
    mlir::emitError(loc, "COMMON symbol processed elsewhere");
  } else {
    TODO(loc, "global"); // Something else
  }
  // Creates zero initializer for globals without initializers, this is a common
  // and expected behavior (although not required by the standard)
  if (!globalIsInitialized(global)) {
    // Fortran does not provide means to specify that a BIND(C) module
    // uninitialized variables will be defined in C.
    // Add the common linkage to those to allow some level of support
    // for this use case. Note that this use case will not work if the Fortran
    // module code is placed in a shared library since, at least for the ELF
    // format, common symbols are assigned a section in shared libraries.
    // The best is still to declare C defined variables in a Fortran module file
    // with no other definitions, and to never link the resulting module object
    // file.
    if (sym.attrs().test(Fortran::semantics::Attr::BIND_C))
      global.setLinkName(builder.createCommonLinkage());
    createGlobalInitialization(
        builder, global, [&](fir::FirOpBuilder &builder) {
          mlir::Value initValue;
          if (converter.getLoweringOptions().getInitGlobalZero())
            initValue = fir::ZeroOp::create(builder, loc, symTy);
          else
            initValue = fir::UndefOp::create(builder, loc, symTy);
          fir::HasValueOp::create(builder, loc, initValue);
        });
  }
  // Set public visibility to prevent global definition to be optimized out
  // even if they have no initializer and are unused in this compilation unit.
  global.setVisibility(mlir::SymbolTable::Visibility::Public);
  return global;
}

/// Return linkage attribute for \p var.
static mlir::StringAttr
getLinkageAttribute(Fortran::lower::AbstractConverter &converter,
                    const Fortran::lower::pft::Variable &var) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  // Runtime type info for a same derived type is identical in each compilation
  // unit. It desired to avoid having to link against module that only define a
  // type. Therefore the runtime type info is generated everywhere it is needed
  // with `linkonce_odr` LLVM linkage (unless the skipExternalRttiDefinition
  // option is set, in which case one will need to link against objects of
  // modules defining types). Builtin objects rtti is always generated because
  // the builtin module is currently not compiled or part of the runtime.
  if (var.isRuntimeTypeInfoData() &&
      (!converter.getLoweringOptions().getSkipExternalRttiDefinition() ||
       Fortran::semantics::IsFromBuiltinModule(var.getSymbol())))
    return builder.createLinkOnceODRLinkage();
  if (var.isModuleOrSubmoduleVariable())
    return {}; // external linkage
  // Otherwise, the variable is owned by a procedure and must not be visible in
  // other compilation units.
  return builder.createInternalLinkage();
}

/// Instantiate a global variable. If it hasn't already been processed, add
/// the global to the ModuleOp as a new uniqued symbol and initialize it with
/// the correct value. It will be referenced on demand using `fir.addr_of`.
static void instantiateGlobal(Fortran::lower::AbstractConverter &converter,
                              const Fortran::lower::pft::Variable &var,
                              Fortran::lower::SymMap &symMap) {
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  assert(!var.isAlias() && "must be handled in instantiateAlias");
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  std::string globalName = converter.mangleName(sym);
  mlir::Location loc = genLocation(converter, sym);
  mlir::StringAttr linkage = getLinkageAttribute(converter, var);
  fir::GlobalOp global;
  if (var.isModuleOrSubmoduleVariable()) {
    // A non-intrinsic module global is defined when lowering the module.
    // Emit only a declaration if the global does not exist.
    global = declareGlobal(converter, var, globalName, linkage);
  } else {
    cuf::DataAttributeAttr dataAttr =
        Fortran::lower::translateSymbolCUFDataAttribute(builder.getContext(),
                                                        sym);
    global = defineGlobal(converter, var, globalName, linkage, dataAttr);
  }
  auto addrOf = fir::AddrOfOp::create(builder, loc, global.resultType(),
                                      global.getSymbol());
  // The type of the global cannot be trusted to be the same as the one
  // of the variable as some existing programs map common blocks to
  // BIND(C) module variables (e.g. mpi_argv_null in MPI and MPI_F08).
  mlir::Type varAddrType = fir::ReferenceType::get(converter.genType(sym));
  mlir::Value cast = builder.createConvert(loc, varAddrType, addrOf);
  Fortran::lower::StatementContext stmtCtx;
  mapSymbolAttributes(converter, var, symMap, stmtCtx, cast);
}

bool needCUDAAlloc(const Fortran::semantics::Symbol &sym) {
  if (Fortran::semantics::IsDummy(sym))
    return false;
  if (const auto *details{
          sym.GetUltimate()
              .detailsIf<Fortran::semantics::ObjectEntityDetails>()}) {
    if (details->cudaDataAttr() &&
        (*details->cudaDataAttr() == Fortran::common::CUDADataAttr::Device ||
         *details->cudaDataAttr() == Fortran::common::CUDADataAttr::Managed ||
         *details->cudaDataAttr() == Fortran::common::CUDADataAttr::Unified ||
         *details->cudaDataAttr() == Fortran::common::CUDADataAttr::Shared ||
         *details->cudaDataAttr() == Fortran::common::CUDADataAttr::Pinned))
      return true;
    const Fortran::semantics::DeclTypeSpec *type{details->type()};
    const Fortran::semantics::DerivedTypeSpec *derived{type ? type->AsDerived()
                                                            : nullptr};
    if (derived)
      if (FindCUDADeviceAllocatableUltimateComponent(*derived))
        return true;
  }
  return false;
}

//===----------------------------------------------------------------===//
// Local variables instantiation (not for alias)
//===----------------------------------------------------------------===//

/// Create a stack slot for a local variable. Precondition: the insertion
/// point of the builder must be in the entry block, which is currently being
/// constructed.
static mlir::Value createNewLocal(Fortran::lower::AbstractConverter &converter,
                                  mlir::Location loc,
                                  const Fortran::lower::pft::Variable &var,
                                  mlir::Value preAlloc,
                                  llvm::ArrayRef<mlir::Value> shape = {},
                                  llvm::ArrayRef<mlir::Value> lenParams = {}) {
  if (preAlloc)
    return preAlloc;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  std::string nm = converter.mangleName(var.getSymbol());
  mlir::Type ty = converter.genType(var);
  const Fortran::semantics::Symbol &ultimateSymbol =
      var.getSymbol().GetUltimate();
  llvm::StringRef symNm = toStringRef(ultimateSymbol.name());
  bool isTarg = var.isTarget();

  // Do not allocate storage for cray pointee. The address inside the cray
  // pointer will be used instead when using the pointee. Allocating space
  // would be a waste of space, and incorrect if the pointee is a non dummy
  // assumed-size (possible with cray pointee).
  if (ultimateSymbol.test(Fortran::semantics::Symbol::Flag::CrayPointee))
    return fir::ZeroOp::create(builder, loc, fir::ReferenceType::get(ty));

  if (needCUDAAlloc(ultimateSymbol)) {
    cuf::DataAttributeAttr dataAttr =
        Fortran::lower::translateSymbolCUFDataAttribute(builder.getContext(),
                                                        ultimateSymbol);
    llvm::SmallVector<mlir::Value> indices;
    llvm::SmallVector<mlir::Value> elidedShape =
        fir::factory::elideExtentsAlreadyInType(ty, shape);
    llvm::SmallVector<mlir::Value> elidedLenParams =
        fir::factory::elideLengthsAlreadyInType(ty, lenParams);
    auto idxTy = builder.getIndexType();
    for (mlir::Value sh : elidedShape)
      indices.push_back(builder.createConvert(loc, idxTy, sh));
    if (dataAttr.getValue() == cuf::DataAttribute::Shared)
      return cuf::SharedMemoryOp::create(builder, loc, ty, nm, symNm, lenParams,
                                         indices);

    if (!cuf::isCUDADeviceContext(builder.getRegion()))
      return cuf::AllocOp::create(builder, loc, ty, nm, symNm, dataAttr,
                                  lenParams, indices);
  }

  // Let the builder do all the heavy lifting.
  if (!Fortran::semantics::IsProcedurePointer(ultimateSymbol))
    return builder.allocateLocal(loc, ty, nm, symNm, shape, lenParams, isTarg);

  // Local procedure pointer.
  auto res{builder.allocateLocal(loc, ty, nm, symNm, shape, lenParams, isTarg)};
  auto box{fir::factory::createNullBoxProc(builder, loc, ty)};
  fir::StoreOp::create(builder, loc, box, res);
  return res;
}

/// Device allocatable components in a derived-type don't have the correct
/// allocator index in their descriptor when they are created. After
/// initialization, cuf.set_allocator_idx operations are inserted to set the
/// correct allocator index for each device component.
static void
initializeDeviceComponentAllocator(Fortran::lower::AbstractConverter &converter,
                                   const Fortran::semantics::Symbol &symbol,
                                   Fortran::lower::SymMap &symMap) {
  if (const auto *details{
          symbol.GetUltimate()
              .detailsIf<Fortran::semantics::ObjectEntityDetails>()}) {
    const Fortran::semantics::DeclTypeSpec *type{details->type()};
    const Fortran::semantics::DerivedTypeSpec *derived{type ? type->AsDerived()
                                                            : nullptr};
    if (derived) {
      if (!FindCUDADeviceAllocatableUltimateComponent(*derived))
        return; // No device components.

      fir::FirOpBuilder &builder = converter.getFirOpBuilder();
      mlir::Location loc = converter.getCurrentLocation();

      fir::ExtendedValue exv =
          converter.getSymbolExtendedValue(symbol.GetUltimate(), &symMap);
      mlir::Type baseTy = fir::unwrapRefType(fir::getBase(exv).getType());
      if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(baseTy))
        baseTy = boxTy.getEleTy();
      baseTy = fir::unwrapRefType(baseTy);

      if (fir::isAllocatableType(fir::getBase(exv).getType()) ||
          fir::isPointerType(fir::getBase(exv).getType()))
        return; // Allocator index need to be set after allocation.

      auto recTy =
          mlir::dyn_cast<fir::RecordType>(fir::unwrapSequenceType(baseTy));
      assert(recTy && "expected fir::RecordType");

      Fortran::semantics::UltimateComponentIterator components{*derived};
      for (const auto &sym : components) {
        if (Fortran::semantics::IsDeviceAllocatable(sym)) {
          llvm::SmallVector<mlir::Value> coord;
          mlir::Type fieldTy =
              Fortran::lower::gatherDeviceComponentCoordinatesAndType(
                  builder, loc, sym, recTy, coord);
          mlir::Value base = fir::getBase(exv);
          mlir::Value comp = fir::CoordinateOp::create(
              builder, loc, builder.getRefType(fieldTy), base, coord);
          cuf::DataAttributeAttr dataAttr =
              Fortran::lower::translateSymbolCUFDataAttribute(
                  builder.getContext(), sym);
          cuf::SetAllocatorIndexOp::create(builder, loc, comp, dataAttr);
        }
      }
    }
  }
}

/// Must \p var be default initialized at runtime when entering its scope.
static bool
mustBeDefaultInitializedAtRuntime(const Fortran::lower::pft::Variable &var) {
  if (!var.hasSymbol())
    return false;
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  if (var.isGlobal())
    // Global variables are statically initialized.
    return false;
  if (Fortran::semantics::IsDummy(sym) && !Fortran::semantics::IsIntentOut(sym))
    return false;
  // Polymorphic intent(out) dummy might need default initialization
  // at runtime.
  if (Fortran::semantics::IsPolymorphic(sym) &&
      Fortran::semantics::IsDummy(sym) &&
      Fortran::semantics::IsIntentOut(sym) &&
      !Fortran::semantics::IsAllocatable(sym) &&
      !Fortran::semantics::IsPointer(sym))
    return true;
  // Local variables (including function results), and intent(out) dummies must
  // be default initialized at runtime if their type has default initialization.
  return Fortran::lower::hasDefaultInitialization(sym);
}

/// Call default initialization runtime routine to initialize \p var.
void Fortran::lower::defaultInitializeAtRuntime(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::semantics::Symbol &sym, Fortran::lower::SymMap &symMap) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  fir::ExtendedValue exv = converter.getSymbolExtendedValue(sym, &symMap);
  if (Fortran::semantics::IsOptional(sym)) {
    // 15.5.2.12 point 3, absent optional dummies are not initialized.
    // Creating descriptor/passing null descriptor to the runtime would
    // create runtime crashes.
    auto isPresent = fir::IsPresentOp::create(builder, loc, builder.getI1Type(),
                                              fir::getBase(exv));
    builder.genIfThen(loc, isPresent)
        .genThen([&]() {
          auto box = builder.createBox(loc, exv);
          fir::runtime::genDerivedTypeInitialize(builder, loc, box);
        })
        .end();
  } else {
    /// For "simpler" types, relying on "_FortranAInitialize"
    /// leads to poor runtime performance. Hence optimize
    /// the same.
    const Fortran::semantics::DeclTypeSpec *declTy = sym.GetType();
    mlir::Type symTy = converter.genType(sym);
    const auto *details =
        sym.detailsIf<Fortran::semantics::ObjectEntityDetails>();
    if (details && !Fortran::semantics::IsPolymorphic(sym) &&
        declTy->category() ==
            Fortran::semantics::DeclTypeSpec::Category::TypeDerived &&
        !mlir::isa<fir::SequenceType>(symTy) &&
        !sym.test(Fortran::semantics::Symbol::Flag::OmpPrivate) &&
        !sym.test(Fortran::semantics::Symbol::Flag::OmpFirstPrivate)) {
      std::string globalName = fir::NameUniquer::doGenerated(
          (converter.mangleName(*declTy->AsDerived()) + fir::kNameSeparator +
           fir::kDerivedTypeInitSuffix)
              .str());
      mlir::Location loc = genLocation(converter, sym);
      mlir::StringAttr linkage = builder.createInternalLinkage();
      fir::GlobalOp global = builder.getNamedGlobal(globalName);
      if (!global && details->init()) {
        global = builder.createGlobal(loc, symTy, globalName, linkage,
                                      mlir::Attribute{},
                                      /*isConst=*/true,
                                      /*isTarget=*/false,
                                      /*dataAttr=*/{});
        createGlobalInitialization(
            builder, global, [&](fir::FirOpBuilder &builder) {
              Fortran::lower::StatementContext stmtCtx(
                  /*cleanupProhibited=*/true);
              fir::ExtendedValue initVal = genInitializerExprValue(
                  converter, loc, details->init().value(), stmtCtx);
              mlir::Value castTo =
                  builder.createConvert(loc, symTy, fir::getBase(initVal));
              fir::HasValueOp::create(builder, loc, castTo);
            });
      } else if (!global) {
        global = builder.createGlobal(loc, symTy, globalName, linkage,
                                      mlir::Attribute{},
                                      /*isConst=*/true,
                                      /*isTarget=*/false,
                                      /*dataAttr=*/{});
        createGlobalInitialization(
            builder, global, [&](fir::FirOpBuilder &builder) {
              Fortran::lower::StatementContext stmtCtx(
                  /*cleanupProhibited=*/true);
              mlir::Value initVal = genDefaultInitializerValue(
                  converter, loc, sym, symTy, stmtCtx);
              mlir::Value castTo = builder.createConvert(loc, symTy, initVal);
              fir::HasValueOp::create(builder, loc, castTo);
            });
      }
      auto addrOf = fir::AddrOfOp::create(builder, loc, global.resultType(),
                                          global.getSymbol());
      fir::CopyOp::create(builder, loc, addrOf, fir::getBase(exv),
                          /*noOverlap=*/true);
    } else {
      mlir::Value box = builder.createBox(loc, exv);
      fir::runtime::genDerivedTypeInitialize(builder, loc, box);
    }
  }
}

/// Call clone initialization runtime routine to initialize \p sym's value.
void Fortran::lower::initializeCloneAtRuntime(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::semantics::Symbol &sym, Fortran::lower::SymMap &symMap) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  fir::ExtendedValue exv = converter.getSymbolExtendedValue(sym, &symMap);
  mlir::Value newBox = builder.createBox(loc, exv);
  lower::SymbolBox hsb = converter.lookupOneLevelUpSymbol(sym);
  fir::ExtendedValue hexv = converter.symBoxToExtendedValue(hsb);
  mlir::Value box = builder.createBox(loc, hexv);
  fir::runtime::genDerivedTypeInitializeClone(builder, loc, newBox, box);
}

enum class VariableCleanUp { Finalize, Deallocate };
/// Check whether a local variable needs to be finalized according to clause
/// 7.5.6.3 point 3 or if it is an allocatable that must be deallocated. Note
/// that deallocation will trigger finalization if the type has any.
static std::optional<VariableCleanUp>
needDeallocationOrFinalization(const Fortran::lower::pft::Variable &var) {
  if (!var.hasSymbol())
    return std::nullopt;
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  const Fortran::semantics::Scope &owner = sym.owner();
  if (owner.kind() == Fortran::semantics::Scope::Kind::MainProgram) {
    // The standard does not require finalizing main program variables.
    return std::nullopt;
  }
  if (!Fortran::semantics::IsPointer(sym) &&
      !Fortran::semantics::IsDummy(sym) &&
      !Fortran::semantics::IsFunctionResult(sym) &&
      !Fortran::semantics::IsSaved(sym)) {
    if (Fortran::semantics::IsAllocatable(sym))
      return VariableCleanUp::Deallocate;
    if (hasFinalization(sym))
      return VariableCleanUp::Finalize;
    // hasFinalization() check above handled all cases that require
    // finalization, but we also have to deallocate all allocatable
    // components of local variables (since they are also local variables
    // according to F18 5.4.3.2.2, p. 2, note 1).
    // Here, the variable itself is not allocatable. If it has an allocatable
    // component the Destroy runtime does the job. Use the Finalize clean-up,
    // though there will be no finalization in runtime.
    if (hasAllocatableDirectComponent(sym))
      return VariableCleanUp::Finalize;
  }
  return std::nullopt;
}

/// Check whether a variable needs the be finalized according to clause 7.5.6.3
/// point 7.
/// Must be nonpointer, nonallocatable, INTENT (OUT) dummy argument.
static bool
needDummyIntentoutFinalization(const Fortran::semantics::Symbol &sym) {
  if (!Fortran::semantics::IsDummy(sym) ||
      !Fortran::semantics::IsIntentOut(sym) ||
      Fortran::semantics::IsAllocatable(sym) ||
      Fortran::semantics::IsPointer(sym))
    return false;
  // Polymorphic and unlimited polymorphic intent(out) dummy argument might need
  // finalization at runtime.
  if (Fortran::semantics::IsPolymorphic(sym) ||
      Fortran::semantics::IsUnlimitedPolymorphic(sym))
    return true;
  // Intent(out) dummies must be finalized at runtime if their type has a
  // finalization.
  // Allocatable components of INTENT(OUT) dummies must be deallocated (9.7.3.2
  // p6). Calling finalization runtime for this works even if the components
  // have no final procedures.
  return hasFinalization(sym) || hasAllocatableDirectComponent(sym);
}

/// Check whether a variable needs the be finalized according to clause 7.5.6.3
/// point 7.
/// Must be nonpointer, nonallocatable, INTENT (OUT) dummy argument.
static bool
needDummyIntentoutFinalization(const Fortran::lower::pft::Variable &var) {
  if (!var.hasSymbol())
    return false;
  return needDummyIntentoutFinalization(var.getSymbol());
}

/// Call default initialization runtime routine to initialize \p var.
static void finalizeAtRuntime(Fortran::lower::AbstractConverter &converter,
                              const Fortran::lower::pft::Variable &var,
                              Fortran::lower::SymMap &symMap) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  fir::ExtendedValue exv = converter.getSymbolExtendedValue(sym, &symMap);
  if (Fortran::semantics::IsOptional(sym)) {
    // Only finalize if present.
    auto isPresent = fir::IsPresentOp::create(builder, loc, builder.getI1Type(),
                                              fir::getBase(exv));
    builder.genIfThen(loc, isPresent)
        .genThen([&]() {
          auto box = builder.createBox(loc, exv);
          fir::runtime::genDerivedTypeDestroy(builder, loc, box);
        })
        .end();
  } else {
    mlir::Value box = builder.createBox(loc, exv);
    fir::runtime::genDerivedTypeDestroy(builder, loc, box);
  }
}

// Fortran 2018 - 9.7.3.2 point 6
// When a procedure is invoked, any allocated allocatable object that is an
// actual argument corresponding to an INTENT(OUT) allocatable dummy argument
// is deallocated; any allocated allocatable object that is a subobject of an
// actual argument corresponding to an INTENT(OUT) dummy argument is
// deallocated.
// Note that allocatable components of non-ALLOCATABLE INTENT(OUT) dummy
// arguments are dealt with needDummyIntentoutFinalization (finalization runtime
// is called to reach the intended component deallocation effect).
static void deallocateIntentOut(Fortran::lower::AbstractConverter &converter,
                                const Fortran::lower::pft::Variable &var,
                                Fortran::lower::SymMap &symMap) {
  if (!var.hasSymbol())
    return;

  const Fortran::semantics::Symbol &sym = var.getSymbol();
  if (Fortran::semantics::IsDummy(sym) &&
      Fortran::semantics::IsIntentOut(sym) &&
      Fortran::semantics::IsAllocatable(sym)) {
    fir::ExtendedValue extVal = converter.getSymbolExtendedValue(sym, &symMap);
    if (auto mutBox = extVal.getBoxOf<fir::MutableBoxValue>()) {
      // The dummy argument is not passed in the ENTRY so it should not be
      // deallocated.
      if (mlir::Operation *op = mutBox->getAddr().getDefiningOp()) {
        if (auto declOp = mlir::dyn_cast<hlfir::DeclareOp>(op))
          op = declOp.getMemref().getDefiningOp();
        if (op && mlir::isa<fir::AllocaOp>(op))
          return;
      }
      mlir::Location loc = converter.getCurrentLocation();
      fir::FirOpBuilder &builder = converter.getFirOpBuilder();

      if (Fortran::semantics::IsOptional(sym)) {
        auto isPresent = fir::IsPresentOp::create(
            builder, loc, builder.getI1Type(), fir::getBase(extVal));
        builder.genIfThen(loc, isPresent)
            .genThen([&]() {
              Fortran::lower::genDeallocateIfAllocated(converter, *mutBox, loc);
            })
            .end();
      } else {
        Fortran::lower::genDeallocateIfAllocated(converter, *mutBox, loc);
      }
    }
  }
}

/// Return true iff the given symbol represents a dummy array
/// that needs to be repacked when -frepack-arrays is set.
/// In general, the repacking is done for assumed-shape
/// dummy arguments, but there are limitations.
static bool needsRepack(Fortran::lower::AbstractConverter &converter,
                        const Fortran::semantics::Symbol &sym) {
  const auto &attrs = sym.attrs();
  if (!converter.getLoweringOptions().getRepackArrays() ||
      !converter.isRegisteredDummySymbol(sym) ||
      !Fortran::semantics::IsAssumedShape(sym) ||
      Fortran::evaluate::IsSimplyContiguous(sym,
                                            converter.getFoldingContext()) ||
      // TARGET dummy may be accessed indirectly, so it is unsafe
      // to repack it. Some compilers provide options to override
      // this.
      // Repacking of VOLATILE and ASYNCHRONOUS is also unsafe.
      attrs.HasAny({Fortran::semantics::Attr::ASYNCHRONOUS,
                    Fortran::semantics::Attr::TARGET,
                    Fortran::semantics::Attr::VOLATILE}))
    return false;

  return true;
}

static mlir::ArrayAttr
getSafeRepackAttrs(Fortran::lower::AbstractConverter &converter) {
  llvm::SmallVector<mlir::Attribute> attrs;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  const auto &langFeatures = converter.getFoldingContext().languageFeatures();
  if (langFeatures.IsEnabled(Fortran::common::LanguageFeature::OpenACC))
    attrs.push_back(
        fir::OpenACCSafeTempArrayCopyAttr::get(builder.getContext()));
  if (langFeatures.IsEnabled(Fortran::common::LanguageFeature::OpenMP))
    attrs.push_back(
        fir::OpenMPSafeTempArrayCopyAttr::get(builder.getContext()));

  return attrs.empty() ? mlir::ArrayAttr{} : builder.getArrayAttr(attrs);
}

/// Instantiate a local variable. Precondition: Each variable will be visited
/// such that if its properties depend on other variables, the variables upon
/// which its properties depend will already have been visited.
static void instantiateLocal(Fortran::lower::AbstractConverter &converter,
                             const Fortran::lower::pft::Variable &var,
                             Fortran::lower::SymMap &symMap) {
  assert(!var.isAlias());
  Fortran::lower::StatementContext stmtCtx;
  // isUnusedEntryDummy must be computed before mapSymbolAttributes.
  const bool isUnusedEntryDummy =
      var.hasSymbol() && Fortran::semantics::IsDummy(var.getSymbol()) &&
      !symMap.lookupSymbol(var.getSymbol()).getAddr();
  mapSymbolAttributes(converter, var, symMap, stmtCtx);
  // Do not generate code to initialize/finalize/destroy dummy arguments that
  // are nor part of the current ENTRY. They do not have backing storage.
  if (isUnusedEntryDummy)
    return;
  deallocateIntentOut(converter, var, symMap);
  if (needDummyIntentoutFinalization(var))
    finalizeAtRuntime(converter, var, symMap);
  if (mustBeDefaultInitializedAtRuntime(var))
    Fortran::lower::defaultInitializeAtRuntime(converter, var.getSymbol(),
                                               symMap);
  if (converter.getFoldingContext().languageFeatures().IsEnabled(
          Fortran::common::LanguageFeature::CUDA))
    initializeDeviceComponentAllocator(converter, var.getSymbol(), symMap);
  auto *builder = &converter.getFirOpBuilder();
  if (needCUDAAlloc(var.getSymbol()) &&
      !cuf::isCUDADeviceContext(builder->getRegion())) {
    cuf::DataAttributeAttr dataAttr =
        Fortran::lower::translateSymbolCUFDataAttribute(builder->getContext(),
                                                        var.getSymbol());
    mlir::Location loc = converter.getCurrentLocation();
    fir::ExtendedValue exv =
        converter.getSymbolExtendedValue(var.getSymbol(), &symMap);
    auto *sym = &var.getSymbol();
    const Fortran::semantics::Scope &owner = sym->owner();
    if (owner.kind() != Fortran::semantics::Scope::Kind::MainProgram &&
        dataAttr.getValue() != cuf::DataAttribute::Shared) {
      converter.getFctCtx().attachCleanup([builder, loc, exv, sym]() {
        cuf::DataAttributeAttr dataAttr =
            Fortran::lower::translateSymbolCUFDataAttribute(
                builder->getContext(), *sym);
        cuf::FreeOp::create(*builder, loc, fir::getBase(exv), dataAttr);
      });
    }
  }
  if (std::optional<VariableCleanUp> cleanup =
          needDeallocationOrFinalization(var)) {
    auto *builder = &converter.getFirOpBuilder();
    mlir::Location loc = converter.getCurrentLocation();
    fir::ExtendedValue exv =
        converter.getSymbolExtendedValue(var.getSymbol(), &symMap);
    switch (*cleanup) {
    case VariableCleanUp::Finalize:
      converter.getFctCtx().attachCleanup([builder, loc, exv]() {
        mlir::Value box = builder->createBox(loc, exv);
        fir::runtime::genDerivedTypeDestroy(*builder, loc, box);
      });
      break;
    case VariableCleanUp::Deallocate:
      auto *converterPtr = &converter;
      auto *sym = &var.getSymbol();
      converter.getFctCtx().attachCleanup([converterPtr, loc, exv, sym]() {
        const fir::MutableBoxValue *mutableBox =
            exv.getBoxOf<fir::MutableBoxValue>();
        assert(mutableBox &&
               "trying to deallocate entity not lowered as allocatable");
        Fortran::lower::genDeallocateIfAllocated(*converterPtr, *mutableBox,
                                                 loc, sym);
      });
    }
  } else if (var.hasSymbol() && needsRepack(converter, var.getSymbol())) {
    auto *converterPtr = &converter;
    mlir::Location loc = converter.getCurrentLocation();
    auto *sym = &var.getSymbol();
    std::optional<fir::FortranVariableOpInterface> varDef =
        symMap.lookupVariableDefinition(*sym);
    assert(varDef && "cannot find defining operation for an array that needs "
                     "to be repacked");
    converter.getFctCtx().attachCleanup([converterPtr, loc, varDef, sym]() {
      Fortran::lower::genUnpackArray(*converterPtr, loc, *varDef, *sym);
    });
  }
}

//===----------------------------------------------------------------===//
// Aliased (EQUIVALENCE) variables instantiation
//===----------------------------------------------------------------===//

/// Insert \p aggregateStore instance into an AggregateStoreMap.
static void insertAggregateStore(Fortran::lower::AggregateStoreMap &storeMap,
                                 const Fortran::lower::pft::Variable &var,
                                 mlir::Value aggregateStore) {
  std::size_t off = var.getAggregateStore().getOffset();
  Fortran::lower::AggregateStoreKey key = {var.getOwningScope(), off};
  storeMap[key] = aggregateStore;
}

/// Retrieve the aggregate store instance of \p alias from an
/// AggregateStoreMap.
static mlir::Value
getAggregateStore(Fortran::lower::AggregateStoreMap &storeMap,
                  const Fortran::lower::pft::Variable &alias) {
  Fortran::lower::AggregateStoreKey key = {alias.getOwningScope(),
                                           alias.getAliasOffset()};
  auto iter = storeMap.find(key);
  assert(iter != storeMap.end());
  return iter->second;
}

/// Build the name for the storage of a global equivalence.
static std::string mangleGlobalAggregateStore(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable::AggregateStore &st) {
  return converter.mangleName(st.getNamingSymbol());
}

/// Build the type for the storage of an equivalence.
static mlir::Type
getAggregateType(Fortran::lower::AbstractConverter &converter,
                 const Fortran::lower::pft::Variable::AggregateStore &st) {
  if (const Fortran::semantics::Symbol *initSym = st.getInitialValueSymbol())
    return converter.genType(*initSym);
  mlir::IntegerType byteTy = converter.getFirOpBuilder().getIntegerType(8);
  return fir::SequenceType::get(std::get<1>(st.interval), byteTy);
}

/// Define a GlobalOp for the storage of a global equivalence described
/// by \p aggregate. The global is named \p aggName and is created with
/// the provided \p linkage.
/// If any of the equivalence members are initialized, an initializer is
/// created for the equivalence.
/// This is to be used when lowering the scope that owns the equivalence
/// (as opposed to simply using it through host or use association).
/// This is not to be used for equivalence of common block members (they
/// already have the common block GlobalOp for them, see defineCommonBlock).
static fir::GlobalOp defineGlobalAggregateStore(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable::AggregateStore &aggregate,
    llvm::StringRef aggName, mlir::StringAttr linkage) {
  assert(aggregate.isGlobal() && "not a global interval");
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  fir::GlobalOp global = builder.getNamedGlobal(aggName);
  if (global && globalIsInitialized(global))
    return global;
  mlir::Location loc = converter.getCurrentLocation();
  mlir::Type aggTy = getAggregateType(converter, aggregate);
  if (!global)
    global = builder.createGlobal(loc, aggTy, aggName, linkage);

  if (const Fortran::semantics::Symbol *initSym =
          aggregate.getInitialValueSymbol())
    if (const auto *objectDetails =
            initSym->detailsIf<Fortran::semantics::ObjectEntityDetails>())
      if (objectDetails->init()) {
        createGlobalInitialization(
            builder, global, [&](fir::FirOpBuilder &builder) {
              Fortran::lower::StatementContext stmtCtx;
              mlir::Value initVal = fir::getBase(genInitializerExprValue(
                  converter, loc, objectDetails->init().value(), stmtCtx));
              fir::HasValueOp::create(builder, loc, initVal);
            });
        return global;
      }
  // Equivalence has no Fortran initial value. Create an undefined FIR initial
  // value to ensure this is consider an object definition in the IR regardless
  // of the linkage.
  createGlobalInitialization(builder, global, [&](fir::FirOpBuilder &builder) {
    Fortran::lower::StatementContext stmtCtx;
    mlir::Value initVal = fir::ZeroOp::create(builder, loc, aggTy);
    fir::HasValueOp::create(builder, loc, initVal);
  });
  return global;
}

/// Declare a GlobalOp for the storage of a global equivalence described
/// by \p aggregate. The global is named \p aggName and is created with
/// the provided \p linkage.
/// No initializer is built for the created GlobalOp.
/// This is to be used when lowering the scope that uses members of an
/// equivalence it through host or use association.
/// This is not to be used for equivalence of common block members (they
/// already have the common block GlobalOp for them, see defineCommonBlock).
static fir::GlobalOp declareGlobalAggregateStore(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::lower::pft::Variable::AggregateStore &aggregate,
    llvm::StringRef aggName, mlir::StringAttr linkage) {
  assert(aggregate.isGlobal() && "not a global interval");
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  if (fir::GlobalOp global = builder.getNamedGlobal(aggName))
    return global;
  mlir::Type aggTy = getAggregateType(converter, aggregate);
  return builder.createGlobal(loc, aggTy, aggName, linkage);
}

/// This is an aggregate store for a set of EQUIVALENCED variables. Create the
/// storage on the stack or global memory and add it to the map.
static void
instantiateAggregateStore(Fortran::lower::AbstractConverter &converter,
                          const Fortran::lower::pft::Variable &var,
                          Fortran::lower::AggregateStoreMap &storeMap) {
  assert(var.isAggregateStore() && "not an interval");
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::IntegerType i8Ty = builder.getIntegerType(8);
  mlir::Location loc = converter.getCurrentLocation();
  std::string aggName =
      mangleGlobalAggregateStore(converter, var.getAggregateStore());
  if (var.isGlobal()) {
    fir::GlobalOp global;
    auto &aggregate = var.getAggregateStore();
    mlir::StringAttr linkage = getLinkageAttribute(converter, var);
    if (var.isModuleOrSubmoduleVariable()) {
      // A module global was or will be defined when lowering the module. Emit
      // only a declaration if the global does not exist at that point.
      global = declareGlobalAggregateStore(converter, loc, aggregate, aggName,
                                           linkage);
    } else {
      global =
          defineGlobalAggregateStore(converter, aggregate, aggName, linkage);
    }
    auto addr = fir::AddrOfOp::create(builder, loc, global.resultType(),
                                      global.getSymbol());
    auto size = std::get<1>(var.getInterval());
    fir::SequenceType::Shape shape(1, size);
    auto seqTy = fir::SequenceType::get(shape, i8Ty);
    mlir::Type refTy = builder.getRefType(seqTy);
    mlir::Value aggregateStore = builder.createConvert(loc, refTy, addr);
    insertAggregateStore(storeMap, var, aggregateStore);
    return;
  }
  // This is a local aggregate, allocate an anonymous block of memory.
  auto size = std::get<1>(var.getInterval());
  fir::SequenceType::Shape shape(1, size);
  auto seqTy = fir::SequenceType::get(shape, i8Ty);
  mlir::Value local = builder.allocateLocal(loc, seqTy, aggName, "", {}, {},
                                            /*target=*/false);
  insertAggregateStore(storeMap, var, local);
}

/// Cast an alias address (variable part of an equivalence) to fir.ptr so that
/// the optimizer is conservative and avoids doing copy elision in assignment
/// involving equivalenced variables.
/// TODO: Represent the equivalence aliasing constraint in another way to avoid
/// pessimizing array assignments involving equivalenced variables.
static mlir::Value castAliasToPointer(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Type aliasType,
                                      mlir::Value aliasAddr) {
  return builder.createConvert(loc, fir::PointerType::get(aliasType),
                               aliasAddr);
}

/// Instantiate a member of an equivalence. Compute its address in its
/// aggregate storage and lower its attributes.
static void instantiateAlias(Fortran::lower::AbstractConverter &converter,
                             const Fortran::lower::pft::Variable &var,
                             Fortran::lower::SymMap &symMap,
                             Fortran::lower::AggregateStoreMap &storeMap) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  assert(var.isAlias());
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  const mlir::Location loc = genLocation(converter, sym);
  mlir::IndexType idxTy = builder.getIndexType();
  mlir::IntegerType i8Ty = builder.getIntegerType(8);
  mlir::Type i8Ptr = builder.getRefType(i8Ty);
  mlir::Type symType = converter.genType(sym);
  std::size_t off = sym.GetUltimate().offset() - var.getAliasOffset();
  mlir::Value storeAddr = getAggregateStore(storeMap, var);
  mlir::Value offset = builder.createIntegerConstant(loc, idxTy, off);
  mlir::Value bytePtr = fir::CoordinateOp::create(
      builder, loc, i8Ptr, storeAddr, mlir::ValueRange{offset});
  mlir::Value typedPtr = castAliasToPointer(builder, loc, symType, bytePtr);
  Fortran::lower::StatementContext stmtCtx;
  mapSymbolAttributes(converter, var, symMap, stmtCtx, typedPtr);
  // Default initialization is possible for equivalence members: see
  // F2018 19.5.3.4. Note that if several equivalenced entities have
  // default initialization, they must have the same type, and the standard
  // allows the storage to be default initialized several times (this has
  // no consequences other than wasting some execution time). For now,
  // do not try optimizing this to single default initializations of
  // the equivalenced storages. Keep lowering simple.
  if (mustBeDefaultInitializedAtRuntime(var))
    Fortran::lower::defaultInitializeAtRuntime(converter, var.getSymbol(),
                                               symMap);
  if (converter.getFoldingContext().languageFeatures().IsEnabled(
          Fortran::common::LanguageFeature::CUDA))
    initializeDeviceComponentAllocator(converter, var.getSymbol(), symMap);
}

//===--------------------------------------------------------------===//
// COMMON blocks instantiation
//===--------------------------------------------------------------===//

/// Does any member of the common block has an initializer ?
static bool
commonBlockHasInit(const Fortran::semantics::MutableSymbolVector &cmnBlkMems) {
  for (const Fortran::semantics::MutableSymbolRef &mem : cmnBlkMems) {
    if (const auto *memDet =
            mem->detailsIf<Fortran::semantics::ObjectEntityDetails>())
      if (memDet->init())
        return true;
  }
  return false;
}

/// Build a tuple type for a common block based on the common block
/// members and the common block size.
/// This type is only needed to build common block initializers where
/// the initial value is the collection of the member initial values.
static mlir::TupleType getTypeOfCommonWithInit(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::semantics::MutableSymbolVector &cmnBlkMems,
    std::size_t commonSize) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  llvm::SmallVector<mlir::Type> members;
  std::size_t counter = 0;
  for (const Fortran::semantics::MutableSymbolRef &mem : cmnBlkMems) {
    if (const auto *memDet =
            mem->detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
      if (mem->offset() > counter) {
        fir::SequenceType::Shape len = {
            static_cast<fir::SequenceType::Extent>(mem->offset() - counter)};
        mlir::IntegerType byteTy = builder.getIntegerType(8);
        auto memTy = fir::SequenceType::get(len, byteTy);
        members.push_back(memTy);
        counter = mem->offset();
      }
      if (memDet->init()) {
        mlir::Type memTy = converter.genType(*mem);
        members.push_back(memTy);
        counter = mem->offset() + mem->size();
      }
    }
  }
  if (counter < commonSize) {
    fir::SequenceType::Shape len = {
        static_cast<fir::SequenceType::Extent>(commonSize - counter)};
    mlir::IntegerType byteTy = builder.getIntegerType(8);
    auto memTy = fir::SequenceType::get(len, byteTy);
    members.push_back(memTy);
  }
  return mlir::TupleType::get(builder.getContext(), members);
}

/// Common block members may have aliases. They are not in the common block
/// member list from the symbol. We need to know about these aliases if they
/// have initializer to generate the common initializer.
/// This function takes care of adding aliases with initializer to the member
/// list.
static Fortran::semantics::MutableSymbolVector
getCommonMembersWithInitAliases(const Fortran::semantics::Symbol &common) {
  const auto &commonDetails =
      common.get<Fortran::semantics::CommonBlockDetails>();
  auto members = commonDetails.objects();

  // The number and size of equivalence and common is expected to be small, so
  // no effort is given to optimize this loop of complexity equivalenced
  // common members * common members
  for (const Fortran::semantics::EquivalenceSet &set :
       common.owner().equivalenceSets())
    for (const Fortran::semantics::EquivalenceObject &obj : set) {
      if (!obj.symbol.test(Fortran::semantics::Symbol::Flag::CompilerCreated)) {
        if (const auto &details =
                obj.symbol
                    .detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
          const Fortran::semantics::Symbol *com =
              FindCommonBlockContaining(obj.symbol);
          if (!details->init() || com != &common)
            continue;
          // This is an alias with an init that belongs to the list
          if (!llvm::is_contained(members, obj.symbol))
            members.emplace_back(obj.symbol);
        }
      }
    }
  return members;
}

/// Return the fir::GlobalOp that was created of COMMON block \p common.
/// It is an error if the fir::GlobalOp was not created before this is
/// called (it cannot be created on the flight because it is not known here
/// what mlir type the GlobalOp should have to satisfy all the
/// appearances in the program).
static fir::GlobalOp
getCommonBlockGlobal(Fortran::lower::AbstractConverter &converter,
                     const Fortran::semantics::Symbol &common) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  std::string commonName = converter.mangleName(common);
  fir::GlobalOp global = builder.getNamedGlobal(commonName);
  // Common blocks are lowered before any subprograms to deal with common
  // whose size may not be the same in every subprograms.
  if (!global)
    fir::emitFatalError(converter.genLocation(common.name()),
                        "COMMON block was not lowered before its usage");
  return global;
}

/// Create the fir::GlobalOp for COMMON block \p common. If \p common has an
/// initial value, it is not created yet. Instead, the common block list
/// members is returned to later create the initial value in
/// finalizeCommonBlockDefinition.
static std::optional<std::tuple<
    fir::GlobalOp, Fortran::semantics::MutableSymbolVector, mlir::Location>>
declareCommonBlock(Fortran::lower::AbstractConverter &converter,
                   const Fortran::semantics::Symbol &common,
                   std::size_t commonSize) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  std::string commonName = converter.mangleName(common);
  fir::GlobalOp global = builder.getNamedGlobal(commonName);
  if (global)
    return std::nullopt;
  Fortran::semantics::MutableSymbolVector cmnBlkMems =
      getCommonMembersWithInitAliases(common);
  mlir::Location loc = converter.genLocation(common.name());
  mlir::StringAttr linkage = builder.createCommonLinkage();
  const auto *details =
      common.detailsIf<Fortran::semantics::CommonBlockDetails>();
  assert(details && "Expect CommonBlockDetails on the common symbol");
  if (!commonBlockHasInit(cmnBlkMems)) {
    // A COMMON block sans initializers is initialized to zero.
    // mlir::Vector types must have a strictly positive size, so at least
    // temporarily, force a zero size COMMON block to have one byte.
    const auto sz =
        static_cast<fir::SequenceType::Extent>(commonSize > 0 ? commonSize : 1);
    fir::SequenceType::Shape shape = {sz};
    mlir::IntegerType i8Ty = builder.getIntegerType(8);
    auto commonTy = fir::SequenceType::get(shape, i8Ty);
    auto vecTy = mlir::VectorType::get(sz, i8Ty);
    mlir::Attribute zero = builder.getIntegerAttr(i8Ty, 0);
    auto init = mlir::DenseElementsAttr::get(vecTy, llvm::ArrayRef(zero));
    global = builder.createGlobal(loc, commonTy, commonName, linkage, init);
    global.setAlignment(details->alignment());
    // No need to add any initial value later.
    return std::nullopt;
  }
  // COMMON block with initializer (note that initialized blank common are
  // accepted as an extension by semantics). Sort members by offset before
  // generating the type and initializer.
  std::sort(cmnBlkMems.begin(), cmnBlkMems.end(),
            [](auto &s1, auto &s2) { return s1->offset() < s2->offset(); });
  mlir::TupleType commonTy =
      getTypeOfCommonWithInit(converter, cmnBlkMems, commonSize);
  // Create the global object, the initial value will be added later.
  global = builder.createGlobal(loc, commonTy, commonName);
  global.setAlignment(details->alignment());
  return std::make_tuple(global, std::move(cmnBlkMems), loc);
}

/// Add initial value to a COMMON block fir::GlobalOp \p global given the list
/// \p cmnBlkMems of the common block member symbols that contains symbols with
/// an initial value.
static void finalizeCommonBlockDefinition(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    fir::GlobalOp global,
    const Fortran::semantics::MutableSymbolVector &cmnBlkMems) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::TupleType commonTy = mlir::cast<mlir::TupleType>(global.getType());
  auto initFunc = [&](fir::FirOpBuilder &builder) {
    mlir::IndexType idxTy = builder.getIndexType();
    mlir::Value cb = fir::ZeroOp::create(builder, loc, commonTy);
    unsigned tupIdx = 0;
    std::size_t offset = 0;
    LLVM_DEBUG(llvm::dbgs() << "block {\n");
    for (const Fortran::semantics::MutableSymbolRef &mem : cmnBlkMems) {
      if (const auto *memDet =
              mem->detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
        if (mem->offset() > offset) {
          ++tupIdx;
          offset = mem->offset();
        }
        if (memDet->init()) {
          LLVM_DEBUG(llvm::dbgs()
                     << "offset: " << mem->offset() << " is " << *mem << '\n');
          Fortran::lower::StatementContext stmtCtx;
          auto initExpr = memDet->init().value();
          fir::ExtendedValue initVal =
              Fortran::semantics::IsPointer(*mem)
                  ? Fortran::lower::genInitialDataTarget(
                        converter, loc, converter.genType(*mem), initExpr)
                  : genInitializerExprValue(converter, loc, initExpr, stmtCtx);
          mlir::IntegerAttr offVal = builder.getIntegerAttr(idxTy, tupIdx);
          mlir::Value castVal = builder.createConvert(
              loc, commonTy.getType(tupIdx), fir::getBase(initVal));
          cb = fir::InsertValueOp::create(builder, loc, commonTy, cb, castVal,
                                          builder.getArrayAttr(offVal));
          ++tupIdx;
          offset = mem->offset() + mem->size();
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "}\n");
    fir::HasValueOp::create(builder, loc, cb);
  };
  createGlobalInitialization(builder, global, initFunc);
}

void Fortran::lower::defineCommonBlocks(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::semantics::CommonBlockList &commonBlocks) {
  // Common blocks may depend on another common block address (if they contain
  // pointers with initial targets). To cover this case, create all common block
  // fir::Global before creating the initial values (if any).
  std::vector<std::tuple<fir::GlobalOp, Fortran::semantics::MutableSymbolVector,
                         mlir::Location>>
      delayedInitializations;
  for (const auto &[common, size] : commonBlocks)
    if (auto delayedInit = declareCommonBlock(converter, common, size))
      delayedInitializations.emplace_back(std::move(*delayedInit));
  for (auto &[global, cmnBlkMems, loc] : delayedInitializations)
    finalizeCommonBlockDefinition(loc, converter, global, cmnBlkMems);
}

mlir::Value Fortran::lower::genCommonBlockMember(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::semantics::Symbol &sym, mlir::Value commonValue) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  std::size_t byteOffset = sym.GetUltimate().offset();
  mlir::IntegerType i8Ty = builder.getIntegerType(8);
  mlir::Type i8Ptr = builder.getRefType(i8Ty);
  mlir::Type seqTy = builder.getRefType(builder.getVarLenSeqTy(i8Ty));
  mlir::Value base = builder.createConvert(loc, seqTy, commonValue);

  mlir::Value offs =
      builder.createIntegerConstant(loc, builder.getIndexType(), byteOffset);
  mlir::Value varAddr = fir::CoordinateOp::create(builder, loc, i8Ptr, base,
                                                  mlir::ValueRange{offs});
  mlir::Type symType = converter.genType(sym);

  return Fortran::semantics::FindEquivalenceSet(sym) != nullptr
             ? castAliasToPointer(builder, loc, symType, varAddr)
             : builder.createConvert(loc, builder.getRefType(symType), varAddr);
}

/// The COMMON block is a global structure. `var` will be at some offset
/// within the COMMON block. Adds the address of `var` (COMMON + offset) to
/// the symbol map.
static void instantiateCommon(Fortran::lower::AbstractConverter &converter,
                              const Fortran::semantics::Symbol &common,
                              const Fortran::lower::pft::Variable &var,
                              Fortran::lower::SymMap &symMap) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  const Fortran::semantics::Symbol &varSym = var.getSymbol();
  mlir::Location loc = converter.genLocation(varSym.name());

  mlir::Value commonAddr;
  if (Fortran::lower::SymbolBox symBox = symMap.lookupSymbol(common))
    commonAddr = symBox.getAddr();
  if (!commonAddr) {
    // introduce a local AddrOf and add it to the map
    fir::GlobalOp global = getCommonBlockGlobal(converter, common);
    commonAddr = fir::AddrOfOp::create(builder, loc, global.resultType(),
                                       global.getSymbol());

    symMap.addSymbol(common, commonAddr);
  }

  mlir::Value local = genCommonBlockMember(converter, loc, varSym, commonAddr);
  Fortran::lower::StatementContext stmtCtx;
  mapSymbolAttributes(converter, var, symMap, stmtCtx, local);
}

//===--------------------------------------------------------------===//
// Lower Variables specification expressions and attributes
//===--------------------------------------------------------------===//

/// Helper to decide if a dummy argument must be tracked in an BoxValue.
static bool lowerToBoxValue(const Fortran::semantics::Symbol &sym,
                            mlir::Value dummyArg,
                            Fortran::lower::AbstractConverter &converter) {
  // Only dummy arguments coming as fir.box can be tracked in an BoxValue.
  if (!dummyArg || !mlir::isa<fir::BaseBoxType>(dummyArg.getType()))
    return false;
  // Non contiguous arrays must be tracked in an BoxValue.
  if (sym.Rank() > 0 && !Fortran::evaluate::IsSimplyContiguous(
                            sym, converter.getFoldingContext()))
    return true;
  // Assumed rank and optional fir.box cannot yet be read while lowering the
  // specifications.
  if (Fortran::evaluate::IsAssumedRank(sym) ||
      Fortran::semantics::IsOptional(sym))
    return true;
  // Polymorphic entity should be tracked through a fir.box that has the
  // dynamic type info.
  if (const Fortran::semantics::DeclTypeSpec *type = sym.GetType())
    if (type->IsPolymorphic())
      return true;
  return false;
}

/// Lower explicit lower bounds into \p result. Does nothing if this is not an
/// array, or if the lower bounds are deferred, or all implicit or one.
static void lowerExplicitLowerBounds(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::lower::BoxAnalyzer &box,
    llvm::SmallVectorImpl<mlir::Value> &result, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  if (!box.isArray() || box.lboundIsAllOnes())
    return;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::IndexType idxTy = builder.getIndexType();
  if (box.isStaticArray()) {
    for (int64_t lb : box.staticLBound())
      result.emplace_back(builder.createIntegerConstant(loc, idxTy, lb));
    return;
  }
  for (const Fortran::semantics::ShapeSpec *spec : box.dynamicBound()) {
    if (auto low = spec->lbound().GetExplicit()) {
      auto expr = Fortran::lower::SomeExpr{*low};
      mlir::Value lb = builder.createConvert(
          loc, idxTy, genScalarValue(converter, loc, expr, symMap, stmtCtx));
      result.emplace_back(lb);
    }
  }
  assert(result.empty() || result.size() == box.dynamicBound().size());
}

/// Return -1 for the last dimension extent/upper bound of assumed-size arrays.
/// This value is required to fulfill the requirements for assumed-rank
/// associated with assumed-size (see for instance UBOUND in 16.9.196, and
/// CFI_desc_t requirements in 18.5.3 point 5.).
static mlir::Value getAssumedSizeExtent(mlir::Location loc,
                                        fir::FirOpBuilder &builder) {
  return builder.createMinusOneInteger(loc, builder.getIndexType());
}

/// Lower explicit extents into \p result if this is an explicit-shape or
/// assumed-size array. Does nothing if this is not an explicit-shape or
/// assumed-size array.
static void
lowerExplicitExtents(Fortran::lower::AbstractConverter &converter,
                     mlir::Location loc, const Fortran::lower::BoxAnalyzer &box,
                     llvm::SmallVectorImpl<mlir::Value> &lowerBounds,
                     llvm::SmallVectorImpl<mlir::Value> &result,
                     Fortran::lower::SymMap &symMap,
                     Fortran::lower::StatementContext &stmtCtx) {
  if (!box.isArray())
    return;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::IndexType idxTy = builder.getIndexType();
  if (box.isStaticArray()) {
    for (int64_t extent : box.staticShape())
      result.emplace_back(builder.createIntegerConstant(loc, idxTy, extent));
    return;
  }
  for (const auto &spec : llvm::enumerate(box.dynamicBound())) {
    if (auto up = spec.value()->ubound().GetExplicit()) {
      auto expr = Fortran::lower::SomeExpr{*up};
      mlir::Value ub = builder.createConvert(
          loc, idxTy, genScalarValue(converter, loc, expr, symMap, stmtCtx));
      if (lowerBounds.empty())
        result.emplace_back(fir::factory::genMaxWithZero(builder, loc, ub));
      else
        result.emplace_back(fir::factory::computeExtent(
            builder, loc, lowerBounds[spec.index()], ub));
    } else if (spec.value()->ubound().isStar()) {
      result.emplace_back(getAssumedSizeExtent(loc, builder));
    }
  }
  assert(result.empty() || result.size() == box.dynamicBound().size());
}

/// Lower explicit character length if any. Return empty mlir::Value if no
/// explicit length.
static mlir::Value
lowerExplicitCharLen(Fortran::lower::AbstractConverter &converter,
                     mlir::Location loc, const Fortran::lower::BoxAnalyzer &box,
                     Fortran::lower::SymMap &symMap,
                     Fortran::lower::StatementContext &stmtCtx) {
  if (!box.isChar())
    return mlir::Value{};
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Type lenTy = builder.getCharacterLengthType();
  if (std::optional<int64_t> len = box.getCharLenConst())
    return builder.createIntegerConstant(loc, lenTy, *len);
  if (std::optional<Fortran::lower::SomeExpr> lenExpr = box.getCharLenExpr())
    // If the length expression is negative, the length is zero. See F2018
    // 7.4.4.2 point 5.
    return fir::factory::genMaxWithZero(
        builder, loc,
        genScalarValue(converter, loc, *lenExpr, symMap, stmtCtx));
  return mlir::Value{};
}

/// Assumed size arrays last extent is -1 in the front end.
static mlir::Value genExtentValue(fir::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Type idxTy,
                                  long frontEndExtent) {
  if (frontEndExtent >= 0)
    return builder.createIntegerConstant(loc, idxTy, frontEndExtent);
  return getAssumedSizeExtent(loc, builder);
}

/// If a symbol is an array, it may have been declared with unknown extent
/// parameters (e.g., `*`), but if it has an initial value then the actual size
/// may be available from the initial array value's type.
inline static llvm::SmallVector<std::int64_t>
recoverShapeVector(llvm::ArrayRef<std::int64_t> shapeVec, mlir::Value initVal) {
  llvm::SmallVector<std::int64_t> result;
  if (initVal) {
    if (auto seqTy = fir::unwrapUntilSeqType(initVal.getType())) {
      for (auto [fst, snd] : llvm::zip(shapeVec, seqTy.getShape()))
        result.push_back(fst == fir::SequenceType::getUnknownExtent() ? snd
                                                                      : fst);
      return result;
    }
  }
  result.assign(shapeVec.begin(), shapeVec.end());
  return result;
}

fir::FortranVariableFlagsAttr Fortran::lower::translateSymbolAttributes(
    mlir::MLIRContext *mlirContext, const Fortran::semantics::Symbol &sym,
    fir::FortranVariableFlagsEnum extraFlags) {
  fir::FortranVariableFlagsEnum flags = extraFlags;
  if (sym.test(Fortran::semantics::Symbol::Flag::CrayPointee)) {
    // CrayPointee are represented as pointers.
    flags = flags | fir::FortranVariableFlagsEnum::pointer;
    return fir::FortranVariableFlagsAttr::get(mlirContext, flags);
  }
  const auto &attrs = sym.attrs();
  if (attrs.test(Fortran::semantics::Attr::ALLOCATABLE))
    flags = flags | fir::FortranVariableFlagsEnum::allocatable;
  if (attrs.test(Fortran::semantics::Attr::ASYNCHRONOUS))
    flags = flags | fir::FortranVariableFlagsEnum::asynchronous;
  if (attrs.test(Fortran::semantics::Attr::BIND_C))
    flags = flags | fir::FortranVariableFlagsEnum::bind_c;
  if (attrs.test(Fortran::semantics::Attr::CONTIGUOUS))
    flags = flags | fir::FortranVariableFlagsEnum::contiguous;
  if (attrs.test(Fortran::semantics::Attr::INTENT_IN))
    flags = flags | fir::FortranVariableFlagsEnum::intent_in;
  if (attrs.test(Fortran::semantics::Attr::INTENT_INOUT))
    flags = flags | fir::FortranVariableFlagsEnum::intent_inout;
  if (attrs.test(Fortran::semantics::Attr::INTENT_OUT))
    flags = flags | fir::FortranVariableFlagsEnum::intent_out;
  if (attrs.test(Fortran::semantics::Attr::OPTIONAL))
    flags = flags | fir::FortranVariableFlagsEnum::optional;
  if (attrs.test(Fortran::semantics::Attr::PARAMETER))
    flags = flags | fir::FortranVariableFlagsEnum::parameter;
  if (attrs.test(Fortran::semantics::Attr::POINTER))
    flags = flags | fir::FortranVariableFlagsEnum::pointer;
  if (attrs.test(Fortran::semantics::Attr::TARGET))
    flags = flags | fir::FortranVariableFlagsEnum::target;
  if (attrs.test(Fortran::semantics::Attr::VALUE))
    flags = flags | fir::FortranVariableFlagsEnum::value;
  if (attrs.test(Fortran::semantics::Attr::VOLATILE))
    flags = flags | fir::FortranVariableFlagsEnum::fortran_volatile;
  if (flags == fir::FortranVariableFlagsEnum::None)
    return {};
  return fir::FortranVariableFlagsAttr::get(mlirContext, flags);
}

static bool
isCapturedInInternalProcedure(Fortran::lower::AbstractConverter &converter,
                              const Fortran::semantics::Symbol &sym) {
  const Fortran::lower::pft::FunctionLikeUnit *funit =
      converter.getCurrentFunctionUnit();
  if (!funit || funit->getHostAssoc().empty())
    return false;
  if (funit->getHostAssoc().isAssociated(sym))
    return true;
  // Consider that any capture of a variable that is in an equivalence with the
  // symbol imply that the storage of the symbol may also be accessed inside
  // symbol implies that the storage of the symbol may also be accessed inside

  // the internal procedure and flag it as captured.
  if (const auto *equivSet = Fortran::semantics::FindEquivalenceSet(sym))
    for (const Fortran::semantics::EquivalenceObject &eqObj : *equivSet)
      if (funit->getHostAssoc().isAssociated(eqObj.symbol))
        return true;
  return false;
}

/// Map a symbol to its FIR address and evaluated specification expressions.
/// Not for symbols lowered to fir.box.
/// Will optionally create fir.declare.
static void genDeclareSymbol(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::SymMap &symMap,
                             const Fortran::semantics::Symbol &sym,
                             mlir::Value base, mlir::Value len = {},
                             llvm::ArrayRef<mlir::Value> shape = {},
                             llvm::ArrayRef<mlir::Value> lbounds = {},
                             bool force = false) {
  // In HLFIR, procedure dummy symbols are not added with an hlfir.declare
  // because they are "values", and hlfir.declare is intended for variables. It
  // would add too much complexity to hlfir.declare to support this case, and
  // this would bring very little (the only point being debug info, that are not
  // yet emitted) since alias analysis is meaningless for those.
  // Commonblock names are not variables, but in some lowerings (like OpenMP) it
  // is useful to maintain the address of the commonblock in an MLIR value and
  // query it. hlfir.declare need not be created for these.
  if (converter.getLoweringOptions().getLowerToHighLevelFIR() &&
      (!Fortran::semantics::IsProcedure(sym) ||
       Fortran::semantics::IsPointer(sym)) &&
      !sym.detailsIf<Fortran::semantics::CommonBlockDetails>()) {
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    const mlir::Location loc = genLocation(converter, sym);
    mlir::Value shapeOrShift;
    if (!shape.empty() && !lbounds.empty())
      shapeOrShift = builder.genShape(loc, lbounds, shape);
    else if (!shape.empty())
      shapeOrShift = builder.genShape(loc, shape);
    else if (!lbounds.empty())
      shapeOrShift = builder.genShift(loc, lbounds);
    llvm::SmallVector<mlir::Value> lenParams;
    if (len)
      lenParams.emplace_back(len);
    auto name = converter.mangleName(sym);
    fir::FortranVariableFlagsEnum extraFlags = {};
    if (isCapturedInInternalProcedure(converter, sym))
      extraFlags = extraFlags | fir::FortranVariableFlagsEnum::internal_assoc;
    fir::FortranVariableFlagsAttr attributes =
        Fortran::lower::translateSymbolAttributes(builder.getContext(), sym,
                                                  extraFlags);
    cuf::DataAttributeAttr dataAttr =
        Fortran::lower::translateSymbolCUFDataAttribute(builder.getContext(),
                                                        sym);

    if (sym.test(Fortran::semantics::Symbol::Flag::CrayPointee)) {
      mlir::Type ptrBoxType =
          Fortran::lower::getCrayPointeeBoxType(base.getType());
      mlir::Value boxAlloc = builder.createTemporary(
          loc, ptrBoxType,
          /*name=*/{}, /*shape=*/{}, /*lenParams=*/{}, /*attrs=*/{},
          Fortran::semantics::GetCUDADataAttr(&sym.GetUltimate()));

      // Declare a local pointer variable.
      auto newBase = hlfir::DeclareOp::create(
          builder, loc, boxAlloc, name, /*shape=*/nullptr, lenParams,
          /*dummy_scope=*/nullptr, attributes);
      mlir::Value nullAddr = builder.createNullConstant(
          loc, llvm::cast<fir::BaseBoxType>(ptrBoxType).getEleTy());

      // If the element type is known-length character, then
      // EmboxOp does not need the length parameters.
      if (auto charType = mlir::dyn_cast<fir::CharacterType>(
              hlfir::getFortranElementType(base.getType())))
        if (!charType.hasDynamicLen())
          lenParams.clear();

      // Inherit the shape (and maybe length parameters) from the pointee
      // declaration.
      mlir::Value initVal =
          fir::EmboxOp::create(builder, loc, ptrBoxType, nullAddr, shapeOrShift,
                               /*slice=*/nullptr, lenParams);
      fir::StoreOp::create(builder, loc, initVal, newBase.getBase());

      // Any reference to the pointee is going to be using the pointer
      // box from now on. The base_addr of the descriptor must be updated
      // to hold the value of the Cray pointer at the point of the pointee
      // access.
      // Note that the same Cray pointer may be associated with
      // multiple pointees and each of them has its own descriptor.
      symMap.addVariableDefinition(sym, newBase, force);
      return;
    }
    mlir::Value dummyScope;
    if (converter.isRegisteredDummySymbol(sym))
      dummyScope = converter.dummyArgsScopeValue();
    auto newBase =
        hlfir::DeclareOp::create(builder, loc, base, name, shapeOrShift,
                                 lenParams, dummyScope, attributes, dataAttr);
    symMap.addVariableDefinition(sym, newBase, force);
    return;
  }

  if (len) {
    if (!shape.empty()) {
      if (!lbounds.empty())
        symMap.addCharSymbolWithBounds(sym, base, len, shape, lbounds, force);
      else
        symMap.addCharSymbolWithShape(sym, base, len, shape, force);
    } else {
      symMap.addCharSymbol(sym, base, len, force);
    }
  } else {
    if (!shape.empty()) {
      if (!lbounds.empty())
        symMap.addSymbolWithBounds(sym, base, shape, lbounds, force);
      else
        symMap.addSymbolWithShape(sym, base, shape, force);
    } else {
      symMap.addSymbol(sym, base, force);
    }
  }
}

/// Map a symbol to its FIR address and evaluated specification expressions
/// provided as a fir::ExtendedValue. Will optionally create fir.declare.
void Fortran::lower::genDeclareSymbol(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symMap, const Fortran::semantics::Symbol &sym,
    const fir::ExtendedValue &exv, fir::FortranVariableFlagsEnum extraFlags,
    bool force) {
  if (converter.getLoweringOptions().getLowerToHighLevelFIR() &&
      (!Fortran::semantics::IsProcedure(sym) ||
       Fortran::semantics::IsPointer(sym.GetUltimate())) &&
      !sym.detailsIf<Fortran::semantics::CommonBlockDetails>()) {
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    const mlir::Location loc = genLocation(converter, sym);
    if (isCapturedInInternalProcedure(converter, sym))
      extraFlags = extraFlags | fir::FortranVariableFlagsEnum::internal_assoc;
    // FIXME: Using the ultimate symbol for translating symbol attributes will
    // lead to situations where the VOLATILE/ASYNCHRONOUS attributes are not
    // propagated to the hlfir.declare (these attributes can be added when
    // using module variables).
    fir::FortranVariableFlagsAttr attributes =
        Fortran::lower::translateSymbolAttributes(
            builder.getContext(), sym.GetUltimate(), extraFlags);
    cuf::DataAttributeAttr dataAttr =
        Fortran::lower::translateSymbolCUFDataAttribute(builder.getContext(),
                                                        sym.GetUltimate());
    auto name = converter.mangleName(sym);
    mlir::Value dummyScope;
    fir::ExtendedValue base = exv;
    if (converter.isRegisteredDummySymbol(sym)) {
      base = genPackArray(converter, sym, exv);
      dummyScope = converter.dummyArgsScopeValue();
    }
    hlfir::EntityWithAttributes declare = hlfir::genDeclare(
        loc, builder, base, name, attributes, dummyScope, dataAttr);
    symMap.addVariableDefinition(sym, declare.getIfVariableInterface(), force);
    return;
  }
  symMap.addSymbol(sym, exv, force);
}

/// Map an allocatable or pointer symbol to its FIR address and evaluated
/// specification expressions. Will optionally create fir.declare.
static void
genAllocatableOrPointerDeclare(Fortran::lower::AbstractConverter &converter,
                               Fortran::lower::SymMap &symMap,
                               const Fortran::semantics::Symbol &sym,
                               fir::MutableBoxValue box, bool force = false) {
  if (!converter.getLoweringOptions().getLowerToHighLevelFIR()) {
    symMap.addAllocatableOrPointer(sym, box, force);
    return;
  }
  assert(!box.isDescribedByVariables() &&
         "HLFIR alloctables/pointers must be fir.ref<fir.box>");
  mlir::Value base = box.getAddr();
  mlir::Value explictLength;
  if (box.hasNonDeferredLenParams()) {
    if (!box.isCharacter())
      TODO(genLocation(converter, sym),
           "Pointer or Allocatable parametrized derived type");
    explictLength = box.nonDeferredLenParams()[0];
  }
  genDeclareSymbol(converter, symMap, sym, base, explictLength,
                   /*shape=*/{},
                   /*lbounds=*/{}, force);
}

/// Map a procedure pointer
static void genProcPointer(Fortran::lower::AbstractConverter &converter,
                           Fortran::lower::SymMap &symMap,
                           const Fortran::semantics::Symbol &sym,
                           mlir::Value addr, bool force = false) {
  genDeclareSymbol(converter, symMap, sym, addr, mlir::Value{},
                   /*shape=*/{},
                   /*lbounds=*/{}, force);
}

/// Map a symbol represented with a runtime descriptor to its FIR fir.box and
/// evaluated specification expressions. Will optionally create fir.declare.
static void genBoxDeclare(Fortran::lower::AbstractConverter &converter,
                          Fortran::lower::SymMap &symMap,
                          const Fortran::semantics::Symbol &sym,
                          mlir::Value box, llvm::ArrayRef<mlir::Value> lbounds,
                          llvm::ArrayRef<mlir::Value> explicitParams,
                          llvm::ArrayRef<mlir::Value> explicitExtents,
                          bool replace = false) {
  if (converter.getLoweringOptions().getLowerToHighLevelFIR()) {
    fir::BoxValue boxValue{box, lbounds, explicitParams, explicitExtents};
    Fortran::lower::genDeclareSymbol(
        converter, symMap, sym, std::move(boxValue),
        fir::FortranVariableFlagsEnum::None, replace);
    return;
  }
  symMap.addBoxSymbol(sym, box, lbounds, explicitParams, explicitExtents,
                      replace);
}

/// Lower specification expressions and attributes of variable \p var and
/// add it to the symbol map. For a global or an alias, the address must be
/// pre-computed and provided in \p preAlloc. A dummy argument for the current
/// entry point has already been mapped to an mlir block argument in
/// mapDummiesAndResults. Its mapping may be updated here.
void Fortran::lower::mapSymbolAttributes(
    AbstractConverter &converter, const Fortran::lower::pft::Variable &var,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
    mlir::Value preAlloc) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  const mlir::Location loc = genLocation(converter, sym);
  mlir::IndexType idxTy = builder.getIndexType();
  const bool isDeclaredDummy = Fortran::semantics::IsDummy(sym);
  // An active dummy from the current entry point.
  const bool isDummy = isDeclaredDummy && symMap.lookupSymbol(sym).getAddr();
  // An unused dummy from another entry point.
  const bool isUnusedEntryDummy = isDeclaredDummy && !isDummy;
  const bool isResult = Fortran::semantics::IsFunctionResult(sym);
  const bool replace = isDummy || isResult;
  fir::factory::CharacterExprHelper charHelp{builder, loc};

  if (Fortran::semantics::IsProcedure(sym)) {
    if (isUnusedEntryDummy) {
      // Additional discussion below.
      mlir::Type dummyProcType =
          Fortran::lower::getDummyProcedureType(sym, converter);
      mlir::Value undefOp = fir::UndefOp::create(builder, loc, dummyProcType);

      Fortran::lower::genDeclareSymbol(converter, symMap, sym, undefOp);
    }

    // Procedure pointer.
    if (Fortran::semantics::IsPointer(sym)) {
      // global
      mlir::Value boxAlloc = preAlloc;
      // dummy or passed result
      if (!boxAlloc)
        if (Fortran::lower::SymbolBox symbox = symMap.lookupSymbol(sym))
          boxAlloc = symbox.getAddr();
      // local
      if (!boxAlloc)
        boxAlloc = createNewLocal(converter, loc, var, preAlloc);
      genProcPointer(converter, symMap, sym, boxAlloc, replace);
    }
    return;
  }

  const bool isAssumedRank = Fortran::evaluate::IsAssumedRank(sym);
  if (isAssumedRank && !allowAssumedRank)
    TODO(loc, "assumed-rank variable in procedure implemented in Fortran");

  Fortran::lower::BoxAnalyzer ba;
  ba.analyze(sym);

  // First deal with pointers and allocatables, because their handling here
  // is the same regardless of their rank.
  if (Fortran::semantics::IsAllocatableOrPointer(sym)) {
    // Get address of fir.box describing the entity.
    // global
    mlir::Value boxAlloc = preAlloc;
    // dummy or passed result
    if (!boxAlloc)
      if (Fortran::lower::SymbolBox symbox = symMap.lookupSymbol(sym))
        boxAlloc = symbox.getAddr();
    assert((boxAlloc || !isAssumedRank) && "assumed-ranks cannot be local");
    // local
    if (!boxAlloc)
      boxAlloc = createNewLocal(converter, loc, var, preAlloc);
    // Lower non deferred parameters.
    llvm::SmallVector<mlir::Value> nonDeferredLenParams;
    if (ba.isChar()) {
      if (mlir::Value len =
              lowerExplicitCharLen(converter, loc, ba, symMap, stmtCtx))
        nonDeferredLenParams.push_back(len);
      else if (Fortran::semantics::IsAssumedLengthCharacter(sym))
        nonDeferredLenParams.push_back(
            Fortran::lower::getAssumedCharAllocatableOrPointerLen(
                builder, loc, sym, boxAlloc));
    } else if (const Fortran::semantics::DeclTypeSpec *declTy = sym.GetType()) {
      if (const Fortran::semantics::DerivedTypeSpec *derived =
              declTy->AsDerived())
        if (Fortran::semantics::CountLenParameters(*derived) != 0)
          TODO(loc,
               "derived type allocatable or pointer with length parameters");
    }
    fir::MutableBoxValue box = Fortran::lower::createMutableBox(
        converter, loc, var, boxAlloc, nonDeferredLenParams,
        /*alwaysUseBox=*/
        converter.getLoweringOptions().getLowerToHighLevelFIR(),
        Fortran::lower::getAllocatorIdx(var.getSymbol()));
    genAllocatableOrPointerDeclare(converter, symMap, var.getSymbol(), box,
                                   replace);
    return;
  }

  if (isDummy) {
    mlir::Value dummyArg = symMap.lookupSymbol(sym).getAddr();
    if (lowerToBoxValue(sym, dummyArg, converter)) {
      llvm::SmallVector<mlir::Value> lbounds;
      llvm::SmallVector<mlir::Value> explicitExtents;
      llvm::SmallVector<mlir::Value> explicitParams;
      // Lower lower bounds, explicit type parameters and explicit
      // extents if any.
      if (ba.isChar()) {
        if (mlir::Value len =
                lowerExplicitCharLen(converter, loc, ba, symMap, stmtCtx))
          explicitParams.push_back(len);
        if (!isAssumedRank && sym.Rank() == 0) {
          // Do not keep scalar characters as fir.box (even when optional).
          // Lowering and FIR is not meant to deal with scalar characters as
          // fir.box outside of calls.
          auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(dummyArg.getType());
          mlir::Type refTy = builder.getRefType(boxTy.getEleTy());
          mlir::Type lenType = builder.getCharacterLengthType();
          mlir::Value addr, len;
          if (Fortran::semantics::IsOptional(sym)) {
            auto isPresent = fir::IsPresentOp::create(
                builder, loc, builder.getI1Type(), dummyArg);
            auto addrAndLen =
                builder
                    .genIfOp(loc, {refTy, lenType}, isPresent,
                             /*withElseRegion=*/true)
                    .genThen([&]() {
                      mlir::Value readAddr =
                          fir::BoxAddrOp::create(builder, loc, refTy, dummyArg);
                      mlir::Value readLength =
                          charHelp.readLengthFromBox(dummyArg);
                      fir::ResultOp::create(
                          builder, loc, mlir::ValueRange{readAddr, readLength});
                    })
                    .genElse([&] {
                      mlir::Value readAddr = builder.genAbsentOp(loc, refTy);
                      mlir::Value readLength =
                          fir::factory::createZeroValue(builder, loc, lenType);
                      fir::ResultOp::create(
                          builder, loc, mlir::ValueRange{readAddr, readLength});
                    })
                    .getResults();
            addr = addrAndLen[0];
            len = addrAndLen[1];
          } else {
            addr = fir::BoxAddrOp::create(builder, loc, refTy, dummyArg);
            len = charHelp.readLengthFromBox(dummyArg);
          }
          if (!explicitParams.empty())
            len = explicitParams[0];
          ::genDeclareSymbol(converter, symMap, sym, addr, len, /*extents=*/{},
                             /*lbounds=*/{}, replace);
          return;
        }
      }
      // TODO: derived type length parameters.
      if (!isAssumedRank) {
        lowerExplicitLowerBounds(converter, loc, ba, lbounds, symMap, stmtCtx);
        lowerExplicitExtents(converter, loc, ba, lbounds, explicitExtents,
                             symMap, stmtCtx);
      }
      genBoxDeclare(converter, symMap, sym, dummyArg, lbounds, explicitParams,
                    explicitExtents, replace);
      return;
    }
  }

  // A dummy from another entry point that is not declared in the current
  // entry point requires a skeleton definition. Most such "unused" dummies
  // will not survive into final generated code, but some will. It is illegal
  // to reference one at run time if it does. Such a dummy is mapped to a
  // value in one of three ways:
  //
  //  - Generate a fir::UndefOp value. This is lightweight, easy to clean up,
  //    and often valid, but it may fail for a dummy with dynamic bounds,
  //    or a dummy used to define another dummy. Information to distinguish
  //    valid cases is not generally available here, with the exception of
  //    dummy procedures. See the first function exit above.
  //
  //  - Allocate an uninitialized stack slot. This is an intermediate-weight
  //    solution that is harder to clean up. It is often valid, but may fail
  //    for an object with dynamic bounds. This option is "automatically"
  //    used by default for cases that do not use one of the other options.
  //
  //  - Allocate a heap box/descriptor, initialized to zero. This always
  //    works, but is more heavyweight and harder to clean up. It is used
  //    for dynamic objects via calls to genUnusedEntryPointBox.

  auto genUnusedEntryPointBox = [&]() {
    if (isUnusedEntryDummy) {
      assert(!Fortran::semantics::IsAllocatableOrPointer(sym) &&
             "handled above");
      // The box is read right away because lowering code does not expect
      // a non pointer/allocatable symbol to be mapped to a MutableBox.
      mlir::Type ty = converter.genType(var);
      bool isPolymorphic = false;
      if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(ty)) {
        isPolymorphic = mlir::isa<fir::ClassType>(ty);
        ty = boxTy.getEleTy();
      }
      Fortran::lower::genDeclareSymbol(
          converter, symMap, sym,
          fir::factory::genMutableBoxRead(
              builder, loc,
              fir::factory::createTempMutableBox(builder, loc, ty, {}, {},
                                                 isPolymorphic)),
          fir::FortranVariableFlagsEnum::None,
          converter.isRegisteredDummySymbol(sym));
      return true;
    }
    return false;
  };

  if (isAssumedRank) {
    assert(isUnusedEntryDummy && "assumed rank must be pointers/allocatables "
                                 "or descriptor dummy arguments");
    genUnusedEntryPointBox();
    return;
  }

  // Helper to generate scalars for the symbol properties.
  auto genValue = [&](const Fortran::lower::SomeExpr &expr) {
    return genScalarValue(converter, loc, expr, symMap, stmtCtx);
  };

  // For symbols reaching this point, all properties are constant and can be
  // read/computed already into ssa values.

  // The origin must be \vec{1}.
  auto populateShape = [&](auto &shapes, const auto &bounds, mlir::Value box) {
    for (auto iter : llvm::enumerate(bounds)) {
      auto *spec = iter.value();
      assert(spec->lbound().GetExplicit() &&
             "lbound must be explicit with constant value 1");
      if (auto high = spec->ubound().GetExplicit()) {
        Fortran::lower::SomeExpr highEx{*high};
        mlir::Value ub = genValue(highEx);
        ub = builder.createConvert(loc, idxTy, ub);
        shapes.emplace_back(fir::factory::genMaxWithZero(builder, loc, ub));
      } else if (spec->ubound().isColon()) {
        assert(box && "assumed bounds require a descriptor");
        mlir::Value dim =
            builder.createIntegerConstant(loc, idxTy, iter.index());
        auto dimInfo =
            fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy, box, dim);
        shapes.emplace_back(dimInfo.getResult(1));
      } else if (spec->ubound().isStar()) {
        shapes.emplace_back(getAssumedSizeExtent(loc, builder));
      } else {
        llvm::report_fatal_error("unknown bound category");
      }
    }
  };

  // The origin is not \vec{1}.
  auto populateLBoundsExtents = [&](auto &lbounds, auto &extents,
                                    const auto &bounds, mlir::Value box) {
    for (auto iter : llvm::enumerate(bounds)) {
      auto *spec = iter.value();
      fir::BoxDimsOp dimInfo;
      mlir::Value ub, lb;
      if (spec->lbound().isColon() || spec->ubound().isColon()) {
        // This is an assumed shape because allocatables and pointers extents
        // are not constant in the scope and are not read here.
        assert(box && "deferred bounds require a descriptor");
        mlir::Value dim =
            builder.createIntegerConstant(loc, idxTy, iter.index());
        dimInfo =
            fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy, box, dim);
        extents.emplace_back(dimInfo.getResult(1));
        if (auto low = spec->lbound().GetExplicit()) {
          auto expr = Fortran::lower::SomeExpr{*low};
          mlir::Value lb = builder.createConvert(loc, idxTy, genValue(expr));
          lbounds.emplace_back(lb);
        } else {
          // Implicit lower bound is 1 (Fortran 2018 section 8.5.8.3 point 3.)
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, 1));
        }
      } else {
        if (auto low = spec->lbound().GetExplicit()) {
          auto expr = Fortran::lower::SomeExpr{*low};
          lb = builder.createConvert(loc, idxTy, genValue(expr));
        } else {
          TODO(loc, "support for assumed rank entities");
        }
        lbounds.emplace_back(lb);

        if (auto high = spec->ubound().GetExplicit()) {
          auto expr = Fortran::lower::SomeExpr{*high};
          ub = builder.createConvert(loc, idxTy, genValue(expr));
          extents.emplace_back(
              fir::factory::computeExtent(builder, loc, lb, ub));
        } else {
          // An assumed size array. The extent is not computed.
          assert(spec->ubound().isStar() && "expected assumed size");
          extents.emplace_back(getAssumedSizeExtent(loc, builder));
        }
      }
    }
  };

  //===--------------------------------------------------------------===//
  // Non Pointer non allocatable scalar, explicit shape, and assumed
  // size arrays.
  // Lower the specification expressions.
  //===--------------------------------------------------------------===//

  mlir::Value len;
  llvm::SmallVector<mlir::Value> extents;
  llvm::SmallVector<mlir::Value> lbounds;
  auto arg = symMap.lookupSymbol(sym).getAddr();
  mlir::Value addr = preAlloc;

  if (arg)
    if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(arg.getType())) {
      // Contiguous assumed shape that can be tracked without a fir.box.
      mlir::Type refTy = builder.getRefType(boxTy.getEleTy());
      addr = fir::BoxAddrOp::create(builder, loc, refTy, arg);
    }

  // Compute/Extract character length.
  if (ba.isChar()) {
    if (arg) {
      assert(!preAlloc && "dummy cannot be pre-allocated");
      if (mlir::isa<fir::BoxCharType>(arg.getType())) {
        std::tie(addr, len) = charHelp.createUnboxChar(arg);
      } else if (mlir::isa<fir::CharacterType>(arg.getType())) {
        // fir.char<1> passed by value (BIND(C) with VALUE attribute).
        addr = fir::AllocaOp::create(builder, loc, arg.getType());
        fir::StoreOp::create(builder, loc, arg, addr);
      } else if (!addr) {
        addr = arg;
      }
      // Ensure proper type is given to array/scalar that was transmitted as a
      // fir.boxchar arg or is a statement function actual argument with
      // a different length than the dummy.
      mlir::Type castTy = builder.getRefType(converter.genType(var));
      addr = builder.createConvert(loc, castTy, addr);
    }
    if (std::optional<int64_t> cstLen = ba.getCharLenConst()) {
      // Static length
      len = builder.createIntegerConstant(loc, idxTy, *cstLen);
    } else {
      // Dynamic length
      if (genUnusedEntryPointBox())
        return;
      if (std::optional<Fortran::lower::SomeExpr> charLenExpr =
              ba.getCharLenExpr()) {
        // Explicit length
        mlir::Value rawLen = genValue(*charLenExpr);
        // If the length expression is negative, the length is zero. See
        // F2018 7.4.4.2 point 5.
        len = fir::factory::genMaxWithZero(builder, loc, rawLen);
      } else if (!len) {
        // Assumed length fir.box (possible for contiguous assumed shapes).
        // Read length from box.
        assert(arg && mlir::isa<fir::BoxType>(arg.getType()) &&
               "must be character dummy fir.box");
        len = charHelp.readLengthFromBox(arg);
      }
    }
  }

  // Compute array extents and lower bounds.
  if (ba.isArray()) {
    if (ba.isStaticArray()) {
      if (ba.lboundIsAllOnes()) {
        for (std::int64_t extent :
             recoverShapeVector(ba.staticShape(), preAlloc))
          extents.push_back(genExtentValue(builder, loc, idxTy, extent));
      } else {
        for (auto [lb, extent] :
             llvm::zip(ba.staticLBound(),
                       recoverShapeVector(ba.staticShape(), preAlloc))) {
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, lb));
          extents.emplace_back(genExtentValue(builder, loc, idxTy, extent));
        }
      }
    } else {
      // Non compile time constant shape.
      if (genUnusedEntryPointBox())
        return;
      if (ba.lboundIsAllOnes())
        populateShape(extents, ba.dynamicBound(), arg);
      else
        populateLBoundsExtents(lbounds, extents, ba.dynamicBound(), arg);
    }
  }

  // Allocate or extract raw address for the entity
  if (!addr) {
    if (arg) {
      mlir::Type argType = arg.getType();
      const bool isCptrByVal = Fortran::semantics::IsBuiltinCPtr(sym) &&
                               Fortran::lower::isCPtrArgByValueType(argType);
      if (isCptrByVal || !fir::conformsWithPassByRef(argType)) {
        // Dummy argument passed in register. Place the value in memory at that
        // point since lowering expect symbols to be mapped to memory addresses.
        mlir::Type symType = converter.genType(sym);
        addr = fir::AllocaOp::create(builder, loc, symType);
        if (isCptrByVal) {
          // Place the void* address into the CPTR address component.
          mlir::Value addrComponent =
              fir::factory::genCPtrOrCFunptrAddr(builder, loc, addr, symType);
          builder.createStoreWithConvert(loc, arg, addrComponent);
        } else {
          builder.createStoreWithConvert(loc, arg, addr);
        }
      } else {
        // Dummy address, or address of result whose storage is passed by the
        // caller.
        assert(fir::isa_ref_type(argType) && "must be a memory address");
        addr = arg;
      }
    } else {
      // Local variables
      llvm::SmallVector<mlir::Value> typeParams;
      if (len)
        typeParams.emplace_back(len);
      addr = createNewLocal(converter, loc, var, preAlloc, extents, typeParams);
    }
  }

  ::genDeclareSymbol(converter, symMap, sym, addr, len, extents, lbounds,
                     replace);
  return;
}

void Fortran::lower::defineModuleVariable(
    AbstractConverter &converter, const Fortran::lower::pft::Variable &var) {
  // Use empty linkage for module variables, which makes them available
  // for use in another unit.
  mlir::StringAttr linkage = getLinkageAttribute(converter, var);
  if (!var.isGlobal())
    fir::emitFatalError(converter.getCurrentLocation(),
                        "attempting to lower module variable as local");
  // Define aggregate storages for equivalenced objects.
  if (var.isAggregateStore()) {
    const Fortran::lower::pft::Variable::AggregateStore &aggregate =
        var.getAggregateStore();
    std::string aggName = mangleGlobalAggregateStore(converter, aggregate);
    defineGlobalAggregateStore(converter, aggregate, aggName, linkage);
    return;
  }
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  if (const Fortran::semantics::Symbol *common =
          Fortran::semantics::FindCommonBlockContaining(var.getSymbol())) {
    // Nothing to do, common block are generated before everything. Ensure
    // this was done by calling getCommonBlockGlobal.
    getCommonBlockGlobal(converter, *common);
  } else if (var.isAlias()) {
    // Do nothing. Mapping will be done on user side.
  } else {
    std::string globalName = converter.mangleName(sym);
    cuf::DataAttributeAttr dataAttr =
        Fortran::lower::translateSymbolCUFDataAttribute(
            converter.getFirOpBuilder().getContext(), sym);
    defineGlobal(converter, var, globalName, linkage, dataAttr);
  }
}

void Fortran::lower::instantiateVariable(AbstractConverter &converter,
                                         const pft::Variable &var,
                                         Fortran::lower::SymMap &symMap,
                                         AggregateStoreMap &storeMap) {
  if (var.hasSymbol()) {
    // Do not try to instantiate symbols twice, except for dummies and results,
    // that may have been mapped to the MLIR entry block arguments, and for
    // which the explicit specifications, if any, has not yet been lowered.
    const auto &sym = var.getSymbol();
    if (!IsDummy(sym) && !IsFunctionResult(sym) && symMap.lookupSymbol(sym))
      return;
  }
  LLVM_DEBUG(llvm::dbgs() << "instantiateVariable: "; var.dump());
  if (var.isAggregateStore())
    instantiateAggregateStore(converter, var, storeMap);
  else if (const Fortran::semantics::Symbol *common =
               Fortran::semantics::FindCommonBlockContaining(
                   var.getSymbol().GetUltimate()))
    instantiateCommon(converter, *common, var, symMap);
  else if (var.isAlias())
    instantiateAlias(converter, var, symMap, storeMap);
  else if (var.isGlobal())
    instantiateGlobal(converter, var, symMap);
  else
    instantiateLocal(converter, var, symMap);
}

static void
mapCallInterfaceSymbol(const Fortran::semantics::Symbol &interfaceSymbol,
                       Fortran::lower::AbstractConverter &converter,
                       const Fortran::lower::CallerInterface &caller,
                       Fortran::lower::SymMap &symMap) {
  Fortran::lower::AggregateStoreMap storeMap;
  for (Fortran::lower::pft::Variable var :
       Fortran::lower::pft::getDependentVariableList(interfaceSymbol)) {
    if (var.isAggregateStore()) {
      instantiateVariable(converter, var, symMap, storeMap);
      continue;
    }
    const Fortran::semantics::Symbol &sym = var.getSymbol();
    if (&sym == &interfaceSymbol)
      continue;
    const auto *hostDetails =
        sym.detailsIf<Fortran::semantics::HostAssocDetails>();
    if (hostDetails && !var.isModuleOrSubmoduleVariable()) {
      // The callee is an internal procedure `A` whose result properties
      // depend on host variables. The caller may be the host, or another
      // internal procedure `B` contained in the same host. In the first
      // case, the host symbol is obviously mapped, in the second case, it
      // must also be mapped because
      // HostAssociations::internalProcedureBindings that was called when
      // lowering `B` will have mapped all host symbols of captured variables
      // to the tuple argument containing the composite of all host associated
      // variables, whether or not the host symbol is actually referred to in
      // `B`. Hence it is possible to simply lookup the variable associated to
      // the host symbol without having to go back to the tuple argument.
      symMap.copySymbolBinding(hostDetails->symbol(), sym);
      // The SymbolBox associated to the host symbols is complete, skip
      // instantiateVariable that would try to allocate a new storage.
      continue;
    }
    if (Fortran::semantics::IsDummy(sym) &&
        sym.owner() == interfaceSymbol.owner()) {
      // Get the argument for the dummy argument symbols of the current call.
      symMap.addSymbol(sym, caller.getArgumentValue(sym));
      // All the properties of the dummy variable may not come from the actual
      // argument, let instantiateVariable handle this.
    }
    // If this is neither a host associated or dummy symbol, it must be a
    // module or common block variable to satisfy specification expression
    // requirements in 10.1.11, instantiateVariable will get its address and
    // properties.
    instantiateVariable(converter, var, symMap, storeMap);
  }
}

void Fortran::lower::mapCallInterfaceSymbolsForResult(
    AbstractConverter &converter, const Fortran::lower::CallerInterface &caller,
    SymMap &symMap) {
  const Fortran::semantics::Symbol &result = caller.getResultSymbol();
  mapCallInterfaceSymbol(result, converter, caller, symMap);
}

void Fortran::lower::mapCallInterfaceSymbolsForDummyArgument(
    AbstractConverter &converter, const Fortran::lower::CallerInterface &caller,
    SymMap &symMap, const Fortran::semantics::Symbol &dummySymbol) {
  mapCallInterfaceSymbol(dummySymbol, converter, caller, symMap);
}

void Fortran::lower::mapSymbolAttributes(
    AbstractConverter &converter, const Fortran::semantics::SymbolRef &symbol,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
    mlir::Value preAlloc) {
  mapSymbolAttributes(converter, pft::Variable{symbol}, symMap, stmtCtx,
                      preAlloc);
}

void Fortran::lower::createIntrinsicModuleGlobal(
    Fortran::lower::AbstractConverter &converter, const pft::Variable &var) {
  defineGlobal(converter, var, converter.mangleName(var.getSymbol()),
               converter.getFirOpBuilder().createLinkOnceODRLinkage());
}

void Fortran::lower::createRuntimeTypeInfoGlobal(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::semantics::Symbol &typeInfoSym) {
  std::string globalName = converter.mangleName(typeInfoSym);
  auto var = Fortran::lower::pft::Variable(typeInfoSym, /*global=*/true);
  mlir::StringAttr linkage = getLinkageAttribute(converter, var);
  defineGlobal(converter, var, globalName, linkage);
}

mlir::Type Fortran::lower::getCrayPointeeBoxType(mlir::Type fortranType) {
  mlir::Type baseType = hlfir::getFortranElementOrSequenceType(fortranType);
  if (auto seqType = mlir::dyn_cast<fir::SequenceType>(baseType)) {
    // The pointer box's sequence type must be with unknown shape.
    llvm::SmallVector<int64_t> shape(seqType.getDimension(),
                                     fir::SequenceType::getUnknownExtent());
    baseType = fir::SequenceType::get(shape, seqType.getEleTy());
  }
  return fir::BoxType::get(fir::PointerType::get(baseType));
}

fir::ExtendedValue
Fortran::lower::genPackArray(Fortran::lower::AbstractConverter &converter,
                             const Fortran::semantics::Symbol &sym,
                             fir::ExtendedValue exv) {
  if (!needsRepack(converter, sym))
    return exv;

  auto &opts = converter.getLoweringOptions();
  llvm::SmallVector<mlir::Value> lenParams;
  exv.match(
      [&](const fir::CharArrayBoxValue &box) {
        lenParams.emplace_back(box.getLen());
      },
      [&](const fir::BoxValue &box) {
        lenParams.append(box.getExplicitParameters().begin(),
                         box.getExplicitParameters().end());
      },
      [](const auto &) {
        llvm_unreachable("unexpected lowering for assumed-shape dummy");
      });
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  const mlir::Location loc = genLocation(converter, sym);
  bool stackAlloc = opts.getStackRepackArrays();
  // 1D arrays must always use 'whole' mode.
  bool isInnermostMode = !opts.getRepackArraysWhole() && sym.Rank() > 1;
  // Avoid copy-in for 'intent(out)' variable, unless this is a dummy
  // argument with INTENT(OUT) that needs finalization on entry
  // to the subprogram. The finalization routine may read the initial
  // value of the array.
  bool noCopy = Fortran::semantics::IsIntentOut(sym) &&
                !needDummyIntentoutFinalization(sym);
  auto boxType = mlir::cast<fir::BaseBoxType>(fir::getBase(exv).getType());
  mlir::Type elementType = boxType.unwrapInnerType();
  llvm::SmallVector<mlir::Value> elidedLenParams =
      fir::factory::elideLengthsAlreadyInType(elementType, lenParams);
  auto packOp = fir::PackArrayOp::create(
      builder, loc, fir::getBase(exv), stackAlloc, isInnermostMode, noCopy,
      /*max_size=*/mlir::IntegerAttr{},
      /*max_element_size=*/mlir::IntegerAttr{},
      /*min_stride=*/mlir::IntegerAttr{}, fir::PackArrayHeuristics::None,
      elidedLenParams, getSafeRepackAttrs(converter));

  mlir::Value newBase = packOp.getResult();
  return exv.match(
      [&](const fir::CharArrayBoxValue &box) -> fir::ExtendedValue {
        return box.clone(newBase);
      },
      [&](const fir::BoxValue &box) -> fir::ExtendedValue {
        return box.clone(newBase);
      },
      [](const auto &) -> fir::ExtendedValue {
        llvm_unreachable("unexpected lowering for assumed-shape dummy");
      });
}

void Fortran::lower::genUnpackArray(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    fir::FortranVariableOpInterface def,
    const Fortran::semantics::Symbol &sym) {
  // Subtle: rely on the fact that the memref of the defining
  // hlfir.declare is a result of fir.pack_array.
  // Alternatively, we can track the pack operation for a symbol
  // via SymMap.
  auto declareOp = mlir::dyn_cast<hlfir::DeclareOp>(def.getOperation());
  assert(declareOp &&
         "cannot find hlfir.declare for an array that needs to be repacked");
  auto packOp = declareOp.getMemref().getDefiningOp<fir::PackArrayOp>();
  assert(packOp && "cannot find fir.pack_array");
  mlir::Value temp = packOp.getResult();
  mlir::Value original = packOp.getArray();
  bool stackAlloc = packOp.getStack();
  // Avoid copy-out for 'intent(in)' variables.
  bool noCopy = Fortran::semantics::IsIntentIn(sym);
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  fir::UnpackArrayOp::create(builder, loc, temp, original, stackAlloc, noCopy,
                             getSafeRepackAttrs(converter));
}
