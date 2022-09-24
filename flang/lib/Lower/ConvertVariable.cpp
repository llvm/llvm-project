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
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Semantics/runtime-type-info.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-variable"

/// Helper to lower a scalar expression using a specific symbol mapping.
static mlir::Value genScalarValue(Fortran::lower::AbstractConverter &converter,
                                  mlir::Location loc,
                                  const Fortran::lower::SomeExpr &expr,
                                  Fortran::lower::SymMap &symMap,
                                  Fortran::lower::StatementContext &context) {
  // This does not use the AbstractConverter member function to override the
  // symbol mapping to be used expression lowering.
  return fir::getBase(Fortran::lower::createSomeExtendedExpression(
      loc, converter, expr, symMap, context));
}
/// Does this variable have a default initialization?
static bool hasDefaultInitialization(const Fortran::semantics::Symbol &sym) {
  if (sym.has<Fortran::semantics::ObjectEntityDetails>() && sym.size())
    if (!Fortran::semantics::IsAllocatableOrPointer(sym))
      if (const Fortran::semantics::DeclTypeSpec *declTypeSpec = sym.GetType())
        if (const Fortran::semantics::DerivedTypeSpec *derivedTypeSpec =
                declTypeSpec->AsDerived())
          return derivedTypeSpec->HasDefaultInitialization();
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

/// Is this a compiler generated symbol to describe derived types ?
static bool isRuntimeTypeInfoData(const Fortran::semantics::Symbol &sym) {
  // So far, use flags to detect if this symbol were generated during
  // semantics::BuildRuntimeDerivedTypeTables(). Scope cannot be used since the
  // symbols are injected in the user scopes defining the described derived
  // types. A robustness improvement for this test could be to get hands on the
  // semantics::RuntimeDerivedTypeTables and to check if the symbol names
  // belongs to this structure.
  return sym.test(Fortran::semantics::Symbol::Flag::CompilerCreated) &&
         sym.test(Fortran::semantics::Symbol::Flag::ReadOnly);
}

static fir::GlobalOp defineGlobal(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::lower::pft::Variable &var,
                                  llvm::StringRef globalName,
                                  mlir::StringAttr linkage);

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
  // Always define linkonce data since it may be optimized out from the module
  // that actually owns the variable if it does not refers to it.
  if (linkage == builder.createLinkOnceODRLinkage() ||
      linkage == builder.createLinkOnceLinkage())
    return defineGlobal(converter, var, globalName, linkage);
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  mlir::Location loc = genLocation(converter, sym);
  // Resolve potential host and module association before checking that this
  // symbol is an object of a function pointer.
  const Fortran::semantics::Symbol &ultimate = sym.GetUltimate();
  if (!ultimate.has<Fortran::semantics::ObjectEntityDetails>() &&
      !Fortran::semantics::IsProcedurePointer(ultimate))
    mlir::emitError(loc, "processing global declaration: symbol '")
        << toStringRef(sym.name()) << "' has unexpected details\n";
  return builder.createGlobal(loc, converter.genType(var), globalName, linkage,
                              mlir::Attribute{}, isConstant(ultimate));
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

static mlir::Type unwrapElementType(mlir::Type type) {
  if (mlir::Type ty = fir::dyn_cast_ptrOrBoxEleTy(type))
    type = ty;
  if (auto seqType = type.dyn_cast<fir::SequenceType>())
    type = seqType.getEleTy();
  return type;
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
  return Fortran::lower::createInitializerAddress(loc, converter, addr,
                                                  globalOpSymMap, stmtCtx);
}

/// create initial-data-target fir.box in a global initializer region.
mlir::Value Fortran::lower::genInitialDataTarget(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Type boxType, const Fortran::lower::SomeExpr &initialTarget) {
  Fortran::lower::SymMap globalOpSymMap;
  Fortran::lower::AggregateStoreMap storeMap;
  Fortran::lower::StatementContext stmtCtx;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
          initialTarget))
    return fir::factory::createUnallocatedBox(builder, loc, boxType,
                                              /*nonDeferredParams=*/llvm::None);
  // Pointer initial data target, and NULL(mold).
  if (const Fortran::semantics::Symbol *sym =
          Fortran::evaluate::GetFirstSymbol(initialTarget)) {
    // Length parameters processing will need care in global initializer
    // context.
    if (hasDerivedTypeWithLengthParameters(*sym))
      TODO(loc, "initial-data-target with derived type length parameters");

    auto var = Fortran::lower::pft::Variable(*sym, /*global=*/true);
    Fortran::lower::instantiateVariable(converter, var, globalOpSymMap,
                                        storeMap);
  }
  mlir::Value box;
  if (initialTarget.Rank() > 0) {
    box = fir::getBase(Fortran::lower::createSomeArrayBox(
        converter, initialTarget, globalOpSymMap, stmtCtx));
  } else {
    fir::ExtendedValue addr = Fortran::lower::createInitializerAddress(
        loc, converter, initialTarget, globalOpSymMap, stmtCtx);
    box = builder.createBox(loc, addr);
  }
  // box is a fir.box<T>, not a fir.box<fir.ptr<T>> as it should to be used
  // for pointers. A fir.convert should not be used here, because it would
  // not actually set the pointer attribute in the descriptor.
  // In a normal context, fir.rebox would be used to set the pointer attribute
  // while copying the projection from another fir.box. But fir.rebox cannot be
  // used in initializer because its current codegen expects that the input
  // fir.box is in memory, which is not the case in initializers.
  // So, just replace the fir.embox that created addr with one with
  // fir.box<fir.ptr<T>> result type.
  // Note that the descriptor cannot have been created with fir.rebox because
  // the initial-data-target cannot be a fir.box itself (it cannot be
  // assumed-shape, deferred-shape, or polymorphic as per C765). However the
  // case where the initial data target is a derived type with length parameters
  // will most likely be a bit trickier, hence the TODO above.

  mlir::Operation *op = box.getDefiningOp();
  if (!op || !mlir::isa<fir::EmboxOp>(*op))
    fir::emitFatalError(
        loc, "fir.box must be created with embox in global initializers");
  mlir::Type targetEleTy = unwrapElementType(box.getType());
  if (!fir::isa_char(targetEleTy))
    return builder.create<fir::EmboxOp>(loc, boxType, op->getOperands(),
                                        op->getAttrs());

  // Handle the character case length particularities: embox takes a length
  // value argument when the result type has unknown length, but not when the
  // result type has constant length. The type of the initial target must be
  // constant length, but the one of the pointer may not be. In this case, a
  // length operand must be added.
  auto targetLen = targetEleTy.cast<fir::CharacterType>().getLen();
  auto ptrLen = unwrapElementType(boxType).cast<fir::CharacterType>().getLen();
  if (ptrLen == targetLen)
    // Nothing to do
    return builder.create<fir::EmboxOp>(loc, boxType, op->getOperands(),
                                        op->getAttrs());
  auto embox = mlir::cast<fir::EmboxOp>(*op);
  auto ptrType = boxType.cast<fir::BoxType>().getEleTy();
  mlir::Value memref = builder.createConvert(loc, ptrType, embox.getMemref());
  if (targetLen == fir::CharacterType::unknownLen())
    // Drop the length argument.
    return builder.create<fir::EmboxOp>(loc, boxType, memref, embox.getShape(),
                                        embox.getSlice());
  // targetLen is constant and ptrLen is unknown. Add a length argument.
  mlir::Value targetLenValue =
      builder.createIntegerConstant(loc, builder.getIndexType(), targetLen);
  return builder.create<fir::EmboxOp>(loc, boxType, memref, embox.getShape(),
                                      embox.getSlice(),
                                      mlir::ValueRange{targetLenValue});
}

static mlir::Value genDefaultInitializerValue(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::semantics::Symbol &sym, mlir::Type symTy,
    Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Type scalarType = symTy;
  fir::SequenceType sequenceType;
  if (auto ty = symTy.dyn_cast<fir::SequenceType>()) {
    sequenceType = ty;
    scalarType = ty.getEleTy();
  }
  // Build a scalar default value of the symbol type, looping through the
  // components to build each component initial value.
  auto recTy = scalarType.cast<fir::RecordType>();
  auto fieldTy = fir::FieldType::get(scalarType.getContext());
  mlir::Value initialValue = builder.create<fir::UndefOp>(loc, scalarType);
  const Fortran::semantics::DeclTypeSpec *declTy = sym.GetType();
  assert(declTy && "var with default initialization must have a type");
  Fortran::semantics::OrderedComponentIterator components(
      declTy->derivedTypeSpec());
  for (const auto &component : components) {
    // Skip parent components, the sub-components of parent types are part of
    // components and will be looped through right after.
    if (component.test(Fortran::semantics::Symbol::Flag::ParentComp))
      continue;
    mlir::Value componentValue;
    llvm::StringRef name = toStringRef(component.name());
    mlir::Type componentTy = recTy.getType(name);
    assert(componentTy && "component not found in type");
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
        componentValue = fir::factory::createUnallocatedBox(
            builder, loc, componentTy, llvm::None);
      } else if (hasDefaultInitialization(component)) {
        // Component type has default initialization.
        componentValue = genDefaultInitializerValue(converter, loc, component,
                                                    componentTy, stmtCtx);
      } else {
        // Component has no initial value.
        componentValue = builder.create<fir::UndefOp>(loc, componentTy);
      }
    } else if (const auto *proc{
                   component
                       .detailsIf<Fortran::semantics::ProcEntityDetails>()}) {
      if (proc->init().has_value())
        TODO(loc, "procedure pointer component default initialization");
      else
        componentValue = builder.create<fir::UndefOp>(loc, componentTy);
    }
    assert(componentValue && "must have been computed");
    componentValue = builder.createConvert(loc, componentTy, componentValue);
    // FIXME: type parameters must come from the derived-type-spec
    auto field = builder.create<fir::FieldIndexOp>(
        loc, fieldTy, name, scalarType,
        /*typeParams=*/mlir::ValueRange{} /*TODO*/);
    initialValue = builder.create<fir::InsertValueOp>(
        loc, recTy, initialValue, componentValue,
        builder.getArrayAttr(field.getAttributes()));
  }

  if (sequenceType) {
    // For arrays, duplicate the scalar value to all elements with an
    // fir.insert_range covering the whole array.
    auto arrayInitialValue = builder.create<fir::UndefOp>(loc, sequenceType);
    llvm::SmallVector<int64_t> rangeBounds;
    for (int64_t extent : sequenceType.getShape()) {
      if (extent == fir::SequenceType::getUnknownExtent())
        TODO(loc,
             "default initial value of array component with length parameters");
      rangeBounds.push_back(0);
      rangeBounds.push_back(extent - 1);
    }
    return builder.create<fir::InsertOnRangeOp>(
        loc, sequenceType, arrayInitialValue, initialValue,
        builder.getIndexVectorAttr(rangeBounds));
  }
  return initialValue;
}

/// Does this global already have an initializer ?
static bool globalIsInitialized(fir::GlobalOp global) {
  return !global.getRegion().empty() || global.getInitVal();
}

/// Call \p genInit to generate code inside \p global initializer region.
void Fortran::lower::createGlobalInitialization(
    fir::FirOpBuilder &builder, fir::GlobalOp global,
    std::function<void(fir::FirOpBuilder &)> genInit) {
  mlir::Region &region = global.getRegion();
  region.push_back(new mlir::Block);
  mlir::Block &block = region.back();
  auto insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&block);
  genInit(builder);
  builder.restoreInsertionPoint(insertPt);
}

/// Create the global op and its init if it has one
static fir::GlobalOp defineGlobal(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::lower::pft::Variable &var,
                                  llvm::StringRef globalName,
                                  mlir::StringAttr linkage) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  mlir::Location loc = genLocation(converter, sym);
  bool isConst = isConstant(sym);
  fir::GlobalOp global = builder.getNamedGlobal(globalName);
  mlir::Type symTy = converter.genType(var);

  if (global && globalIsInitialized(global))
    return global;

  if (Fortran::semantics::IsProcedurePointer(sym))
    TODO(loc, "procedure pointer globals");

  // If this is an array, check to see if we can use a dense attribute
  // with a tensor mlir type.  This optimization currently only supports
  // rank-1 Fortran arrays of integer, real, or logical. The tensor
  // type does not support nested structures which are needed for
  // complex numbers.
  // To get multidimensional arrays to work, we will have to use column major
  // array ordering with the tensor type (so it matches column major ordering
  // with the Fortran fir.array).  By default, tensor types assume row major
  // ordering. How to create this tensor type is to be determined.
  if (symTy.isa<fir::SequenceType>() && sym.Rank() == 1 &&
      !Fortran::semantics::IsAllocatableOrPointer(sym)) {
    mlir::Type eleTy = symTy.cast<fir::SequenceType>().getEleTy();
    if (eleTy.isa<mlir::IntegerType, mlir::FloatType, fir::LogicalType>()) {
      const auto *details =
          sym.detailsIf<Fortran::semantics::ObjectEntityDetails>();
      if (details->init()) {
        global = Fortran::lower::createDenseGlobal(
            loc, symTy, globalName, linkage, isConst, details->init().value(),
            converter);
        if (global) {
          global.setVisibility(mlir::SymbolTable::Visibility::Public);
          return global;
        }
      }
    }
  }
  if (!global)
    global = builder.createGlobal(loc, symTy, globalName, linkage,
                                  mlir::Attribute{}, isConst);
  if (Fortran::semantics::IsAllocatableOrPointer(sym)) {
    const auto *details =
        sym.detailsIf<Fortran::semantics::ObjectEntityDetails>();
    if (details && details->init()) {
      auto expr = *details->init();
      Fortran::lower::createGlobalInitialization(
          builder, global, [&](fir::FirOpBuilder &b) {
            mlir::Value box = Fortran::lower::genInitialDataTarget(
                converter, loc, symTy, expr);
            b.create<fir::HasValueOp>(loc, box);
          });
    } else {
      // Create unallocated/disassociated descriptor if no explicit init
      Fortran::lower::createGlobalInitialization(
          builder, global, [&](fir::FirOpBuilder &b) {
            mlir::Value box =
                fir::factory::createUnallocatedBox(b, loc, symTy, llvm::None);
            b.create<fir::HasValueOp>(loc, box);
          });
    }

  } else if (const auto *details =
                 sym.detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
    if (details->init()) {
      Fortran::lower::createGlobalInitialization(
          builder, global, [&](fir::FirOpBuilder &builder) {
            Fortran::lower::StatementContext stmtCtx(
                /*cleanupProhibited=*/true);
            fir::ExtendedValue initVal = genInitializerExprValue(
                converter, loc, details->init().value(), stmtCtx);
            mlir::Value castTo =
                builder.createConvert(loc, symTy, fir::getBase(initVal));
            builder.create<fir::HasValueOp>(loc, castTo);
          });
    } else if (hasDefaultInitialization(sym)) {
      Fortran::lower::createGlobalInitialization(
          builder, global, [&](fir::FirOpBuilder &builder) {
            Fortran::lower::StatementContext stmtCtx(
                /*cleanupProhibited=*/true);
            mlir::Value initVal =
                genDefaultInitializerValue(converter, loc, sym, symTy, stmtCtx);
            mlir::Value castTo = builder.createConvert(loc, symTy, initVal);
            builder.create<fir::HasValueOp>(loc, castTo);
          });
    }
  } else if (sym.has<Fortran::semantics::CommonBlockDetails>()) {
    mlir::emitError(loc, "COMMON symbol processed elsewhere");
  } else {
    TODO(loc, "global"); // Procedure pointer or something else
  }
  // Creates undefined initializer for globals without initializers
  if (!globalIsInitialized(global)) {
    // TODO: Is it really required to add the undef init if the Public
    // visibility is set ? We need to make sure the global is not optimized out
    // by LLVM if unused in the current compilation unit, but at least for
    // BIND(C) variables, an initial value may be given in another compilation
    // unit (on the C side), and setting an undef init here creates linkage
    // conflicts.
    if (sym.attrs().test(Fortran::semantics::Attr::BIND_C))
      TODO(loc, "BIND(C) module variable linkage");
    Fortran::lower::createGlobalInitialization(
        builder, global, [&](fir::FirOpBuilder &builder) {
          builder.create<fir::HasValueOp>(
              loc, builder.create<fir::UndefOp>(loc, symTy));
        });
  }
  // Set public visibility to prevent global definition to be optimized out
  // even if they have no initializer and are unused in this compilation unit.
  global.setVisibility(mlir::SymbolTable::Visibility::Public);
  return global;
}

/// Return linkage attribute for \p var.
static mlir::StringAttr
getLinkageAttribute(fir::FirOpBuilder &builder,
                    const Fortran::lower::pft::Variable &var) {
  // Runtime type info for a same derived type is identical in each compilation
  // unit. It desired to avoid having to link against module that only define a
  // type. Therefore the runtime type info is generated everywhere it is needed
  // with `linkonce_odr` LLVM linkage.
  if (var.hasSymbol() && isRuntimeTypeInfoData(var.getSymbol()))
    return builder.createLinkOnceODRLinkage();
  if (var.isModuleVariable())
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
  std::string globalName = Fortran::lower::mangle::mangleName(sym);
  mlir::Location loc = genLocation(converter, sym);
  fir::GlobalOp global = builder.getNamedGlobal(globalName);
  mlir::StringAttr linkage = getLinkageAttribute(builder, var);
  if (var.isModuleVariable()) {
    // A module global was or will be defined when lowering the module. Emit
    // only a declaration if the global does not exist at that point.
    global = declareGlobal(converter, var, globalName, linkage);
  } else {
    global = defineGlobal(converter, var, globalName, linkage);
  }
  auto addrOf = builder.create<fir::AddrOfOp>(loc, global.resultType(),
                                              global.getSymbol());
  Fortran::lower::StatementContext stmtCtx;
  mapSymbolAttributes(converter, var, symMap, stmtCtx, addrOf);
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
  std::string nm = Fortran::lower::mangle::mangleName(var.getSymbol());
  mlir::Type ty = converter.genType(var);
  const Fortran::semantics::Symbol &ultimateSymbol =
      var.getSymbol().GetUltimate();
  llvm::StringRef symNm = toStringRef(ultimateSymbol.name());
  bool isTarg = var.isTarget();
  // Let the builder do all the heavy lifting.
  return builder.allocateLocal(loc, ty, nm, symNm, shape, lenParams, isTarg);
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
  // Local variables (including function results), and intent(out) dummies must
  // be default initialized at runtime if their type has default initialization.
  return hasDefaultInitialization(sym);
}

/// Call default initialization runtime routine to initialize \p var.
static void
defaultInitializeAtRuntime(Fortran::lower::AbstractConverter &converter,
                           const Fortran::lower::pft::Variable &var,
                           Fortran::lower::SymMap &symMap) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  fir::ExtendedValue exv = symMap.lookupSymbol(sym).toExtendedValue();
  if (Fortran::semantics::IsOptional(sym)) {
    // 15.5.2.12 point 3, absent optional dummies are not initialized.
    // Creating descriptor/passing null descriptor to the runtime would
    // create runtime crashes.
    auto isPresent = builder.create<fir::IsPresentOp>(loc, builder.getI1Type(),
                                                      fir::getBase(exv));
    builder.genIfThen(loc, isPresent)
        .genThen([&]() {
          auto box = builder.createBox(loc, exv);
          fir::runtime::genDerivedTypeInitialize(builder, loc, box);
        })
        .end();
  } else {
    mlir::Value box = builder.createBox(loc, exv);
    fir::runtime::genDerivedTypeInitialize(builder, loc, box);
  }
}

// Fortran 2018 - 9.7.3.2 point 6
// When a procedure is invoked, any allocated allocatable object that is an
// actual argument corresponding to an INTENT(OUT) allocatable dummy argument
// is deallocated; any allocated allocatable object that is a subobject of an
// actual argument corresponding to an INTENT(OUT) dummy argument is
// deallocated.
static void deallocateIntentOut(Fortran::lower::AbstractConverter &converter,
                                const Fortran::lower::pft::Variable &var,
                                Fortran::lower::SymMap &symMap) {
  if (!var.hasSymbol())
    return;

  const Fortran::semantics::Symbol &sym = var.getSymbol();
  if (Fortran::semantics::IsDummy(sym) &&
      Fortran::semantics::IsIntentOut(sym) &&
      Fortran::semantics::IsAllocatable(sym)) {
    if (auto symbox = symMap.lookupSymbol(sym)) {
      fir::ExtendedValue extVal = symbox.toExtendedValue();
      if (auto mutBox = extVal.getBoxOf<fir::MutableBoxValue>()) {
        // The dummy argument is not passed in the ENTRY so it should not be
        // deallocated.
        if (mlir::Operation *op = mutBox->getAddr().getDefiningOp())
          if (mlir::isa<fir::AllocaOp>(op))
            return;
        mlir::Location loc = converter.getCurrentLocation();
        if (Fortran::semantics::IsOptional(sym)) {
          fir::FirOpBuilder &builder = converter.getFirOpBuilder();
          auto isPresent = builder.create<fir::IsPresentOp>(
              loc, builder.getI1Type(), fir::getBase(extVal));
          builder.genIfThen(loc, isPresent)
              .genThen([&]() { genDeallocateBox(converter, *mutBox, loc); })
              .end();
        } else {
          genDeallocateBox(converter, *mutBox, loc);
        }
      }
    }
  }
}

/// Instantiate a local variable. Precondition: Each variable will be visited
/// such that if its properties depend on other variables, the variables upon
/// which its properties depend will already have been visited.
static void instantiateLocal(Fortran::lower::AbstractConverter &converter,
                             const Fortran::lower::pft::Variable &var,
                             Fortran::lower::SymMap &symMap) {
  assert(!var.isAlias());
  Fortran::lower::StatementContext stmtCtx;
  mapSymbolAttributes(converter, var, symMap, stmtCtx);
  deallocateIntentOut(converter, var, symMap);
  if (mustBeDefaultInitializedAtRuntime(var))
    defaultInitializeAtRuntime(converter, var, symMap);
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
                                           alias.getAlias()};
  auto iter = storeMap.find(key);
  assert(iter != storeMap.end());
  return iter->second;
}

/// Build the name for the storage of a global equivalence.
static std::string mangleGlobalAggregateStore(
    const Fortran::lower::pft::Variable::AggregateStore &st) {
  return Fortran::lower::mangle::mangleName(st.getNamingSymbol());
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
        Fortran::lower::createGlobalInitialization(
            builder, global, [&](fir::FirOpBuilder &builder) {
              Fortran::lower::StatementContext stmtCtx;
              mlir::Value initVal = fir::getBase(genInitializerExprValue(
                  converter, loc, objectDetails->init().value(), stmtCtx));
              builder.create<fir::HasValueOp>(loc, initVal);
            });
        return global;
      }
  // Equivalence has no Fortran initial value. Create an undefined FIR initial
  // value to ensure this is consider an object definition in the IR regardless
  // of the linkage.
  Fortran::lower::createGlobalInitialization(
      builder, global, [&](fir::FirOpBuilder &builder) {
        Fortran::lower::StatementContext stmtCtx;
        mlir::Value initVal = builder.create<fir::UndefOp>(loc, aggTy);
        builder.create<fir::HasValueOp>(loc, initVal);
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
  std::string aggName = mangleGlobalAggregateStore(var.getAggregateStore());
  if (var.isGlobal()) {
    fir::GlobalOp global;
    auto &aggregate = var.getAggregateStore();
    mlir::StringAttr linkage = getLinkageAttribute(builder, var);
    if (var.isModuleVariable()) {
      // A module global was or will be defined when lowering the module. Emit
      // only a declaration if the global does not exist at that point.
      global = declareGlobalAggregateStore(converter, loc, aggregate, aggName,
                                           linkage);
    } else {
      global =
          defineGlobalAggregateStore(converter, aggregate, aggName, linkage);
    }
    auto addr = builder.create<fir::AddrOfOp>(loc, global.resultType(),
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
  mlir::Value local =
      builder.allocateLocal(loc, seqTy, aggName, "", llvm::None, llvm::None,
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
  std::size_t aliasOffset = var.getAlias();
  mlir::Value store = getAggregateStore(storeMap, var);
  mlir::IntegerType i8Ty = builder.getIntegerType(8);
  mlir::Type i8Ptr = builder.getRefType(i8Ty);
  mlir::Value offset = builder.createIntegerConstant(
      loc, idxTy, sym.GetUltimate().offset() - aliasOffset);
  auto ptr = builder.create<fir::CoordinateOp>(loc, i8Ptr, store,
                                               mlir::ValueRange{offset});
  mlir::Value preAlloc =
      castAliasToPointer(builder, loc, converter.genType(sym), ptr);
  Fortran::lower::StatementContext stmtCtx;
  mapSymbolAttributes(converter, var, symMap, stmtCtx, preAlloc);
  // Default initialization is possible for equivalence members: see
  // F2018 19.5.3.4. Note that if several equivalenced entities have
  // default initialization, they must have the same type, and the standard
  // allows the storage to be default initialized several times (this has
  // no consequences other than wasting some execution time). For now,
  // do not try optimizing this to single default initializations of
  // the equivalenced storages. Keep lowering simple.
  if (mustBeDefaultInitializedAtRuntime(var))
    defaultInitializeAtRuntime(converter, var, symMap);
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
  std::string commonName = Fortran::lower::mangle::mangleName(common);
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
  std::string commonName = Fortran::lower::mangle::mangleName(common);
  fir::GlobalOp global = builder.getNamedGlobal(commonName);
  if (global)
    return std::nullopt;
  Fortran::semantics::MutableSymbolVector cmnBlkMems =
      getCommonMembersWithInitAliases(common);
  mlir::Location loc = converter.genLocation(common.name());
  mlir::StringAttr linkage = builder.createCommonLinkage();
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
    auto init = mlir::DenseElementsAttr::get(vecTy, llvm::makeArrayRef(zero));
    builder.createGlobal(loc, commonTy, commonName, linkage, init);
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
  mlir::TupleType commonTy = global.getType().cast<mlir::TupleType>();
  auto initFunc = [&](fir::FirOpBuilder &builder) {
    mlir::IndexType idxTy = builder.getIndexType();
    mlir::Value cb = builder.create<fir::UndefOp>(loc, commonTy);
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
          cb = builder.create<fir::InsertValueOp>(loc, commonTy, cb, castVal,
                                                  builder.getArrayAttr(offVal));
          ++tupIdx;
          offset = mem->offset() + mem->size();
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "}\n");
    builder.create<fir::HasValueOp>(loc, cb);
  };
  Fortran::lower::createGlobalInitialization(builder, global, initFunc);
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
    commonAddr = builder.create<fir::AddrOfOp>(loc, global.resultType(),
                                               global.getSymbol());

    symMap.addSymbol(common, commonAddr);
  }
  std::size_t byteOffset = varSym.GetUltimate().offset();
  mlir::IntegerType i8Ty = builder.getIntegerType(8);
  mlir::Type i8Ptr = builder.getRefType(i8Ty);
  mlir::Type seqTy = builder.getRefType(builder.getVarLenSeqTy(i8Ty));
  mlir::Value base = builder.createConvert(loc, seqTy, commonAddr);
  mlir::Value offs =
      builder.createIntegerConstant(loc, builder.getIndexType(), byteOffset);
  auto varAddr = builder.create<fir::CoordinateOp>(loc, i8Ptr, base,
                                                   mlir::ValueRange{offs});
  mlir::Type symType = converter.genType(var.getSymbol());
  mlir::Value local;
  if (Fortran::semantics::FindEquivalenceSet(var.getSymbol()) != nullptr)
    local = castAliasToPointer(builder, loc, symType, varAddr);
  else
    local = builder.createConvert(loc, builder.getRefType(symType), varAddr);
  Fortran::lower::StatementContext stmtCtx;
  mapSymbolAttributes(converter, var, symMap, stmtCtx, local);
}

//===--------------------------------------------------------------===//
// Lower Variables specification expressions and attributes
//===--------------------------------------------------------------===//

/// Helper to decide if a dummy argument must be tracked in an BoxValue.
static bool lowerToBoxValue(const Fortran::semantics::Symbol &sym,
                            mlir::Value dummyArg) {
  // Only dummy arguments coming as fir.box can be tracked in an BoxValue.
  if (!dummyArg || !dummyArg.getType().isa<fir::BoxType>())
    return false;
  // Non contiguous arrays must be tracked in an BoxValue.
  if (sym.Rank() > 0 && !sym.attrs().test(Fortran::semantics::Attr::CONTIGUOUS))
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

/// Compute extent from lower and upper bound.
static mlir::Value computeExtent(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value lb, mlir::Value ub) {
  mlir::IndexType idxTy = builder.getIndexType();
  // Let the folder deal with the common `ub - <const> + 1` case.
  auto diff = builder.create<mlir::arith::SubIOp>(loc, idxTy, ub, lb);
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  auto rawExtent = builder.create<mlir::arith::AddIOp>(loc, idxTy, diff, one);
  return fir::factory::genMaxWithZero(builder, loc, rawExtent);
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
        result.emplace_back(
            computeExtent(builder, loc, lowerBounds[spec.index()], ub));
    } else if (spec.value()->ubound().isStar()) {
      // Assumed extent is undefined. Must be provided by user's code.
      result.emplace_back(builder.create<fir::UndefOp>(loc, idxTy));
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
  if (llvm::Optional<int64_t> len = box.getCharLenConst())
    return builder.createIntegerConstant(loc, lenTy, *len);
  if (llvm::Optional<Fortran::lower::SomeExpr> lenExpr = box.getCharLenExpr())
    // If the length expression is negative, the length is zero. See F2018
    // 7.4.4.2 point 5.
    return fir::factory::genMaxWithZero(
        builder, loc,
        genScalarValue(converter, loc, *lenExpr, symMap, stmtCtx));
  return mlir::Value{};
}

/// Treat negative values as undefined. Assumed size arrays will return -1 from
/// the front end for example. Using negative values can produce hard to find
/// bugs much further along in the compilation.
static mlir::Value genExtentValue(fir::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Type idxTy,
                                  long frontEndExtent) {
  if (frontEndExtent >= 0)
    return builder.createIntegerConstant(loc, idxTy, frontEndExtent);
  return builder.create<fir::UndefOp>(loc, idxTy);
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

/// Lower specification expressions and attributes of variable \p var and
/// add it to the symbol map.  For a global or an alias, the address must be
/// pre-computed and provided in \p preAlloc.  A dummy argument for the current
/// entry point has already been mapped to an mlir block argument in
/// mapDummiesAndResults.  Its mapping may be updated here.
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
      mlir::Value undefOp = builder.create<fir::UndefOp>(loc, dummyProcType);
      symMap.addSymbol(sym, undefOp);
    }
    if (Fortran::semantics::IsPointer(sym))
      TODO(loc, "procedure pointers");
    return;
  }

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
        TODO(loc, "assumed length character allocatable");
    } else if (const Fortran::semantics::DeclTypeSpec *declTy = sym.GetType()) {
      if (const Fortran::semantics::DerivedTypeSpec *derived =
              declTy->AsDerived())
        if (Fortran::semantics::CountLenParameters(*derived) != 0)
          TODO(loc,
               "derived type allocatable or pointer with length parameters");
    }
    fir::MutableBoxValue box = Fortran::lower::createMutableBox(
        converter, loc, var, boxAlloc, nonDeferredLenParams);
    symMap.addAllocatableOrPointer(var.getSymbol(), box, replace);
    return;
  }

  if (isDummy) {
    mlir::Value dummyArg = symMap.lookupSymbol(sym).getAddr();
    if (lowerToBoxValue(sym, dummyArg)) {
      llvm::SmallVector<mlir::Value> lbounds;
      llvm::SmallVector<mlir::Value> explicitExtents;
      llvm::SmallVector<mlir::Value> explicitParams;
      // Lower lower bounds, explicit type parameters and explicit
      // extents if any.
      if (ba.isChar())
        if (mlir::Value len =
                lowerExplicitCharLen(converter, loc, ba, symMap, stmtCtx))
          explicitParams.push_back(len);
      // TODO: derived type length parameters.
      lowerExplicitLowerBounds(converter, loc, ba, lbounds, symMap, stmtCtx);
      lowerExplicitExtents(converter, loc, ba, lbounds, explicitExtents, symMap,
                           stmtCtx);
      symMap.addBoxSymbol(sym, dummyArg, lbounds, explicitParams,
                          explicitExtents, replace);
      return;
    }
  }

  // A dummy from another entry point that is not declared in the current
  // entry point requires a skeleton definition.  Most such "unused" dummies
  // will not survive into final generated code, but some will.  It is illegal
  // to reference one at run time if it does.  Such a dummy is mapped to a
  // value in one of three ways:
  //
  //  - Generate a fir::UndefOp value.  This is lightweight, easy to clean up,
  //    and often valid, but it may fail for a dummy with dynamic bounds,
  //    or a dummy used to define another dummy.  Information to distinguish
  //    valid cases is not generally available here, with the exception of
  //    dummy procedures.  See the first function exit above.
  //
  //  - Allocate an uninitialized stack slot.  This is an intermediate-weight
  //    solution that is harder to clean up.  It is often valid, but may fail
  //    for an object with dynamic bounds.  This option is "automatically"
  //    used by default for cases that do not use one of the other options.
  //
  //  - Allocate a heap box/descriptor, initialized to zero.  This always
  //    works, but is more heavyweight and harder to clean up.  It is used
  //    for dynamic objects via calls to genUnusedEntryPointBox.

  auto genUnusedEntryPointBox = [&]() {
    if (isUnusedEntryDummy) {
      assert(!Fortran::semantics::IsAllocatableOrPointer(sym) &&
             "handled above");
      // The box is read right away because lowering code does not expect
      // a non pointer/allocatable symbol to be mapped to a MutableBox.
      symMap.addSymbol(sym, fir::factory::genMutableBoxRead(
                                builder, loc,
                                fir::factory::createTempMutableBox(
                                    builder, loc, converter.genType(var))));
      return true;
    }
    return false;
  };

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
            builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, box, dim);
        shapes.emplace_back(dimInfo.getResult(1));
      } else if (spec->ubound().isStar()) {
        shapes.emplace_back(builder.create<fir::UndefOp>(loc, idxTy));
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
            builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, box, dim);
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
          extents.emplace_back(computeExtent(builder, loc, lb, ub));
        } else {
          // An assumed size array. The extent is not computed.
          assert(spec->ubound().isStar() && "expected assumed size");
          extents.emplace_back(builder.create<fir::UndefOp>(loc, idxTy));
        }
      }
    }
  };

  // Lower length expression for non deferred and non dummy assumed length
  // characters.
  auto genExplicitCharLen =
      [&](llvm::Optional<Fortran::lower::SomeExpr> charLen) -> mlir::Value {
    if (!charLen)
      fir::emitFatalError(loc, "expected explicit character length");
    mlir::Value rawLen = genValue(*charLen);
    // If the length expression is negative, the length is zero. See
    // F2018 7.4.4.2 point 5.
    return fir::factory::genMaxWithZero(builder, loc, rawLen);
  };

  ba.match(
      //===--------------------------------------------------------------===//
      // Trivial case.
      //===--------------------------------------------------------------===//
      [&](const Fortran::lower::details::ScalarSym &) {
        if (isDummy) {
          // This is an argument.
          if (!symMap.lookupSymbol(sym))
            mlir::emitError(loc, "symbol \"")
                << toStringRef(sym.name()) << "\" must already be in map";
          return;
        } else if (isResult) {
          // Some Fortran results may be passed by argument (e.g. derived
          // types)
          if (symMap.lookupSymbol(sym))
            return;
        }
        // Otherwise, it's a local variable or function result.
        mlir::Value local = createNewLocal(converter, loc, var, preAlloc);
        symMap.addSymbol(sym, local);
      },

      //===--------------------------------------------------------------===//
      // The non-trivial cases are when we have an argument or local that has
      // a repetition value. Arguments might be passed as simple pointers and
      // need to be cast to a multi-dimensional array with constant bounds
      // (possibly with a missing column), bounds computed in the callee
      // (here), or with bounds from the caller (boxed somewhere else). Locals
      // have the same properties except they are never boxed arguments from
      // the caller and never having a missing column size.
      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::ScalarStaticChar &x) {
        // type is a CHARACTER, determine the LEN value
        auto charLen = x.charLen();
        if (replace) {
          Fortran::lower::SymbolBox symBox = symMap.lookupSymbol(sym);
          if (symBox) {
            std::pair<mlir::Value, mlir::Value> unboxchar =
                charHelp.createUnboxChar(symBox.getAddr());
            mlir::Value boxAddr = unboxchar.first;
            // Set/override LEN with a constant
            mlir::Value len =
                builder.createIntegerConstant(loc, idxTy, charLen);
            symMap.addCharSymbol(sym, boxAddr, len, true);
            return;
          }
        }
        mlir::Value len = builder.createIntegerConstant(loc, idxTy, charLen);
        if (preAlloc) {
          symMap.addCharSymbol(sym, preAlloc, len);
          return;
        }
        mlir::Value local = createNewLocal(converter, loc, var, preAlloc);
        symMap.addCharSymbol(sym, local, len);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::ScalarDynamicChar &x) {
        if (genUnusedEntryPointBox())
          return;
        // type is a CHARACTER, determine the LEN value
        auto charLen = x.charLen();
        if (replace) {
          Fortran::lower::SymbolBox symBox = symMap.lookupSymbol(sym);
          mlir::Value boxAddr = symBox.getAddr();
          mlir::Value len;
          mlir::Type addrTy = boxAddr.getType();
          if (addrTy.isa<fir::BoxCharType>() || addrTy.isa<fir::BoxType>())
            std::tie(boxAddr, len) = charHelp.createUnboxChar(symBox.getAddr());
          // Override LEN with an expression
          if (charLen)
            len = genExplicitCharLen(charLen);
          symMap.addCharSymbol(sym, boxAddr, len, true);
          return;
        }
        // local CHARACTER variable
        mlir::Value len = genExplicitCharLen(charLen);
        if (preAlloc) {
          symMap.addCharSymbol(sym, preAlloc, len);
          return;
        }
        llvm::SmallVector<mlir::Value> lengths = {len};
        mlir::Value local =
            createNewLocal(converter, loc, var, preAlloc, llvm::None, lengths);
        symMap.addCharSymbol(sym, local, len);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::StaticArray &x) {
        // object shape is constant, not a character
        mlir::Type castTy = builder.getRefType(converter.genType(var));
        mlir::Value addr = symMap.lookupSymbol(sym).getAddr();
        if (addr)
          addr = builder.createConvert(loc, castTy, addr);
        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shape;
          for (int64_t i : recoverShapeVector(x.shapes, preAlloc))
            shape.push_back(genExtentValue(builder, loc, idxTy, i));
          mlir::Value local =
              isDummy ? addr : createNewLocal(converter, loc, var, preAlloc);
          symMap.addSymbolWithShape(sym, local, shape, isDummy);
          return;
        }
        // If object is an array process the lower bound and extent values by
        // constructing constants and populating the lbounds and extents.
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;
        for (auto [fst, snd] :
             llvm::zip(x.lbounds, recoverShapeVector(x.shapes, preAlloc))) {
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, fst));
          extents.emplace_back(genExtentValue(builder, loc, idxTy, snd));
        }
        mlir::Value local =
            isDummy ? addr
                    : createNewLocal(converter, loc, var, preAlloc, extents);
        // Must be a dummy argument, have an explicit shape, or be a PARAMETER.
        assert(isDummy || Fortran::lower::isExplicitShape(sym) ||
               Fortran::semantics::IsNamedConstant(sym));
        symMap.addSymbolWithBounds(sym, local, extents, lbounds, isDummy);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::DynamicArray &x) {
        if (genUnusedEntryPointBox())
          return;
        // cast to the known constant parts from the declaration
        mlir::Type varType = converter.genType(var);
        mlir::Value addr = symMap.lookupSymbol(sym).getAddr();
        mlir::Value argBox;
        mlir::Type castTy = builder.getRefType(varType);
        if (addr) {
          if (auto boxTy = addr.getType().dyn_cast<fir::BoxType>()) {
            argBox = addr;
            mlir::Type refTy = builder.getRefType(boxTy.getEleTy());
            addr = builder.create<fir::BoxAddrOp>(loc, refTy, argBox);
          }
          addr = builder.createConvert(loc, castTy, addr);
        }
        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shapes;
          populateShape(shapes, x.bounds, argBox);
          if (isDummy) {
            symMap.addSymbolWithShape(sym, addr, shapes, true);
            return;
          }
          // local array with computed bounds
          assert(Fortran::lower::isExplicitShape(sym) ||
                 Fortran::semantics::IsAllocatableOrPointer(sym));
          mlir::Value local =
              createNewLocal(converter, loc, var, preAlloc, shapes);
          symMap.addSymbolWithShape(sym, local, shapes);
          return;
        }
        // if object is an array process the lower bound and extent values
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;
        populateLBoundsExtents(lbounds, extents, x.bounds, argBox);
        if (isDummy) {
          symMap.addSymbolWithBounds(sym, addr, extents, lbounds, true);
          return;
        }
        // local array with computed bounds
        assert(Fortran::lower::isExplicitShape(sym));
        mlir::Value local =
            createNewLocal(converter, loc, var, preAlloc, extents);
        symMap.addSymbolWithBounds(sym, local, extents, lbounds);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::StaticArrayStaticChar &x) {
        // if element type is a CHARACTER, determine the LEN value
        auto charLen = x.charLen();
        mlir::Value addr;
        mlir::Value len;
        if (isDummy) {
          Fortran::lower::SymbolBox symBox = symMap.lookupSymbol(sym);
          std::pair<mlir::Value, mlir::Value> unboxchar =
              charHelp.createUnboxChar(symBox.getAddr());
          addr = unboxchar.first;
          // Set/override LEN with a constant
          len = builder.createIntegerConstant(loc, idxTy, charLen);
        } else {
          // local CHARACTER variable
          len = builder.createIntegerConstant(loc, idxTy, charLen);
        }

        // object shape is constant
        mlir::Type castTy = builder.getRefType(converter.genType(var));
        if (addr)
          addr = builder.createConvert(loc, castTy, addr);

        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shape;
          for (int64_t i : recoverShapeVector(x.shapes, preAlloc))
            shape.push_back(genExtentValue(builder, loc, idxTy, i));
          mlir::Value local =
              isDummy ? addr : createNewLocal(converter, loc, var, preAlloc);
          symMap.addCharSymbolWithShape(sym, local, len, shape, isDummy);
          return;
        }

        // if object is an array process the lower bound and extent values
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;
        // construct constants and populate `bounds`
        for (auto [fst, snd] :
             llvm::zip(x.lbounds, recoverShapeVector(x.shapes, preAlloc))) {
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, fst));
          extents.emplace_back(genExtentValue(builder, loc, idxTy, snd));
        }

        if (isDummy) {
          symMap.addCharSymbolWithBounds(sym, addr, len, extents, lbounds,
                                         true);
          return;
        }
        // local CHARACTER array with computed bounds
        assert(Fortran::lower::isExplicitShape(sym));
        mlir::Value local =
            createNewLocal(converter, loc, var, preAlloc, extents);
        symMap.addCharSymbolWithBounds(sym, local, len, extents, lbounds);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::StaticArrayDynamicChar &x) {
        if (genUnusedEntryPointBox())
          return;
        mlir::Value addr;
        mlir::Value len;
        [[maybe_unused]] bool mustBeDummy = false;
        auto charLen = x.charLen();
        // if element type is a CHARACTER, determine the LEN value
        if (isDummy) {
          Fortran::lower::SymbolBox symBox = symMap.lookupSymbol(sym);
          std::pair<mlir::Value, mlir::Value> unboxchar =
              charHelp.createUnboxChar(symBox.getAddr());
          addr = unboxchar.first;
          if (charLen) {
            // Set/override LEN with an expression
            len = genExplicitCharLen(charLen);
          } else {
            // LEN is from the boxchar
            len = unboxchar.second;
            mustBeDummy = true;
          }
        } else {
          // local CHARACTER variable
          len = genExplicitCharLen(charLen);
        }
        llvm::SmallVector<mlir::Value> lengths = {len};

        // cast to the known constant parts from the declaration
        mlir::Type castTy = builder.getRefType(converter.genType(var));
        if (addr)
          addr = builder.createConvert(loc, castTy, addr);

        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shape;
          for (int64_t i : recoverShapeVector(x.shapes, preAlloc))
            shape.push_back(genExtentValue(builder, loc, idxTy, i));
          if (isDummy) {
            symMap.addCharSymbolWithShape(sym, addr, len, shape, true);
            return;
          }
          // local CHARACTER array with constant size
          mlir::Value local = createNewLocal(converter, loc, var, preAlloc,
                                             llvm::None, lengths);
          symMap.addCharSymbolWithShape(sym, local, len, shape);
          return;
        }

        // if object is an array process the lower bound and extent values
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;

        // construct constants and populate `bounds`
        for (auto [fst, snd] :
             llvm::zip(x.lbounds, recoverShapeVector(x.shapes, preAlloc))) {
          lbounds.emplace_back(builder.createIntegerConstant(loc, idxTy, fst));
          extents.emplace_back(genExtentValue(builder, loc, idxTy, snd));
        }
        if (isDummy) {
          symMap.addCharSymbolWithBounds(sym, addr, len, extents, lbounds,
                                         true);
          return;
        }
        // local CHARACTER array with computed bounds
        assert((!mustBeDummy) && (Fortran::lower::isExplicitShape(sym)));
        mlir::Value local =
            createNewLocal(converter, loc, var, preAlloc, llvm::None, lengths);
        symMap.addCharSymbolWithBounds(sym, local, len, extents, lbounds);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::DynamicArrayStaticChar &x) {
        if (genUnusedEntryPointBox())
          return;
        mlir::Value addr;
        mlir::Value len;
        mlir::Value argBox;
        auto charLen = x.charLen();
        // if element type is a CHARACTER, determine the LEN value
        if (isDummy) {
          mlir::Value actualArg = symMap.lookupSymbol(sym).getAddr();
          if (auto boxTy = actualArg.getType().dyn_cast<fir::BoxType>()) {
            argBox = actualArg;
            mlir::Type refTy = builder.getRefType(boxTy.getEleTy());
            addr = builder.create<fir::BoxAddrOp>(loc, refTy, argBox);
          } else {
            addr = charHelp.createUnboxChar(actualArg).first;
          }
          // Set/override LEN with a constant
          len = builder.createIntegerConstant(loc, idxTy, charLen);
        } else {
          // local CHARACTER variable
          len = builder.createIntegerConstant(loc, idxTy, charLen);
        }

        // cast to the known constant parts from the declaration
        mlir::Type castTy = builder.getRefType(converter.genType(var));
        if (addr)
          addr = builder.createConvert(loc, castTy, addr);
        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shape;
          populateShape(shape, x.bounds, argBox);
          if (isDummy) {
            symMap.addCharSymbolWithShape(sym, addr, len, shape, true);
            return;
          }
          // local CHARACTER array
          mlir::Value local =
              createNewLocal(converter, loc, var, preAlloc, shape);
          symMap.addCharSymbolWithShape(sym, local, len, shape);
          return;
        }
        // if object is an array process the lower bound and extent values
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;
        populateLBoundsExtents(lbounds, extents, x.bounds, argBox);
        if (isDummy) {
          symMap.addCharSymbolWithBounds(sym, addr, len, extents, lbounds,
                                         true);
          return;
        }
        // local CHARACTER array with computed bounds
        assert(Fortran::lower::isExplicitShape(sym));
        mlir::Value local =
            createNewLocal(converter, loc, var, preAlloc, extents);
        symMap.addCharSymbolWithBounds(sym, local, len, extents, lbounds);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::details::DynamicArrayDynamicChar &x) {
        if (genUnusedEntryPointBox())
          return;
        mlir::Value addr;
        mlir::Value len;
        mlir::Value argBox;
        auto charLen = x.charLen();
        // if element type is a CHARACTER, determine the LEN value
        if (isDummy) {
          mlir::Value actualArg = symMap.lookupSymbol(sym).getAddr();
          if (auto boxTy = actualArg.getType().dyn_cast<fir::BoxType>()) {
            argBox = actualArg;
            mlir::Type refTy = builder.getRefType(boxTy.getEleTy());
            addr = builder.create<fir::BoxAddrOp>(loc, refTy, argBox);
            if (charLen)
              // Set/override LEN with an expression.
              len = genExplicitCharLen(charLen);
            else
              // Get the length from the actual arguments.
              len = charHelp.readLengthFromBox(argBox);
          } else {
            std::pair<mlir::Value, mlir::Value> unboxchar =
                charHelp.createUnboxChar(actualArg);
            addr = unboxchar.first;
            if (charLen) {
              // Set/override LEN with an expression
              len = genExplicitCharLen(charLen);
            } else {
              // Get the length from the actual arguments.
              len = unboxchar.second;
            }
          }
        } else {
          // local CHARACTER variable
          len = genExplicitCharLen(charLen);
        }
        llvm::SmallVector<mlir::Value> lengths = {len};

        // cast to the known constant parts from the declaration
        mlir::Type castTy = builder.getRefType(converter.genType(var));
        if (addr)
          addr = builder.createConvert(loc, castTy, addr);
        if (x.lboundAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value> shape;
          populateShape(shape, x.bounds, argBox);
          if (isDummy) {
            symMap.addCharSymbolWithShape(sym, addr, len, shape, true);
            return;
          }
          // local CHARACTER array
          mlir::Value local =
              createNewLocal(converter, loc, var, preAlloc, shape, lengths);
          symMap.addCharSymbolWithShape(sym, local, len, shape);
          return;
        }
        // Process the lower bound and extent values.
        llvm::SmallVector<mlir::Value> extents;
        llvm::SmallVector<mlir::Value> lbounds;
        populateLBoundsExtents(lbounds, extents, x.bounds, argBox);
        if (isDummy) {
          symMap.addCharSymbolWithBounds(sym, addr, len, extents, lbounds,
                                         true);
          return;
        }
        // local CHARACTER array with computed bounds
        assert(Fortran::lower::isExplicitShape(sym));
        mlir::Value local =
            createNewLocal(converter, loc, var, preAlloc, extents, lengths);
        symMap.addCharSymbolWithBounds(sym, local, len, extents, lbounds);
      },

      //===--------------------------------------------------------------===//

      [&](const Fortran::lower::BoxAnalyzer::None &) {
        mlir::emitError(loc, "symbol analysis failed on ")
            << toStringRef(sym.name());
      });
}

void Fortran::lower::defineModuleVariable(
    AbstractConverter &converter, const Fortran::lower::pft::Variable &var) {
  // Use empty linkage for module variables, which makes them available
  // for use in another unit.
  mlir::StringAttr linkage =
      getLinkageAttribute(converter.getFirOpBuilder(), var);
  if (!var.isGlobal())
    fir::emitFatalError(converter.getCurrentLocation(),
                        "attempting to lower module variable as local");
  // Define aggregate storages for equivalenced objects.
  if (var.isAggregateStore()) {
    const Fortran::lower::pft::Variable::AggregateStore &aggregate =
        var.getAggregateStore();
    std::string aggName = mangleGlobalAggregateStore(aggregate);
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
    std::string globalName = Fortran::lower::mangle::mangleName(sym);
    defineGlobal(converter, var, globalName, linkage);
  }
}

void Fortran::lower::instantiateVariable(AbstractConverter &converter,
                                         const pft::Variable &var,
                                         Fortran::lower::SymMap &symMap,
                                         AggregateStoreMap &storeMap) {
  if (var.isAggregateStore()) {
    instantiateAggregateStore(converter, var, storeMap);
  } else if (const Fortran::semantics::Symbol *common =
                 Fortran::semantics::FindCommonBlockContaining(
                     var.getSymbol().GetUltimate())) {
    instantiateCommon(converter, *common, var, symMap);
  } else if (var.isAlias()) {
    instantiateAlias(converter, var, symMap, storeMap);
  } else if (var.isGlobal()) {
    instantiateGlobal(converter, var, symMap);
  } else {
    instantiateLocal(converter, var, symMap);
  }
}

void Fortran::lower::mapCallInterfaceSymbols(
    AbstractConverter &converter, const Fortran::lower::CallerInterface &caller,
    SymMap &symMap) {
  Fortran::lower::AggregateStoreMap storeMap;
  const Fortran::semantics::Symbol &result = caller.getResultSymbol();
  for (Fortran::lower::pft::Variable var :
       Fortran::lower::pft::buildFuncResultDependencyList(result)) {
    if (var.isAggregateStore()) {
      instantiateVariable(converter, var, symMap, storeMap);
    } else {
      const Fortran::semantics::Symbol &sym = var.getSymbol();
      const auto *hostDetails =
          sym.detailsIf<Fortran::semantics::HostAssocDetails>();
      if (hostDetails && !var.isModuleVariable()) {
        // The callee is an internal procedure `A` whose result properties
        // depend on host variables. The caller may be the host, or another
        // internal procedure `B` contained in the same host.  In the first
        // case, the host symbol is obviously mapped, in the second case, it
        // must also be mapped because
        // HostAssociations::internalProcedureBindings that was called when
        // lowering `B` will have mapped all host symbols of captured variables
        // to the tuple argument containing the composite of all host associated
        // variables, whether or not the host symbol is actually referred to in
        // `B`. Hence it is possible to simply lookup the variable associated to
        // the host symbol without having to go back to the tuple argument.
        Fortran::lower::SymbolBox hostValue =
            symMap.lookupSymbol(hostDetails->symbol());
        assert(hostValue && "callee host symbol must be mapped on caller side");
        symMap.addSymbol(sym, hostValue.toExtendedValue());
        // The SymbolBox associated to the host symbols is complete, skip
        // instantiateVariable that would try to allocate a new storage.
        continue;
      }
      if (Fortran::semantics::IsDummy(sym) && sym.owner() == result.owner()) {
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
}

void Fortran::lower::createRuntimeTypeInfoGlobal(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::semantics::Symbol &typeInfoSym) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  std::string globalName = Fortran::lower::mangle::mangleName(typeInfoSym);
  auto var = Fortran::lower::pft::Variable(typeInfoSym, /*global=*/true);
  mlir::StringAttr linkage = getLinkageAttribute(builder, var);
  defineGlobal(converter, var, globalName, linkage);
}
