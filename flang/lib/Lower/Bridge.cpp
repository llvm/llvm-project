//===-- Bridge.cpp -- bridge to lower to MLIR -----------------------------===//
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

#include "flang/Lower/Bridge.h"
#include "DirectivesCommon.h"
#include "flang/Common/Version.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/Coarray.h"
#include "flang/Lower/ConvertCall.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/Cuda.h"
#include "flang/Lower/HostAssociations.h"
#include "flang/Lower/IO.h"
#include "flang/Lower/IterationSpace.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/OpenACC.h"
#include "flang/Lower/OpenMP.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Assign.h"
#include "flang/Optimizer/Builder/Runtime/Character.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Runtime/EnvironmentDefaults.h"
#include "flang/Optimizer/Builder/Runtime/Main.h"
#include "flang/Optimizer/Builder/Runtime/Ragged.h"
#include "flang/Optimizer/Builder/Runtime/Stop.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Runtime/iostat.h"
#include "flang/Semantics/runtime-type-info.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetMachine.h"
#include <optional>

#define DEBUG_TYPE "flang-lower-bridge"

static llvm::cl::opt<bool> dumpBeforeFir(
    "fdebug-dump-pre-fir", llvm::cl::init(false),
    llvm::cl::desc("dump the Pre-FIR tree prior to FIR generation"));

static llvm::cl::opt<bool> forceLoopToExecuteOnce(
    "always-execute-loop-body", llvm::cl::init(false),
    llvm::cl::desc("force the body of a loop to execute at least once"));

namespace {
/// Information for generating a structured or unstructured increment loop.
struct IncrementLoopInfo {
  template <typename T>
  explicit IncrementLoopInfo(Fortran::semantics::Symbol &sym, const T &lower,
                             const T &upper, const std::optional<T> &step,
                             bool isUnordered = false)
      : loopVariableSym{&sym}, lowerExpr{Fortran::semantics::GetExpr(lower)},
        upperExpr{Fortran::semantics::GetExpr(upper)},
        stepExpr{Fortran::semantics::GetExpr(step)}, isUnordered{isUnordered} {}

  IncrementLoopInfo(IncrementLoopInfo &&) = default;
  IncrementLoopInfo &operator=(IncrementLoopInfo &&x) = default;

  bool isStructured() const { return !headerBlock; }

  mlir::Type getLoopVariableType() const {
    assert(loopVariable && "must be set");
    return fir::unwrapRefType(loopVariable.getType());
  }

  bool hasLocalitySpecs() const {
    return !localSymList.empty() || !localInitSymList.empty() ||
           !reduceSymList.empty() || !sharedSymList.empty();
  }

  // Data members common to both structured and unstructured loops.
  const Fortran::semantics::Symbol *loopVariableSym;
  const Fortran::lower::SomeExpr *lowerExpr;
  const Fortran::lower::SomeExpr *upperExpr;
  const Fortran::lower::SomeExpr *stepExpr;
  const Fortran::lower::SomeExpr *maskExpr = nullptr;
  bool isUnordered; // do concurrent, forall
  llvm::SmallVector<const Fortran::semantics::Symbol *> localSymList;
  llvm::SmallVector<const Fortran::semantics::Symbol *> localInitSymList;
  llvm::SmallVector<
      std::pair<fir::ReduceOperationEnum, const Fortran::semantics::Symbol *>>
      reduceSymList;
  llvm::SmallVector<const Fortran::semantics::Symbol *> sharedSymList;
  mlir::Value loopVariable = nullptr;

  // Data members for structured loops.
  fir::DoLoopOp doLoop = nullptr;

  // Data members for unstructured loops.
  bool hasRealControl = false;
  mlir::Value tripVariable = nullptr;
  mlir::Value stepVariable = nullptr;
  mlir::Block *headerBlock = nullptr; // loop entry and test block
  mlir::Block *maskBlock = nullptr;   // concurrent loop mask block
  mlir::Block *bodyBlock = nullptr;   // first loop body block
  mlir::Block *exitBlock = nullptr;   // loop exit target block
};

/// Information to support stack management, object deallocation, and
/// object finalization at early and normal construct exits.
struct ConstructContext {
  explicit ConstructContext(Fortran::lower::pft::Evaluation &eval,
                            Fortran::lower::StatementContext &stmtCtx)
      : eval{eval}, stmtCtx{stmtCtx} {}

  Fortran::lower::pft::Evaluation &eval;     // construct eval
  Fortran::lower::StatementContext &stmtCtx; // construct exit code
  std::optional<hlfir::Entity> selector;     // construct selector, if any.
  bool pushedScope = false; // was a scoped pushed for this construct?
};

/// Helper to gather the lower bounds of array components with non deferred
/// shape when they are not all ones. Return an empty array attribute otherwise.
static mlir::DenseI64ArrayAttr
gatherComponentNonDefaultLowerBounds(mlir::Location loc,
                                     mlir::MLIRContext *mlirContext,
                                     const Fortran::semantics::Symbol &sym) {
  if (Fortran::semantics::IsAllocatableOrObjectPointer(&sym))
    return {};
  mlir::DenseI64ArrayAttr lbs_attr;
  if (const auto *objDetails =
          sym.detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
    llvm::SmallVector<std::int64_t> lbs;
    bool hasNonDefaultLbs = false;
    for (const Fortran::semantics::ShapeSpec &bounds : objDetails->shape())
      if (auto lb = bounds.lbound().GetExplicit()) {
        if (auto constant = Fortran::evaluate::ToInt64(*lb)) {
          hasNonDefaultLbs |= (*constant != 1);
          lbs.push_back(*constant);
        } else {
          TODO(loc, "generate fir.dt_component for length parametrized derived "
                    "types");
        }
      }
    if (hasNonDefaultLbs) {
      assert(static_cast<int>(lbs.size()) == sym.Rank() &&
             "expected component bounds to be constant or deferred");
      lbs_attr = mlir::DenseI64ArrayAttr::get(mlirContext, lbs);
    }
  }
  return lbs_attr;
}

// Helper class to generate name of fir.global containing component explicit
// default value for objects, and initial procedure target for procedure pointer
// components.
static mlir::FlatSymbolRefAttr gatherComponentInit(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::semantics::Symbol &sym, fir::RecordType derivedType) {
  mlir::MLIRContext *mlirContext = &converter.getMLIRContext();
  // Return procedure target mangled name for procedure pointer components.
  if (const auto *procPtr =
          sym.detailsIf<Fortran::semantics::ProcEntityDetails>()) {
    if (std::optional<const Fortran::semantics::Symbol *> maybeInitSym =
            procPtr->init()) {
      // So far, do not make distinction between p => NULL() and p without init,
      // f18 always initialize pointers to NULL anyway.
      if (!*maybeInitSym)
        return {};
      return mlir::FlatSymbolRefAttr::get(mlirContext,
                                          converter.mangleName(**maybeInitSym));
    }
  }

  const auto *objDetails =
      sym.detailsIf<Fortran::semantics::ObjectEntityDetails>();
  if (!objDetails || !objDetails->init().has_value())
    return {};
  // Object component initial value. Semantic package component object default
  // value into compiler generated symbols that are lowered as read-only
  // fir.global. Get the name of this global.
  std::string name = fir::NameUniquer::getComponentInitName(
      derivedType.getName(), toStringRef(sym.name()));
  return mlir::FlatSymbolRefAttr::get(mlirContext, name);
}

/// Helper class to generate the runtime type info global data and the
/// fir.type_info operations that contain the dipatch tables (if any).
/// The type info global data is required to describe the derived type to the
/// runtime so that it can operate over it.
/// It must be ensured these operations will be generated for every derived type
/// lowered in the current translated unit. However, these operations
/// cannot be generated before FuncOp have been created for functions since the
/// initializers may take their address (e.g for type bound procedures). This
/// class allows registering all the required type info while it is not
/// possible to create GlobalOp/TypeInfoOp, and to generate this data afte
/// function lowering.
class TypeInfoConverter {
  /// Store the location and symbols of derived type info to be generated.
  /// The location of the derived type instantiation is also stored because
  /// runtime type descriptor symbols are compiler generated and cannot be
  /// mapped to user code on their own.
  struct TypeInfo {
    Fortran::semantics::SymbolRef symbol;
    const Fortran::semantics::DerivedTypeSpec &typeSpec;
    fir::RecordType type;
    mlir::Location loc;
  };

public:
  void registerTypeInfo(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc,
                        Fortran::semantics::SymbolRef typeInfoSym,
                        const Fortran::semantics::DerivedTypeSpec &typeSpec,
                        fir::RecordType type) {
    if (seen.contains(typeInfoSym))
      return;
    seen.insert(typeInfoSym);
    currentTypeInfoStack->emplace_back(
        TypeInfo{typeInfoSym, typeSpec, type, loc});
    return;
  }

  void createTypeInfo(Fortran::lower::AbstractConverter &converter) {
    while (!registeredTypeInfoA.empty()) {
      currentTypeInfoStack = &registeredTypeInfoB;
      for (const TypeInfo &info : registeredTypeInfoA)
        createTypeInfoOpAndGlobal(converter, info);
      registeredTypeInfoA.clear();
      currentTypeInfoStack = &registeredTypeInfoA;
      for (const TypeInfo &info : registeredTypeInfoB)
        createTypeInfoOpAndGlobal(converter, info);
      registeredTypeInfoB.clear();
    }
  }

private:
  void createTypeInfoOpAndGlobal(Fortran::lower::AbstractConverter &converter,
                                 const TypeInfo &info) {
    Fortran::lower::createRuntimeTypeInfoGlobal(converter, info.symbol.get());
    createTypeInfoOp(converter, info);
  }

  void createTypeInfoOp(Fortran::lower::AbstractConverter &converter,
                        const TypeInfo &info) {
    fir::RecordType parentType{};
    if (const Fortran::semantics::DerivedTypeSpec *parent =
            Fortran::evaluate::GetParentTypeSpec(info.typeSpec))
      parentType = mlir::cast<fir::RecordType>(converter.genType(*parent));

    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    fir::TypeInfoOp dt;
    mlir::OpBuilder::InsertPoint insertPointIfCreated;
    std::tie(dt, insertPointIfCreated) =
        builder.createTypeInfoOp(info.loc, info.type, parentType);
    if (!insertPointIfCreated.isSet())
      return; // fir.type_info was already built in a previous call.

    // Set init, destroy, and nofinal attributes.
    if (!info.typeSpec.HasDefaultInitialization(/*ignoreAllocatable=*/false,
                                                /*ignorePointer=*/false))
      dt->setAttr(dt.getNoInitAttrName(), builder.getUnitAttr());
    if (!info.typeSpec.HasDestruction())
      dt->setAttr(dt.getNoDestroyAttrName(), builder.getUnitAttr());
    if (!Fortran::semantics::MayRequireFinalization(info.typeSpec))
      dt->setAttr(dt.getNoFinalAttrName(), builder.getUnitAttr());

    const Fortran::semantics::Scope &derivedScope =
        DEREF(info.typeSpec.GetScope());

    // Fill binding table region if the derived type has bindings.
    Fortran::semantics::SymbolVector bindings =
        Fortran::semantics::CollectBindings(derivedScope);
    if (!bindings.empty()) {
      builder.createBlock(&dt.getDispatchTable());
      for (const Fortran::semantics::SymbolRef &binding : bindings) {
        const auto &details =
            binding.get().get<Fortran::semantics::ProcBindingDetails>();
        std::string tbpName = binding.get().name().ToString();
        if (details.numPrivatesNotOverridden() > 0)
          tbpName += "."s + std::to_string(details.numPrivatesNotOverridden());
        std::string bindingName = converter.mangleName(details.symbol());
        builder.create<fir::DTEntryOp>(
            info.loc, mlir::StringAttr::get(builder.getContext(), tbpName),
            mlir::SymbolRefAttr::get(builder.getContext(), bindingName));
      }
      builder.create<fir::FirEndOp>(info.loc);
    }
    // Gather info about components that is not reflected in fir.type and may be
    // needed later: component initial values and array component non default
    // lower bounds.
    mlir::Block *componentInfo = nullptr;
    for (const auto &componentName :
         info.typeSpec.typeSymbol()
             .get<Fortran::semantics::DerivedTypeDetails>()
             .componentNames()) {
      auto scopeIter = derivedScope.find(componentName);
      assert(scopeIter != derivedScope.cend() &&
             "failed to find derived type component symbol");
      const Fortran::semantics::Symbol &component = scopeIter->second.get();
      mlir::FlatSymbolRefAttr init_val =
          gatherComponentInit(info.loc, converter, component, info.type);
      mlir::DenseI64ArrayAttr lbs = gatherComponentNonDefaultLowerBounds(
          info.loc, builder.getContext(), component);
      if (init_val || lbs) {
        if (!componentInfo)
          componentInfo = builder.createBlock(&dt.getComponentInfo());
        auto compName = mlir::StringAttr::get(builder.getContext(),
                                              toStringRef(component.name()));
        builder.create<fir::DTComponentOp>(info.loc, compName, lbs, init_val);
      }
    }
    if (componentInfo)
      builder.create<fir::FirEndOp>(info.loc);
    builder.restoreInsertionPoint(insertPointIfCreated);
  }

  /// Store the front-end data that will be required to generate the type info
  /// for the derived types that have been converted to fir.type<>. There are
  /// two stacks since the type info may visit new types, so the new types must
  /// be added to a new stack.
  llvm::SmallVector<TypeInfo> registeredTypeInfoA;
  llvm::SmallVector<TypeInfo> registeredTypeInfoB;
  llvm::SmallVector<TypeInfo> *currentTypeInfoStack = &registeredTypeInfoA;
  /// Track symbols symbols processed during and after the registration
  /// to avoid infinite loops between type conversions and global variable
  /// creation.
  llvm::SmallSetVector<Fortran::semantics::SymbolRef, 32> seen;
};

using IncrementLoopNestInfo = llvm::SmallVector<IncrementLoopInfo, 8>;
} // namespace

//===----------------------------------------------------------------------===//
// FirConverter
//===----------------------------------------------------------------------===//

namespace {

/// Traverse the pre-FIR tree (PFT) to generate the FIR dialect of MLIR.
class FirConverter : public Fortran::lower::AbstractConverter {
public:
  explicit FirConverter(Fortran::lower::LoweringBridge &bridge)
      : Fortran::lower::AbstractConverter(bridge.getLoweringOptions()),
        bridge{bridge}, foldingContext{bridge.createFoldingContext()},
        mlirSymbolTable{bridge.getModule()} {}
  virtual ~FirConverter() = default;

  /// Convert the PFT to FIR.
  void run(Fortran::lower::pft::Program &pft) {
    // Preliminary translation pass.

    // Lower common blocks, taking into account initialization and the largest
    // size of all instances of each common block. This is done before lowering
    // since the global definition may differ from any one local definition.
    lowerCommonBlocks(pft.getCommonBlocks());

    // - Declare all functions that have definitions so that definition
    //   signatures prevail over call site signatures.
    // - Define module variables and OpenMP/OpenACC declarative constructs so
    //   they are available before lowering any function that may use them.
    bool hasMainProgram = false;
    const Fortran::semantics::Symbol *globalOmpRequiresSymbol = nullptr;
    for (Fortran::lower::pft::Program::Units &u : pft.getUnits()) {
      Fortran::common::visit(
          Fortran::common::visitors{
              [&](Fortran::lower::pft::FunctionLikeUnit &f) {
                if (f.isMainProgram())
                  hasMainProgram = true;
                declareFunction(f);
                if (!globalOmpRequiresSymbol)
                  globalOmpRequiresSymbol = f.getScope().symbol();
              },
              [&](Fortran::lower::pft::ModuleLikeUnit &m) {
                lowerModuleDeclScope(m);
                for (Fortran::lower::pft::ContainedUnit &unit :
                     m.containedUnitList)
                  if (auto *f =
                          std::get_if<Fortran::lower::pft::FunctionLikeUnit>(
                              &unit))
                    declareFunction(*f);
              },
              [&](Fortran::lower::pft::BlockDataUnit &b) {
                if (!globalOmpRequiresSymbol)
                  globalOmpRequiresSymbol = b.symTab.symbol();
              },
              [&](Fortran::lower::pft::CompilerDirectiveUnit &d) {},
              [&](Fortran::lower::pft::OpenACCDirectiveUnit &d) {},
          },
          u);
    }

    // Create definitions of intrinsic module constants.
    createGlobalOutsideOfFunctionLowering(
        [&]() { createIntrinsicModuleDefinitions(pft); });

    // Primary translation pass.
    for (Fortran::lower::pft::Program::Units &u : pft.getUnits()) {
      Fortran::common::visit(
          Fortran::common::visitors{
              [&](Fortran::lower::pft::FunctionLikeUnit &f) { lowerFunc(f); },
              [&](Fortran::lower::pft::ModuleLikeUnit &m) { lowerMod(m); },
              [&](Fortran::lower::pft::BlockDataUnit &b) {},
              [&](Fortran::lower::pft::CompilerDirectiveUnit &d) {},
              [&](Fortran::lower::pft::OpenACCDirectiveUnit &d) {
                builder = new fir::FirOpBuilder(
                    bridge.getModule(), bridge.getKindMap(), &mlirSymbolTable);
                Fortran::lower::genOpenACCRoutineConstruct(
                    *this, bridge.getSemanticsContext(), bridge.getModule(),
                    d.routine, accRoutineInfos);
                builder = nullptr;
              },
          },
          u);
    }

    // Once all the code has been translated, create global runtime type info
    // data structures for the derived types that have been processed, as well
    // as fir.type_info operations for the dispatch tables.
    createGlobalOutsideOfFunctionLowering(
        [&]() { typeInfoConverter.createTypeInfo(*this); });

    // Generate the `main` entry point if necessary
    if (hasMainProgram)
      createGlobalOutsideOfFunctionLowering([&]() {
        fir::runtime::genMain(*builder, toLocation(),
                              bridge.getEnvironmentDefaults());
      });

    finalizeOpenACCLowering();
    finalizeOpenMPLowering(globalOmpRequiresSymbol);
  }

  /// Declare a function.
  void declareFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    setCurrentPosition(funit.getStartingSourceLoc());
    for (int entryIndex = 0, last = funit.entryPointList.size();
         entryIndex < last; ++entryIndex) {
      funit.setActiveEntry(entryIndex);
      // Calling CalleeInterface ctor will build a declaration
      // mlir::func::FuncOp with no other side effects.
      // TODO: when doing some compiler profiling on real apps, it may be worth
      // to check it's better to save the CalleeInterface instead of recomputing
      // it later when lowering the body. CalleeInterface ctor should be linear
      // with the number of arguments, so it is not awful to do it that way for
      // now, but the linear coefficient might be non negligible. Until
      // measured, stick to the solution that impacts the code less.
      Fortran::lower::CalleeInterface{funit, *this};
    }
    funit.setActiveEntry(0);

    // Compute the set of host associated entities from the nested functions.
    llvm::SetVector<const Fortran::semantics::Symbol *> escapeHost;
    for (Fortran::lower::pft::ContainedUnit &unit : funit.containedUnitList)
      if (auto *f = std::get_if<Fortran::lower::pft::FunctionLikeUnit>(&unit))
        collectHostAssociatedVariables(*f, escapeHost);
    funit.setHostAssociatedSymbols(escapeHost);

    // Declare internal procedures
    for (Fortran::lower::pft::ContainedUnit &unit : funit.containedUnitList)
      if (auto *f = std::get_if<Fortran::lower::pft::FunctionLikeUnit>(&unit))
        declareFunction(*f);
  }

  /// Get the scope that is defining or using \p sym. The returned scope is not
  /// the ultimate scope, since this helper does not traverse use association.
  /// This allows capturing module variables that are referenced in an internal
  /// procedure but whose use statement is inside the host program.
  const Fortran::semantics::Scope &
  getSymbolHostScope(const Fortran::semantics::Symbol &sym) {
    const Fortran::semantics::Symbol *hostSymbol = &sym;
    while (const auto *details =
               hostSymbol->detailsIf<Fortran::semantics::HostAssocDetails>())
      hostSymbol = &details->symbol();
    return hostSymbol->owner();
  }

  /// Collects the canonical list of all host associated symbols. These bindings
  /// must be aggregated into a tuple which can then be added to each of the
  /// internal procedure declarations and passed at each call site.
  void collectHostAssociatedVariables(
      Fortran::lower::pft::FunctionLikeUnit &funit,
      llvm::SetVector<const Fortran::semantics::Symbol *> &escapees) {
    const Fortran::semantics::Scope *internalScope =
        funit.getSubprogramSymbol().scope();
    assert(internalScope && "internal procedures symbol must create a scope");
    auto addToListIfEscapee = [&](const Fortran::semantics::Symbol &sym) {
      const Fortran::semantics::Symbol &ultimate = sym.GetUltimate();
      const auto *namelistDetails =
          ultimate.detailsIf<Fortran::semantics::NamelistDetails>();
      if (ultimate.has<Fortran::semantics::ObjectEntityDetails>() ||
          Fortran::semantics::IsProcedurePointer(ultimate) ||
          Fortran::semantics::IsDummy(sym) || namelistDetails) {
        const Fortran::semantics::Scope &symbolScope = getSymbolHostScope(sym);
        if (symbolScope.kind() ==
                Fortran::semantics::Scope::Kind::MainProgram ||
            symbolScope.kind() == Fortran::semantics::Scope::Kind::Subprogram)
          if (symbolScope != *internalScope &&
              symbolScope.Contains(*internalScope)) {
            if (namelistDetails) {
              // So far, namelist symbols are processed on the fly in IO and
              // the related namelist data structure is not added to the symbol
              // map, so it cannot be passed to the internal procedures.
              // Instead, all the symbols of the host namelist used in the
              // internal procedure must be considered as host associated so
              // that IO lowering can find them when needed.
              for (const auto &namelistObject : namelistDetails->objects())
                escapees.insert(&*namelistObject);
            } else {
              escapees.insert(&ultimate);
            }
          }
      }
    };
    Fortran::lower::pft::visitAllSymbols(funit, addToListIfEscapee);
  }

  //===--------------------------------------------------------------------===//
  // AbstractConverter overrides
  //===--------------------------------------------------------------------===//

  mlir::Value getSymbolAddress(Fortran::lower::SymbolRef sym) override final {
    return lookupSymbol(sym).getAddr();
  }

  fir::ExtendedValue
  symBoxToExtendedValue(const Fortran::lower::SymbolBox &symBox) {
    return symBox.match(
        [](const Fortran::lower::SymbolBox::Intrinsic &box)
            -> fir::ExtendedValue { return box.getAddr(); },
        [](const Fortran::lower::SymbolBox::None &) -> fir::ExtendedValue {
          llvm::report_fatal_error("symbol not mapped");
        },
        [&](const fir::FortranVariableOpInterface &x) -> fir::ExtendedValue {
          return hlfir::translateToExtendedValue(getCurrentLocation(),
                                                 getFirOpBuilder(), x);
        },
        [](const auto &box) -> fir::ExtendedValue { return box; });
  }

  fir::ExtendedValue
  getSymbolExtendedValue(const Fortran::semantics::Symbol &sym,
                         Fortran::lower::SymMap *symMap) override final {
    Fortran::lower::SymbolBox sb = lookupSymbol(sym, symMap);
    if (!sb) {
      LLVM_DEBUG(llvm::dbgs() << "unknown symbol: " << sym << "\nmap: "
                              << (symMap ? *symMap : localSymbols) << '\n');
      fir::emitFatalError(getCurrentLocation(),
                          "symbol is not mapped to any IR value");
    }
    return symBoxToExtendedValue(sb);
  }

  mlir::Value impliedDoBinding(llvm::StringRef name) override final {
    mlir::Value val = localSymbols.lookupImpliedDo(name);
    if (!val)
      fir::emitFatalError(toLocation(), "ac-do-variable has no binding");
    return val;
  }

  void copySymbolBinding(Fortran::lower::SymbolRef src,
                         Fortran::lower::SymbolRef target) override final {
    localSymbols.copySymbolBinding(src, target);
  }

  /// Add the symbol binding to the inner-most level of the symbol map and
  /// return true if it is not already present. Otherwise, return false.
  bool bindIfNewSymbol(Fortran::lower::SymbolRef sym,
                       const fir::ExtendedValue &exval) {
    if (shallowLookupSymbol(sym))
      return false;
    bindSymbol(sym, exval);
    return true;
  }

  void bindSymbol(Fortran::lower::SymbolRef sym,
                  const fir::ExtendedValue &exval) override final {
    addSymbol(sym, exval, /*forced=*/true);
  }

  void
  overrideExprValues(const Fortran::lower::ExprToValueMap *map) override final {
    exprValueOverrides = map;
  }

  const Fortran::lower::ExprToValueMap *getExprOverrides() override final {
    return exprValueOverrides;
  }

  bool lookupLabelSet(Fortran::lower::SymbolRef sym,
                      Fortran::lower::pft::LabelSet &labelSet) override final {
    Fortran::lower::pft::FunctionLikeUnit &owningProc =
        *getEval().getOwningProcedure();
    auto iter = owningProc.assignSymbolLabelMap.find(sym);
    if (iter == owningProc.assignSymbolLabelMap.end())
      return false;
    labelSet = iter->second;
    return true;
  }

  Fortran::lower::pft::Evaluation *
  lookupLabel(Fortran::lower::pft::Label label) override final {
    Fortran::lower::pft::FunctionLikeUnit &owningProc =
        *getEval().getOwningProcedure();
    return owningProc.labelEvaluationMap.lookup(label);
  }

  fir::ExtendedValue
  genExprAddr(const Fortran::lower::SomeExpr &expr,
              Fortran::lower::StatementContext &context,
              mlir::Location *locPtr = nullptr) override final {
    mlir::Location loc = locPtr ? *locPtr : toLocation();
    if (lowerToHighLevelFIR())
      return Fortran::lower::convertExprToAddress(loc, *this, expr,
                                                  localSymbols, context);
    return Fortran::lower::createSomeExtendedAddress(loc, *this, expr,
                                                     localSymbols, context);
  }

  fir::ExtendedValue
  genExprValue(const Fortran::lower::SomeExpr &expr,
               Fortran::lower::StatementContext &context,
               mlir::Location *locPtr = nullptr) override final {
    mlir::Location loc = locPtr ? *locPtr : toLocation();
    if (lowerToHighLevelFIR())
      return Fortran::lower::convertExprToValue(loc, *this, expr, localSymbols,
                                                context);
    return Fortran::lower::createSomeExtendedExpression(loc, *this, expr,
                                                        localSymbols, context);
  }

  fir::ExtendedValue
  genExprBox(mlir::Location loc, const Fortran::lower::SomeExpr &expr,
             Fortran::lower::StatementContext &stmtCtx) override final {
    if (lowerToHighLevelFIR())
      return Fortran::lower::convertExprToBox(loc, *this, expr, localSymbols,
                                              stmtCtx);
    return Fortran::lower::createBoxValue(loc, *this, expr, localSymbols,
                                          stmtCtx);
  }

  Fortran::evaluate::FoldingContext &getFoldingContext() override final {
    return foldingContext;
  }

  mlir::Type genType(const Fortran::lower::SomeExpr &expr) override final {
    return Fortran::lower::translateSomeExprToFIRType(*this, expr);
  }
  mlir::Type genType(const Fortran::lower::pft::Variable &var) override final {
    return Fortran::lower::translateVariableToFIRType(*this, var);
  }
  mlir::Type genType(Fortran::lower::SymbolRef sym) override final {
    return Fortran::lower::translateSymbolToFIRType(*this, sym);
  }
  mlir::Type
  genType(Fortran::common::TypeCategory tc, int kind,
          llvm::ArrayRef<std::int64_t> lenParameters) override final {
    return Fortran::lower::getFIRType(&getMLIRContext(), tc, kind,
                                      lenParameters);
  }
  mlir::Type
  genType(const Fortran::semantics::DerivedTypeSpec &tySpec) override final {
    return Fortran::lower::translateDerivedTypeToFIRType(*this, tySpec);
  }
  mlir::Type genType(Fortran::common::TypeCategory tc) override final {
    return Fortran::lower::getFIRType(
        &getMLIRContext(), tc, bridge.getDefaultKinds().GetDefaultKind(tc),
        std::nullopt);
  }

  Fortran::lower::TypeConstructionStack &
  getTypeConstructionStack() override final {
    return typeConstructionStack;
  }

  bool
  isPresentShallowLookup(const Fortran::semantics::Symbol &sym) override final {
    return bool(shallowLookupSymbol(sym));
  }

  bool createHostAssociateVarClone(
      const Fortran::semantics::Symbol &sym) override final {
    mlir::Location loc = genLocation(sym.name());
    mlir::Type symType = genType(sym);
    const auto *details = sym.detailsIf<Fortran::semantics::HostAssocDetails>();
    assert(details && "No host-association found");
    const Fortran::semantics::Symbol &hsym = details->symbol();
    mlir::Type hSymType = genType(hsym.GetUltimate());
    Fortran::lower::SymbolBox hsb =
        lookupSymbol(hsym, /*symMap=*/nullptr, /*forceHlfirBase=*/true);

    auto allocate = [&](llvm::ArrayRef<mlir::Value> shape,
                        llvm::ArrayRef<mlir::Value> typeParams) -> mlir::Value {
      mlir::Value allocVal = builder->allocateLocal(
          loc,
          Fortran::semantics::IsAllocatableOrObjectPointer(&hsym.GetUltimate())
              ? hSymType
              : symType,
          mangleName(sym), toStringRef(sym.GetUltimate().name()),
          /*pinned=*/true, shape, typeParams,
          sym.GetUltimate().attrs().test(Fortran::semantics::Attr::TARGET));
      return allocVal;
    };

    fir::ExtendedValue hexv = symBoxToExtendedValue(hsb);
    fir::ExtendedValue exv = hexv.match(
        [&](const fir::BoxValue &box) -> fir::ExtendedValue {
          const Fortran::semantics::DeclTypeSpec *type = sym.GetType();
          if (type && type->IsPolymorphic())
            TODO(loc, "create polymorphic host associated copy");
          // Create a contiguous temp with the same shape and length as
          // the original variable described by a fir.box.
          llvm::SmallVector<mlir::Value> extents =
              fir::factory::getExtents(loc, *builder, hexv);
          if (box.isDerivedWithLenParameters())
            TODO(loc, "get length parameters from derived type BoxValue");
          if (box.isCharacter()) {
            mlir::Value len = fir::factory::readCharLen(*builder, loc, box);
            mlir::Value temp = allocate(extents, {len});
            return fir::CharArrayBoxValue{temp, len, extents};
          }
          return fir::ArrayBoxValue{allocate(extents, {}), extents};
        },
        [&](const fir::MutableBoxValue &box) -> fir::ExtendedValue {
          // Allocate storage for a pointer/allocatble descriptor.
          // No shape/lengths to be passed to the alloca.
          return fir::MutableBoxValue(allocate({}, {}), {}, {});
        },
        [&](const auto &) -> fir::ExtendedValue {
          mlir::Value temp =
              allocate(fir::factory::getExtents(loc, *builder, hexv),
                       fir::factory::getTypeParams(loc, *builder, hexv));
          return fir::substBase(hexv, temp);
        });

    // Initialise cloned allocatable
    hexv.match(
        [&](const fir::MutableBoxValue &box) -> void {
          // Do not process pointers
          if (Fortran::semantics::IsPointer(sym.GetUltimate())) {
            return;
          }
          // Allocate storage for a pointer/allocatble descriptor.
          // No shape/lengths to be passed to the alloca.
          const auto new_box = exv.getBoxOf<fir::MutableBoxValue>();

          // allocate if allocated
          mlir::Value isAllocated =
              fir::factory::genIsAllocatedOrAssociatedTest(*builder, loc, box);
          auto if_builder = builder->genIfThenElse(loc, isAllocated);
          if_builder.genThen([&]() {
            std::string name = mangleName(sym) + ".alloc";
            fir::ExtendedValue read = fir::factory::genMutableBoxRead(
                *builder, loc, box, /*mayBePolymorphic=*/false);
            if (auto read_arr_box = read.getBoxOf<fir::ArrayBoxValue>()) {
              fir::factory::genInlinedAllocation(
                  *builder, loc, *new_box, read_arr_box->getLBounds(),
                  read_arr_box->getExtents(),
                  /*lenParams=*/std::nullopt, name,
                  /*mustBeHeap=*/true);
            } else if (auto read_char_arr_box =
                           read.getBoxOf<fir::CharArrayBoxValue>()) {
              fir::factory::genInlinedAllocation(
                  *builder, loc, *new_box, read_char_arr_box->getLBounds(),
                  read_char_arr_box->getExtents(), read_char_arr_box->getLen(),
                  name,
                  /*mustBeHeap=*/true);
            } else if (auto read_char_box =
                           read.getBoxOf<fir::CharBoxValue>()) {
              fir::factory::genInlinedAllocation(*builder, loc, *new_box,
                                                 /*lbounds=*/std::nullopt,
                                                 /*extents=*/std::nullopt,
                                                 read_char_box->getLen(), name,
                                                 /*mustBeHeap=*/true);
            } else {
              fir::factory::genInlinedAllocation(
                  *builder, loc, *new_box, box.getMutableProperties().lbounds,
                  box.getMutableProperties().extents,
                  box.nonDeferredLenParams(), name,
                  /*mustBeHeap=*/true);
            }
          });
          if_builder.genElse([&]() {
            // nullify box
            auto empty = fir::factory::createUnallocatedBox(
                *builder, loc, new_box->getBoxTy(),
                new_box->nonDeferredLenParams(), {});
            builder->create<fir::StoreOp>(loc, empty, new_box->getAddr());
          });
          if_builder.end();
        },
        [&](const auto &) -> void {
          // Do nothing
        });

    return bindIfNewSymbol(sym, exv);
  }

  void createHostAssociateVarCloneDealloc(
      const Fortran::semantics::Symbol &sym) override final {
    mlir::Location loc = genLocation(sym.name());
    Fortran::lower::SymbolBox hsb =
        lookupSymbol(sym, /*symMap=*/nullptr, /*forceHlfirBase=*/true);

    fir::ExtendedValue hexv = symBoxToExtendedValue(hsb);
    hexv.match(
        [&](const fir::MutableBoxValue &new_box) -> void {
          // Do not process pointers
          if (Fortran::semantics::IsPointer(sym.GetUltimate())) {
            return;
          }
          // deallocate allocated in createHostAssociateVarClone value
          Fortran::lower::genDeallocateIfAllocated(*this, new_box, loc);
        },
        [&](const auto &) -> void {
          // Do nothing
        });
  }

  void copyVar(mlir::Location loc, mlir::Value dst, mlir::Value src,
               fir::FortranVariableFlagsEnum attrs) override final {
    bool isAllocatable =
        bitEnumContainsAny(attrs, fir::FortranVariableFlagsEnum::allocatable);
    bool isPointer =
        bitEnumContainsAny(attrs, fir::FortranVariableFlagsEnum::pointer);

    copyVarHLFIR(loc, Fortran::lower::SymbolBox::Intrinsic{dst},
                 Fortran::lower::SymbolBox::Intrinsic{src}, isAllocatable,
                 isPointer, Fortran::semantics::Symbol::Flags());
  }

  void copyHostAssociateVar(
      const Fortran::semantics::Symbol &sym,
      mlir::OpBuilder::InsertPoint *copyAssignIP = nullptr) override final {
    // 1) Fetch the original copy of the variable.
    assert(sym.has<Fortran::semantics::HostAssocDetails>() &&
           "No host-association found");
    const Fortran::semantics::Symbol &hsym = sym.GetUltimate();
    Fortran::lower::SymbolBox hsb = lookupOneLevelUpSymbol(hsym);
    assert(hsb && "Host symbol box not found");

    // 2) Fetch the copied one that will mask the original.
    Fortran::lower::SymbolBox sb = shallowLookupSymbol(sym);
    assert(sb && "Host-associated symbol box not found");
    assert(hsb.getAddr() != sb.getAddr() &&
           "Host and associated symbol boxes are the same");

    // 3) Perform the assignment.
    mlir::OpBuilder::InsertPoint insPt = builder->saveInsertionPoint();
    if (copyAssignIP && copyAssignIP->isSet())
      builder->restoreInsertionPoint(*copyAssignIP);
    else
      builder->setInsertionPointAfter(sb.getAddr().getDefiningOp());

    Fortran::lower::SymbolBox *lhs_sb, *rhs_sb;
    if (copyAssignIP && copyAssignIP->isSet() &&
        sym.test(Fortran::semantics::Symbol::Flag::OmpLastPrivate)) {
      // lastprivate case
      lhs_sb = &hsb;
      rhs_sb = &sb;
    } else {
      lhs_sb = &sb;
      rhs_sb = &hsb;
    }

    copyVar(sym, *lhs_sb, *rhs_sb, sym.flags());

    if (copyAssignIP && copyAssignIP->isSet() &&
        sym.test(Fortran::semantics::Symbol::Flag::OmpLastPrivate)) {
      builder->restoreInsertionPoint(insPt);
    }
  }

  void genEval(Fortran::lower::pft::Evaluation &eval,
               bool unstructuredContext) override final {
    genFIR(eval, unstructuredContext);
  }

  //===--------------------------------------------------------------------===//
  // Utility methods
  //===--------------------------------------------------------------------===//

  void collectSymbolSet(
      Fortran::lower::pft::Evaluation &eval,
      llvm::SetVector<const Fortran::semantics::Symbol *> &symbolSet,
      Fortran::semantics::Symbol::Flag flag, bool collectSymbols,
      bool checkHostAssociatedSymbols) override final {
    auto addToList = [&](const Fortran::semantics::Symbol &sym) {
      std::function<void(const Fortran::semantics::Symbol &, bool)>
          insertSymbols = [&](const Fortran::semantics::Symbol &oriSymbol,
                              bool collectSymbol) {
            if (collectSymbol && oriSymbol.test(flag))
              symbolSet.insert(&oriSymbol);
            else if (checkHostAssociatedSymbols)
              if (const auto *details{
                      oriSymbol
                          .detailsIf<Fortran::semantics::HostAssocDetails>()})
                insertSymbols(details->symbol(), true);
          };
      insertSymbols(sym, collectSymbols);
    };
    Fortran::lower::pft::visitAllSymbols(eval, addToList);
  }

  mlir::Location getCurrentLocation() override final { return toLocation(); }

  /// Generate a dummy location.
  mlir::Location genUnknownLocation() override final {
    // Note: builder may not be instantiated yet
    return mlir::UnknownLoc::get(&getMLIRContext());
  }

  static mlir::Location genLocation(Fortran::parser::SourcePosition pos,
                                    mlir::MLIRContext &ctx) {
    llvm::SmallString<256> path(*pos.path);
    llvm::sys::fs::make_absolute(path);
    llvm::sys::path::remove_dots(path);
    return mlir::FileLineColLoc::get(&ctx, path.str(), pos.line, pos.column);
  }

  /// Generate a `Location` from the `CharBlock`.
  mlir::Location
  genLocation(const Fortran::parser::CharBlock &block) override final {
    mlir::Location mainLocation = genUnknownLocation();
    if (const Fortran::parser::AllCookedSources *cooked =
            bridge.getCookedSource()) {
      if (std::optional<Fortran::parser::ProvenanceRange> provenance =
              cooked->GetProvenanceRange(block)) {
        if (std::optional<Fortran::parser::SourcePosition> filePos =
                cooked->allSources().GetSourcePosition(provenance->start()))
          mainLocation = genLocation(*filePos, getMLIRContext());

        llvm::SmallVector<mlir::Location> locs;
        locs.push_back(mainLocation);

        llvm::SmallVector<fir::LocationKindAttr> locAttrs;
        locAttrs.push_back(fir::LocationKindAttr::get(&getMLIRContext(),
                                                      fir::LocationKind::Base));

        // Gather include location information if any.
        Fortran::parser::ProvenanceRange *prov = &*provenance;
        while (prov) {
          if (std::optional<Fortran::parser::ProvenanceRange> include =
                  cooked->allSources().GetInclusionInfo(*prov)) {
            if (std::optional<Fortran::parser::SourcePosition> incPos =
                    cooked->allSources().GetSourcePosition(include->start())) {
              locs.push_back(genLocation(*incPos, getMLIRContext()));
              locAttrs.push_back(fir::LocationKindAttr::get(
                  &getMLIRContext(), fir::LocationKind::Inclusion));
            }
            prov = &*include;
          } else {
            prov = nullptr;
          }
        }
        if (locs.size() > 1) {
          assert(locs.size() == locAttrs.size() &&
                 "expect as many attributes as locations");
          return mlir::FusedLocWith<fir::LocationKindArrayAttr>::get(
              &getMLIRContext(), locs,
              fir::LocationKindArrayAttr::get(&getMLIRContext(), locAttrs));
        }
      }
    }
    return mainLocation;
  }

  const Fortran::semantics::Scope &getCurrentScope() override final {
    return bridge.getSemanticsContext().FindScope(currentPosition);
  }

  fir::FirOpBuilder &getFirOpBuilder() override final { return *builder; }

  mlir::ModuleOp &getModuleOp() override final { return bridge.getModule(); }

  mlir::MLIRContext &getMLIRContext() override final {
    return bridge.getMLIRContext();
  }
  std::string
  mangleName(const Fortran::semantics::Symbol &symbol) override final {
    return Fortran::lower::mangle::mangleName(
        symbol, scopeBlockIdMap, /*keepExternalInScope=*/false,
        getLoweringOptions().getUnderscoring());
  }
  std::string mangleName(
      const Fortran::semantics::DerivedTypeSpec &derivedType) override final {
    return Fortran::lower::mangle::mangleName(derivedType, scopeBlockIdMap);
  }
  std::string mangleName(std::string &name) override final {
    return Fortran::lower::mangle::mangleName(name, getCurrentScope(),
                                              scopeBlockIdMap);
  }
  std::string getRecordTypeFieldName(
      const Fortran::semantics::Symbol &component) override final {
    return Fortran::lower::mangle::getRecordTypeFieldName(component,
                                                          scopeBlockIdMap);
  }
  const fir::KindMapping &getKindMap() override final {
    return bridge.getKindMap();
  }

  /// Return the current function context, which may be a nested BLOCK context
  /// or a full subprogram context.
  Fortran::lower::StatementContext &getFctCtx() override final {
    if (!activeConstructStack.empty() &&
        activeConstructStack.back().eval.isA<Fortran::parser::BlockConstruct>())
      return activeConstructStack.back().stmtCtx;
    return bridge.fctCtx();
  }

  mlir::Value hostAssocTupleValue() override final { return hostAssocTuple; }

  /// Record a binding for the ssa-value of the tuple for this function.
  void bindHostAssocTuple(mlir::Value val) override final {
    assert(!hostAssocTuple && val);
    hostAssocTuple = val;
  }

  mlir::Value dummyArgsScopeValue() const override final {
    return dummyArgsScope;
  }

  bool isRegisteredDummySymbol(
      Fortran::semantics::SymbolRef symRef) const override final {
    auto *sym = &*symRef;
    return registeredDummySymbols.contains(sym);
  }

  void registerTypeInfo(mlir::Location loc,
                        Fortran::lower::SymbolRef typeInfoSym,
                        const Fortran::semantics::DerivedTypeSpec &typeSpec,
                        fir::RecordType type) override final {
    typeInfoConverter.registerTypeInfo(*this, loc, typeInfoSym, typeSpec, type);
  }

  llvm::StringRef
  getUniqueLitName(mlir::Location loc,
                   std::unique_ptr<Fortran::lower::SomeExpr> expr,
                   mlir::Type eleTy) override final {
    std::string namePrefix =
        getConstantExprManglePrefix(loc, *expr.get(), eleTy);
    auto [it, inserted] = literalNamesMap.try_emplace(
        expr.get(), namePrefix + std::to_string(uniqueLitId));
    const auto &name = it->second;
    if (inserted) {
      // Keep ownership of the expr key.
      literalExprsStorage.push_back(std::move(expr));

      // If we've just added a new name, we have to make sure
      // there is no global object with the same name in the module.
      fir::GlobalOp global = builder->getNamedGlobal(name);
      if (global)
        fir::emitFatalError(loc, llvm::Twine("global object with name '") +
                                     llvm::Twine(name) +
                                     llvm::Twine("' already exists"));
      ++uniqueLitId;
      return name;
    }

    // The name already exists. Verify that the prefix is the same.
    if (!llvm::StringRef(name).starts_with(namePrefix))
      fir::emitFatalError(loc, llvm::Twine("conflicting prefixes: '") +
                                   llvm::Twine(name) +
                                   llvm::Twine("' does not start with '") +
                                   llvm::Twine(namePrefix) + llvm::Twine("'"));

    return name;
  }

private:
  FirConverter() = delete;
  FirConverter(const FirConverter &) = delete;
  FirConverter &operator=(const FirConverter &) = delete;

  //===--------------------------------------------------------------------===//
  // Helper member functions
  //===--------------------------------------------------------------------===//

  mlir::Value createFIRExpr(mlir::Location loc,
                            const Fortran::lower::SomeExpr *expr,
                            Fortran::lower::StatementContext &stmtCtx) {
    return fir::getBase(genExprValue(*expr, stmtCtx, &loc));
  }

  /// Find the symbol in the local map or return null.
  Fortran::lower::SymbolBox
  lookupSymbol(const Fortran::semantics::Symbol &sym,
               Fortran::lower::SymMap *symMap = nullptr,
               bool forceHlfirBase = false) {
    symMap = symMap ? symMap : &localSymbols;
    if (lowerToHighLevelFIR()) {
      if (std::optional<fir::FortranVariableOpInterface> var =
              symMap->lookupVariableDefinition(sym)) {
        auto exv = hlfir::translateToExtendedValue(toLocation(), *builder, *var,
                                                   forceHlfirBase);
        return exv.match(
            [](mlir::Value x) -> Fortran::lower::SymbolBox {
              return Fortran::lower::SymbolBox::Intrinsic{x};
            },
            [](auto x) -> Fortran::lower::SymbolBox { return x; });
      }

      // Entry character result represented as an argument pair
      // needs to be represented in the symbol table even before
      // we can create DeclareOp for it. The temporary mapping
      // is EmboxCharOp that conveys the address and length information.
      // After mapSymbolAttributes is done, the mapping is replaced
      // with the new DeclareOp, and the following table lookups
      // do not reach here.
      if (sym.IsFuncResult())
        if (const Fortran::semantics::DeclTypeSpec *declTy = sym.GetType())
          if (declTy->category() ==
              Fortran::semantics::DeclTypeSpec::Category::Character)
            return symMap->lookupSymbol(sym);

      // Procedure dummies are not mapped with an hlfir.declare because
      // they are not "variable" (cannot be assigned to), and it would
      // make hlfir.declare more complex than it needs to to allow this.
      // Do a regular lookup.
      if (Fortran::semantics::IsProcedure(sym))
        return symMap->lookupSymbol(sym);

      // Commonblock names are not variables, but in some lowerings (like
      // OpenMP) it is useful to maintain the address of the commonblock in an
      // MLIR value and query it. hlfir.declare need not be created for these.
      if (sym.detailsIf<Fortran::semantics::CommonBlockDetails>())
        return symMap->lookupSymbol(sym);

      // For symbols to be privatized in OMP, the symbol is mapped to an
      // instance of `SymbolBox::Intrinsic` (i.e. a direct mapping to an MLIR
      // SSA value). This MLIR SSA value is the block argument to the
      // `omp.private`'s `alloc` block. If this is the case, we return this
      // `SymbolBox::Intrinsic` value.
      if (Fortran::lower::SymbolBox v = symMap->lookupSymbol(sym))
        return v;

      return {};
    }
    if (Fortran::lower::SymbolBox v = symMap->lookupSymbol(sym))
      return v;
    return {};
  }

  /// Find the symbol in the inner-most level of the local map or return null.
  Fortran::lower::SymbolBox
  shallowLookupSymbol(const Fortran::semantics::Symbol &sym) {
    if (Fortran::lower::SymbolBox v = localSymbols.shallowLookupSymbol(sym))
      return v;
    return {};
  }

  /// Find the symbol in one level up of symbol map such as for host-association
  /// in OpenMP code or return null.
  Fortran::lower::SymbolBox
  lookupOneLevelUpSymbol(const Fortran::semantics::Symbol &sym) override {
    if (Fortran::lower::SymbolBox v = localSymbols.lookupOneLevelUpSymbol(sym))
      return v;
    return {};
  }

  mlir::SymbolTable *getMLIRSymbolTable() override { return &mlirSymbolTable; }

  /// Add the symbol to the local map and return `true`. If the symbol is
  /// already in the map and \p forced is `false`, the map is not updated.
  /// Instead the value `false` is returned.
  bool addSymbol(const Fortran::semantics::SymbolRef sym,
                 fir::ExtendedValue val, bool forced = false) {
    if (!forced && lookupSymbol(sym))
      return false;
    if (lowerToHighLevelFIR()) {
      Fortran::lower::genDeclareSymbol(*this, localSymbols, sym, val,
                                       fir::FortranVariableFlagsEnum::None,
                                       forced);
    } else {
      localSymbols.addSymbol(sym, val, forced);
    }
    return true;
  }

  void copyVar(const Fortran::semantics::Symbol &sym,
               const Fortran::lower::SymbolBox &lhs_sb,
               const Fortran::lower::SymbolBox &rhs_sb,
               Fortran::semantics::Symbol::Flags flags) {
    mlir::Location loc = genLocation(sym.name());
    if (lowerToHighLevelFIR())
      copyVarHLFIR(loc, lhs_sb, rhs_sb, flags);
    else
      copyVarFIR(loc, sym, lhs_sb, rhs_sb);
  }

  void copyVarHLFIR(mlir::Location loc, Fortran::lower::SymbolBox dst,
                    Fortran::lower::SymbolBox src,
                    Fortran::semantics::Symbol::Flags flags) {
    assert(lowerToHighLevelFIR());

    bool isBoxAllocatable = dst.match(
        [](const fir::MutableBoxValue &box) { return box.isAllocatable(); },
        [](const fir::FortranVariableOpInterface &box) {
          return fir::FortranVariableOpInterface(box).isAllocatable();
        },
        [](const auto &box) { return false; });

    bool isBoxPointer = dst.match(
        [](const fir::MutableBoxValue &box) { return box.isPointer(); },
        [](const fir::FortranVariableOpInterface &box) {
          return fir::FortranVariableOpInterface(box).isPointer();
        },
        [](const auto &box) { return false; });

    copyVarHLFIR(loc, dst, src, isBoxAllocatable, isBoxPointer, flags);
  }

  void copyVarHLFIR(mlir::Location loc, Fortran::lower::SymbolBox dst,
                    Fortran::lower::SymbolBox src, bool isAllocatable,
                    bool isPointer, Fortran::semantics::Symbol::Flags flags) {
    assert(lowerToHighLevelFIR());
    hlfir::Entity lhs{dst.getAddr()};
    hlfir::Entity rhs{src.getAddr()};

    auto copyData = [&](hlfir::Entity l, hlfir::Entity r) {
      // Dereference RHS and load it if trivial scalar.
      r = hlfir::loadTrivialScalar(loc, *builder, r);
      builder->create<hlfir::AssignOp>(loc, r, l, isAllocatable);
    };

    if (isPointer) {
      // Set LHS target to the target of RHS (do not copy the RHS
      // target data into the LHS target storage).
      auto loadVal = builder->create<fir::LoadOp>(loc, rhs);
      builder->create<fir::StoreOp>(loc, loadVal, lhs);
    } else if (isAllocatable &&
               (flags.test(Fortran::semantics::Symbol::Flag::OmpFirstPrivate) ||
                flags.test(Fortran::semantics::Symbol::Flag::OmpCopyIn))) {
      // For firstprivate and copyin allocatable variables, RHS must be copied
      // only when LHS is allocated.
      hlfir::Entity temp =
          hlfir::derefPointersAndAllocatables(loc, *builder, lhs);
      mlir::Value addr = hlfir::genVariableRawAddress(loc, *builder, temp);
      mlir::Value isAllocated = builder->genIsNotNullAddr(loc, addr);
      builder->genIfThen(loc, isAllocated)
          .genThen([&]() { copyData(lhs, rhs); })
          .end();
    } else {
      copyData(lhs, rhs);
    }
  }

  void copyVarFIR(mlir::Location loc, const Fortran::semantics::Symbol &sym,
                  const Fortran::lower::SymbolBox &lhs_sb,
                  const Fortran::lower::SymbolBox &rhs_sb) {
    assert(!lowerToHighLevelFIR());
    fir::ExtendedValue lhs = symBoxToExtendedValue(lhs_sb);
    fir::ExtendedValue rhs = symBoxToExtendedValue(rhs_sb);
    mlir::Type symType = genType(sym);
    if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(symType)) {
      Fortran::lower::StatementContext stmtCtx;
      Fortran::lower::createSomeArrayAssignment(*this, lhs, rhs, localSymbols,
                                                stmtCtx);
      stmtCtx.finalizeAndReset();
    } else if (lhs.getBoxOf<fir::CharBoxValue>()) {
      fir::factory::CharacterExprHelper{*builder, loc}.createAssign(lhs, rhs);
    } else {
      auto loadVal = builder->create<fir::LoadOp>(loc, fir::getBase(rhs));
      builder->create<fir::StoreOp>(loc, loadVal, fir::getBase(lhs));
    }
  }

  /// Map a block argument to a result or dummy symbol. This is not the
  /// definitive mapping. The specification expression have not been lowered
  /// yet. The final mapping will be done using this pre-mapping in
  /// Fortran::lower::mapSymbolAttributes.
  bool mapBlockArgToDummyOrResult(const Fortran::semantics::SymbolRef sym,
                                  mlir::Value val, bool isResult) {
    localSymbols.addSymbol(sym, val);
    if (!isResult)
      registerDummySymbol(sym);

    return true;
  }

  /// Generate the address of loop variable \p sym.
  /// If \p sym is not mapped yet, allocate local storage for it.
  mlir::Value genLoopVariableAddress(mlir::Location loc,
                                     const Fortran::semantics::Symbol &sym,
                                     bool isUnordered) {
    if (isUnordered || sym.has<Fortran::semantics::HostAssocDetails>() ||
        sym.has<Fortran::semantics::UseDetails>()) {
      if (!shallowLookupSymbol(sym) &&
          !sym.test(Fortran::semantics::Symbol::Flag::OmpShared)) {
        // Do concurrent loop variables are not mapped yet since they are local
        // to the Do concurrent scope (same for OpenMP loops).
        mlir::OpBuilder::InsertPoint insPt = builder->saveInsertionPoint();
        builder->setInsertionPointToStart(builder->getAllocaBlock());
        mlir::Type tempTy = genType(sym);
        mlir::Value temp =
            builder->createTemporaryAlloc(loc, tempTy, toStringRef(sym.name()));
        bindIfNewSymbol(sym, temp);
        builder->restoreInsertionPoint(insPt);
      }
    }
    auto entry = lookupSymbol(sym);
    (void)entry;
    assert(entry && "loop control variable must already be in map");
    Fortran::lower::StatementContext stmtCtx;
    return fir::getBase(
        genExprAddr(Fortran::evaluate::AsGenericExpr(sym).value(), stmtCtx));
  }

  static bool isNumericScalarCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Integer ||
           cat == Fortran::common::TypeCategory::Real ||
           cat == Fortran::common::TypeCategory::Complex ||
           cat == Fortran::common::TypeCategory::Logical;
  }
  static bool isLogicalCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Logical;
  }
  static bool isCharacterCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Character;
  }
  static bool isDerivedCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Derived;
  }

  /// Insert a new block before \p block. Leave the insertion point unchanged.
  mlir::Block *insertBlock(mlir::Block *block) {
    mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
    mlir::Block *newBlock = builder->createBlock(block);
    builder->restoreInsertionPoint(insertPt);
    return newBlock;
  }

  Fortran::lower::pft::Evaluation &evalOfLabel(Fortran::parser::Label label) {
    const Fortran::lower::pft::LabelEvalMap &labelEvaluationMap =
        getEval().getOwningProcedure()->labelEvaluationMap;
    const auto iter = labelEvaluationMap.find(label);
    assert(iter != labelEvaluationMap.end() && "label missing from map");
    return *iter->second;
  }

  void genBranch(mlir::Block *targetBlock) {
    assert(targetBlock && "missing unconditional target block");
    builder->create<mlir::cf::BranchOp>(toLocation(), targetBlock);
  }

  void genConditionalBranch(mlir::Value cond, mlir::Block *trueTarget,
                            mlir::Block *falseTarget) {
    assert(trueTarget && "missing conditional branch true block");
    assert(falseTarget && "missing conditional branch false block");
    mlir::Location loc = toLocation();
    mlir::Value bcc = builder->createConvert(loc, builder->getI1Type(), cond);
    builder->create<mlir::cf::CondBranchOp>(loc, bcc, trueTarget, std::nullopt,
                                            falseTarget, std::nullopt);
  }
  void genConditionalBranch(mlir::Value cond,
                            Fortran::lower::pft::Evaluation *trueTarget,
                            Fortran::lower::pft::Evaluation *falseTarget) {
    genConditionalBranch(cond, trueTarget->block, falseTarget->block);
  }
  void genConditionalBranch(const Fortran::parser::ScalarLogicalExpr &expr,
                            mlir::Block *trueTarget, mlir::Block *falseTarget) {
    Fortran::lower::StatementContext stmtCtx;
    mlir::Value cond =
        createFIRExpr(toLocation(), Fortran::semantics::GetExpr(expr), stmtCtx);
    stmtCtx.finalizeAndReset();
    genConditionalBranch(cond, trueTarget, falseTarget);
  }
  void genConditionalBranch(const Fortran::parser::ScalarLogicalExpr &expr,
                            Fortran::lower::pft::Evaluation *trueTarget,
                            Fortran::lower::pft::Evaluation *falseTarget) {
    Fortran::lower::StatementContext stmtCtx;
    mlir::Value cond =
        createFIRExpr(toLocation(), Fortran::semantics::GetExpr(expr), stmtCtx);
    stmtCtx.finalizeAndReset();
    genConditionalBranch(cond, trueTarget->block, falseTarget->block);
  }

  /// Return the nearest active ancestor construct of \p eval, or nullptr.
  Fortran::lower::pft::Evaluation *
  getActiveAncestor(const Fortran::lower::pft::Evaluation &eval) {
    Fortran::lower::pft::Evaluation *ancestor = eval.parentConstruct;
    for (; ancestor; ancestor = ancestor->parentConstruct)
      if (ancestor->activeConstruct)
        break;
    return ancestor;
  }

  /// Return the predicate: "a branch to \p targetEval has exit code".
  bool hasExitCode(const Fortran::lower::pft::Evaluation &targetEval) {
    Fortran::lower::pft::Evaluation *activeAncestor =
        getActiveAncestor(targetEval);
    for (auto it = activeConstructStack.rbegin(),
              rend = activeConstructStack.rend();
         it != rend; ++it) {
      if (&it->eval == activeAncestor)
        break;
      if (it->stmtCtx.hasCode())
        return true;
    }
    return false;
  }

  /// Generate a branch to \p targetEval after generating on-exit code for
  /// any enclosing construct scopes that are exited by taking the branch.
  void
  genConstructExitBranch(const Fortran::lower::pft::Evaluation &targetEval) {
    Fortran::lower::pft::Evaluation *activeAncestor =
        getActiveAncestor(targetEval);
    for (auto it = activeConstructStack.rbegin(),
              rend = activeConstructStack.rend();
         it != rend; ++it) {
      if (&it->eval == activeAncestor)
        break;
      it->stmtCtx.finalizeAndKeep();
    }
    genBranch(targetEval.block);
  }

  /// A construct contains nested evaluations. Some of these evaluations
  /// may start a new basic block, others will add code to an existing
  /// block.
  /// Collect the list of nested evaluations that are last in their block,
  /// organize them into two sets:
  /// 1. Exiting evaluations: they may need a branch exiting from their
  ///    parent construct,
  /// 2. Fall-through evaluations: they will continue to the following
  ///    evaluation. They may still need a branch, but they do not exit
  ///    the construct. They appear in cases where the following evaluation
  ///    is a target of some branch.
  void collectFinalEvaluations(
      Fortran::lower::pft::Evaluation &construct,
      llvm::SmallVector<Fortran::lower::pft::Evaluation *> &exits,
      llvm::SmallVector<Fortran::lower::pft::Evaluation *> &fallThroughs) {
    Fortran::lower::pft::EvaluationList &nested =
        construct.getNestedEvaluations();
    if (nested.empty())
      return;

    Fortran::lower::pft::Evaluation *exit = construct.constructExit;
    Fortran::lower::pft::Evaluation *previous = &nested.front();

    for (auto it = ++nested.begin(), end = nested.end(); it != end;
         previous = &*it++) {
      if (it->block == nullptr)
        continue;
      // "*it" starts a new block, check what to do with "previous"
      if (it->isIntermediateConstructStmt() && previous != exit)
        exits.push_back(previous);
      else if (previous->lexicalSuccessor && previous->lexicalSuccessor->block)
        fallThroughs.push_back(previous);
    }
    if (previous != exit)
      exits.push_back(previous);
  }

  /// Generate a SelectOp or branch sequence that compares \p selector against
  /// values in \p valueList and targets corresponding labels in \p labelList.
  /// If no value matches the selector, branch to \p defaultEval.
  ///
  /// Three cases require special processing.
  ///
  /// An empty \p valueList indicates an ArithmeticIfStmt context that requires
  /// two comparisons against 0 or 0.0. The selector may have either INTEGER
  /// or REAL type.
  ///
  /// A nonpositive \p valuelist value indicates an IO statement context
  /// (0 for ERR, -1 for END, -2 for EOR). An ERR branch must be taken for
  /// any positive (IOSTAT) value. A missing (zero) label requires a branch
  /// to \p defaultEval for that value.
  ///
  /// A non-null \p errorBlock indicates an AssignedGotoStmt context that
  /// must always branch to an explicit target. There is no valid defaultEval
  /// in this case. Generate a branch to \p errorBlock for an AssignedGotoStmt
  /// that violates this program requirement.
  ///
  /// If this is not an ArithmeticIfStmt and no targets have exit code,
  /// generate a SelectOp. Otherwise, for each target, if it has exit code,
  /// branch to a new block, insert exit code, and then branch to the target.
  /// Otherwise, branch directly to the target.
  void genMultiwayBranch(mlir::Value selector,
                         llvm::SmallVector<int64_t> valueList,
                         llvm::SmallVector<Fortran::parser::Label> labelList,
                         const Fortran::lower::pft::Evaluation &defaultEval,
                         mlir::Block *errorBlock = nullptr) {
    bool inArithmeticIfContext = valueList.empty();
    assert(((inArithmeticIfContext && labelList.size() == 2) ||
            (valueList.size() && labelList.size() == valueList.size())) &&
           "mismatched multiway branch targets");
    mlir::Block *defaultBlock = errorBlock ? errorBlock : defaultEval.block;
    bool defaultHasExitCode = !errorBlock && hasExitCode(defaultEval);
    bool hasAnyExitCode = defaultHasExitCode;
    if (!hasAnyExitCode)
      for (auto label : labelList)
        if (label && hasExitCode(evalOfLabel(label))) {
          hasAnyExitCode = true;
          break;
        }
    mlir::Location loc = toLocation();
    size_t branchCount = labelList.size();
    if (!inArithmeticIfContext && !hasAnyExitCode &&
        !getEval().forceAsUnstructured()) { // from -no-structured-fir option
      // Generate a SelectOp.
      llvm::SmallVector<mlir::Block *> blockList;
      for (auto label : labelList) {
        mlir::Block *block =
            label ? evalOfLabel(label).block : defaultEval.block;
        assert(block && "missing multiway branch block");
        blockList.push_back(block);
      }
      blockList.push_back(defaultBlock);
      if (valueList[branchCount - 1] == 0) // Swap IO ERR and default blocks.
        std::swap(blockList[branchCount - 1], blockList[branchCount]);
      builder->create<fir::SelectOp>(loc, selector, valueList, blockList);
      return;
    }
    mlir::Type selectorType = selector.getType();
    bool realSelector = mlir::isa<mlir::FloatType>(selectorType);
    assert((inArithmeticIfContext || !realSelector) && "invalid selector type");
    mlir::Value zero;
    if (inArithmeticIfContext)
      zero =
          realSelector
              ? builder->create<mlir::arith::ConstantOp>(
                    loc, selectorType, builder->getFloatAttr(selectorType, 0.0))
              : builder->createIntegerConstant(loc, selectorType, 0);
    for (auto label : llvm::enumerate(labelList)) {
      mlir::Value cond;
      if (realSelector) // inArithmeticIfContext
        cond = builder->create<mlir::arith::CmpFOp>(
            loc,
            label.index() == 0 ? mlir::arith::CmpFPredicate::OLT
                               : mlir::arith::CmpFPredicate::OGT,
            selector, zero);
      else if (inArithmeticIfContext) // INTEGER selector
        cond = builder->create<mlir::arith::CmpIOp>(
            loc,
            label.index() == 0 ? mlir::arith::CmpIPredicate::slt
                               : mlir::arith::CmpIPredicate::sgt,
            selector, zero);
      else // A value of 0 is an IO ERR branch: invert comparison.
        cond = builder->create<mlir::arith::CmpIOp>(
            loc,
            valueList[label.index()] == 0 ? mlir::arith::CmpIPredicate::ne
                                          : mlir::arith::CmpIPredicate::eq,
            selector,
            builder->createIntegerConstant(loc, selectorType,
                                           valueList[label.index()]));
      // Branch to a new block with exit code and then to the target, or branch
      // directly to the target. defaultBlock is the "else" target.
      bool lastBranch = label.index() == branchCount - 1;
      mlir::Block *nextBlock =
          lastBranch && !defaultHasExitCode
              ? defaultBlock
              : builder->getBlock()->splitBlock(builder->getInsertionPoint());
      const Fortran::lower::pft::Evaluation &targetEval =
          label.value() ? evalOfLabel(label.value()) : defaultEval;
      if (hasExitCode(targetEval)) {
        mlir::Block *jumpBlock =
            builder->getBlock()->splitBlock(builder->getInsertionPoint());
        genConditionalBranch(cond, jumpBlock, nextBlock);
        startBlock(jumpBlock);
        genConstructExitBranch(targetEval);
      } else {
        genConditionalBranch(cond, targetEval.block, nextBlock);
      }
      if (!lastBranch) {
        startBlock(nextBlock);
      } else if (defaultHasExitCode) {
        startBlock(nextBlock);
        genConstructExitBranch(defaultEval);
      }
    }
  }

  void pushActiveConstruct(Fortran::lower::pft::Evaluation &eval,
                           Fortran::lower::StatementContext &stmtCtx) {
    activeConstructStack.push_back(ConstructContext{eval, stmtCtx});
    eval.activeConstruct = true;
  }
  void popActiveConstruct() {
    assert(!activeConstructStack.empty() && "invalid active construct stack");
    activeConstructStack.back().eval.activeConstruct = false;
    if (activeConstructStack.back().pushedScope)
      localSymbols.popScope();
    activeConstructStack.pop_back();
  }

  //===--------------------------------------------------------------------===//
  // Termination of symbolically referenced execution units
  //===--------------------------------------------------------------------===//

  /// END of program
  ///
  /// Generate the cleanup block before the program exits
  void genExitRoutine() {

    if (blockIsUnterminated())
      builder->create<mlir::func::ReturnOp>(toLocation());
  }

  /// END of procedure-like constructs
  ///
  /// Generate the cleanup block before the procedure exits
  void genReturnSymbol(const Fortran::semantics::Symbol &functionSymbol) {
    const Fortran::semantics::Symbol &resultSym =
        functionSymbol.get<Fortran::semantics::SubprogramDetails>().result();
    Fortran::lower::SymbolBox resultSymBox = lookupSymbol(resultSym);
    mlir::Location loc = toLocation();
    if (!resultSymBox) {
      mlir::emitError(loc, "internal error when processing function return");
      return;
    }
    mlir::Value resultVal = resultSymBox.match(
        [&](const fir::CharBoxValue &x) -> mlir::Value {
          if (Fortran::semantics::IsBindCProcedure(functionSymbol))
            return builder->create<fir::LoadOp>(loc, x.getBuffer());
          return fir::factory::CharacterExprHelper{*builder, loc}
              .createEmboxChar(x.getBuffer(), x.getLen());
        },
        [&](const fir::MutableBoxValue &x) -> mlir::Value {
          mlir::Value resultRef = resultSymBox.getAddr();
          mlir::Value load = builder->create<fir::LoadOp>(loc, resultRef);
          unsigned rank = x.rank();
          if (x.isAllocatable() && rank > 0) {
            // ALLOCATABLE array result must have default lower bounds.
            // At the call site the result box of a function reference
            // might be considered having default lower bounds, but
            // the runtime box should probably comply with this assumption
            // as well. If the result box has proper lbounds in runtime,
            // this may improve the debugging experience of Fortran apps.
            // We may consider removing this, if the overhead of setting
            // default lower bounds is too big.
            mlir::Value one =
                builder->createIntegerConstant(loc, builder->getIndexType(), 1);
            llvm::SmallVector<mlir::Value> lbounds{rank, one};
            auto shiftTy = fir::ShiftType::get(builder->getContext(), rank);
            mlir::Value shiftOp =
                builder->create<fir::ShiftOp>(loc, shiftTy, lbounds);
            load = builder->create<fir::ReboxOp>(
                loc, load.getType(), load, shiftOp, /*slice=*/mlir::Value{});
          }
          return load;
        },
        [&](const auto &) -> mlir::Value {
          mlir::Value resultRef = resultSymBox.getAddr();
          mlir::Type resultType = genType(resultSym);
          mlir::Type resultRefType = builder->getRefType(resultType);
          // A function with multiple entry points returning different types
          // tags all result variables with one of the largest types to allow
          // them to share the same storage. Convert this to the actual type.
          if (resultRef.getType() != resultRefType)
            resultRef = builder->createConvert(loc, resultRefType, resultRef);
          return builder->create<fir::LoadOp>(loc, resultRef);
        });
    bridge.openAccCtx().finalizeAndPop();
    bridge.fctCtx().finalizeAndPop();
    builder->create<mlir::func::ReturnOp>(loc, resultVal);
  }

  /// Get the return value of a call to \p symbol, which is a subroutine entry
  /// point that has alternative return specifiers.
  const mlir::Value
  getAltReturnResult(const Fortran::semantics::Symbol &symbol) {
    assert(Fortran::semantics::HasAlternateReturns(symbol) &&
           "subroutine does not have alternate returns");
    return getSymbolAddress(symbol);
  }

  void genFIRProcedureExit(Fortran::lower::pft::FunctionLikeUnit &funit,
                           const Fortran::semantics::Symbol &symbol) {
    if (mlir::Block *finalBlock = funit.finalBlock) {
      // The current block must end with a terminator.
      if (blockIsUnterminated())
        builder->create<mlir::cf::BranchOp>(toLocation(), finalBlock);
      // Set insertion point to final block.
      builder->setInsertionPoint(finalBlock, finalBlock->end());
    }
    if (Fortran::semantics::IsFunction(symbol)) {
      genReturnSymbol(symbol);
    } else if (Fortran::semantics::HasAlternateReturns(symbol)) {
      mlir::Value retval = builder->create<fir::LoadOp>(
          toLocation(), getAltReturnResult(symbol));
      bridge.openAccCtx().finalizeAndPop();
      bridge.fctCtx().finalizeAndPop();
      builder->create<mlir::func::ReturnOp>(toLocation(), retval);
    } else {
      bridge.openAccCtx().finalizeAndPop();
      bridge.fctCtx().finalizeAndPop();
      genExitRoutine();
    }
  }

  //
  // Statements that have control-flow semantics
  //

  /// Generate an If[Then]Stmt condition or its negation.
  template <typename A>
  mlir::Value genIfCondition(const A *stmt, bool negate = false) {
    mlir::Location loc = toLocation();
    Fortran::lower::StatementContext stmtCtx;
    mlir::Value condExpr = createFIRExpr(
        loc,
        Fortran::semantics::GetExpr(
            std::get<Fortran::parser::ScalarLogicalExpr>(stmt->t)),
        stmtCtx);
    stmtCtx.finalizeAndReset();
    mlir::Value cond =
        builder->createConvert(loc, builder->getI1Type(), condExpr);
    if (negate)
      cond = builder->create<mlir::arith::XOrIOp>(
          loc, cond, builder->createIntegerConstant(loc, cond.getType(), 1));
    return cond;
  }

  mlir::func::FuncOp getFunc(llvm::StringRef name, mlir::FunctionType ty) {
    if (mlir::func::FuncOp func = builder->getNamedFunction(name)) {
      assert(func.getFunctionType() == ty);
      return func;
    }
    return builder->createFunction(toLocation(), name, ty);
  }

  /// Lowering of CALL statement
  void genFIR(const Fortran::parser::CallStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    Fortran::lower::pft::Evaluation &eval = getEval();
    setCurrentPosition(stmt.source);
    assert(stmt.typedCall && "Call was not analyzed");
    mlir::Value res{};
    if (lowerToHighLevelFIR()) {
      std::optional<mlir::Type> resultType;
      if (stmt.typedCall->hasAlternateReturns())
        resultType = builder->getIndexType();
      auto hlfirRes = Fortran::lower::convertCallToHLFIR(
          toLocation(), *this, *stmt.typedCall, resultType, localSymbols,
          stmtCtx);
      if (hlfirRes)
        res = *hlfirRes;
    } else {
      // Call statement lowering shares code with function call lowering.
      res = Fortran::lower::createSubroutineCall(
          *this, *stmt.typedCall, explicitIterSpace, implicitIterSpace,
          localSymbols, stmtCtx, /*isUserDefAssignment=*/false);
    }
    stmtCtx.finalizeAndReset();
    if (!res)
      return; // "Normal" subroutine call.
    // Call with alternate return specifiers.
    // The call returns an index that selects an alternate return branch target.
    llvm::SmallVector<int64_t> indexList;
    llvm::SmallVector<Fortran::parser::Label> labelList;
    int64_t index = 0;
    for (const Fortran::parser::ActualArgSpec &arg :
         std::get<std::list<Fortran::parser::ActualArgSpec>>(stmt.call.t)) {
      const auto &actual = std::get<Fortran::parser::ActualArg>(arg.t);
      if (const auto *altReturn =
              std::get_if<Fortran::parser::AltReturnSpec>(&actual.u)) {
        indexList.push_back(++index);
        labelList.push_back(altReturn->v);
      }
    }
    genMultiwayBranch(res, indexList, labelList, eval.nonNopSuccessor());
  }

  void genFIR(const Fortran::parser::ComputedGotoStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    Fortran::lower::pft::Evaluation &eval = getEval();
    mlir::Value selectExpr =
        createFIRExpr(toLocation(),
                      Fortran::semantics::GetExpr(
                          std::get<Fortran::parser::ScalarIntExpr>(stmt.t)),
                      stmtCtx);
    stmtCtx.finalizeAndReset();
    llvm::SmallVector<int64_t> indexList;
    llvm::SmallVector<Fortran::parser::Label> labelList;
    int64_t index = 0;
    for (Fortran::parser::Label label :
         std::get<std::list<Fortran::parser::Label>>(stmt.t)) {
      indexList.push_back(++index);
      labelList.push_back(label);
    }
    genMultiwayBranch(selectExpr, indexList, labelList, eval.nonNopSuccessor());
  }

  void genFIR(const Fortran::parser::ArithmeticIfStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    mlir::Value expr = createFIRExpr(
        toLocation(),
        Fortran::semantics::GetExpr(std::get<Fortran::parser::Expr>(stmt.t)),
        stmtCtx);
    stmtCtx.finalizeAndReset();
    // Raise an exception if REAL expr is a NaN.
    if (mlir::isa<mlir::FloatType>(expr.getType()))
      expr = builder->create<mlir::arith::AddFOp>(toLocation(), expr, expr);
    // An empty valueList indicates to genMultiwayBranch that the branch is
    // an ArithmeticIfStmt that has two branches on value 0 or 0.0.
    llvm::SmallVector<int64_t> valueList;
    llvm::SmallVector<Fortran::parser::Label> labelList;
    labelList.push_back(std::get<1>(stmt.t));
    labelList.push_back(std::get<3>(stmt.t));
    const Fortran::lower::pft::LabelEvalMap &labelEvaluationMap =
        getEval().getOwningProcedure()->labelEvaluationMap;
    const auto iter = labelEvaluationMap.find(std::get<2>(stmt.t));
    assert(iter != labelEvaluationMap.end() && "label missing from map");
    genMultiwayBranch(expr, valueList, labelList, *iter->second);
  }

  void genFIR(const Fortran::parser::AssignedGotoStmt &stmt) {
    // See Fortran 90 Clause 8.2.4.
    // Relax the requirement that the GOTO variable must have a value in the
    // label list when a list is present, and allow a branch to any non-format
    // target that has an ASSIGN statement for the variable.
    mlir::Location loc = toLocation();
    Fortran::lower::pft::Evaluation &eval = getEval();
    Fortran::lower::pft::FunctionLikeUnit &owningProc =
        *eval.getOwningProcedure();
    const Fortran::lower::pft::SymbolLabelMap &symbolLabelMap =
        owningProc.assignSymbolLabelMap;
    const Fortran::lower::pft::LabelEvalMap &labelEvalMap =
        owningProc.labelEvaluationMap;
    const Fortran::semantics::Symbol &symbol =
        *std::get<Fortran::parser::Name>(stmt.t).symbol;
    auto labelSetIter = symbolLabelMap.find(symbol);
    llvm::SmallVector<int64_t> valueList;
    llvm::SmallVector<Fortran::parser::Label> labelList;
    if (labelSetIter != symbolLabelMap.end()) {
      for (auto &label : labelSetIter->second) {
        const auto evalIter = labelEvalMap.find(label);
        assert(evalIter != labelEvalMap.end() && "assigned goto label missing");
        if (evalIter->second->block) { // non-format statement
          valueList.push_back(label);  // label as an integer
          labelList.push_back(label);
        }
      }
    }
    if (!labelList.empty()) {
      auto selectExpr =
          builder->create<fir::LoadOp>(loc, getSymbolAddress(symbol));
      // Add a default error target in case the goto is nonconforming.
      mlir::Block *errorBlock =
          builder->getBlock()->splitBlock(builder->getInsertionPoint());
      genMultiwayBranch(selectExpr, valueList, labelList,
                        eval.nonNopSuccessor(), errorBlock);
      startBlock(errorBlock);
    }
    fir::runtime::genReportFatalUserError(
        *builder, loc,
        "Assigned GOTO variable '" + symbol.name().ToString() +
            "' does not have a valid target label value");
    builder->create<fir::UnreachableOp>(loc);
  }

  fir::ReduceOperationEnum
  getReduceOperationEnum(const Fortran::parser::ReductionOperator &rOpr) {
    switch (rOpr.v) {
    case Fortran::parser::ReductionOperator::Operator::Plus:
      return fir::ReduceOperationEnum::Add;
    case Fortran::parser::ReductionOperator::Operator::Multiply:
      return fir::ReduceOperationEnum::Multiply;
    case Fortran::parser::ReductionOperator::Operator::And:
      return fir::ReduceOperationEnum::AND;
    case Fortran::parser::ReductionOperator::Operator::Or:
      return fir::ReduceOperationEnum::OR;
    case Fortran::parser::ReductionOperator::Operator::Eqv:
      return fir::ReduceOperationEnum::EQV;
    case Fortran::parser::ReductionOperator::Operator::Neqv:
      return fir::ReduceOperationEnum::NEQV;
    case Fortran::parser::ReductionOperator::Operator::Max:
      return fir::ReduceOperationEnum::MAX;
    case Fortran::parser::ReductionOperator::Operator::Min:
      return fir::ReduceOperationEnum::MIN;
    case Fortran::parser::ReductionOperator::Operator::Iand:
      return fir::ReduceOperationEnum::IAND;
    case Fortran::parser::ReductionOperator::Operator::Ior:
      return fir::ReduceOperationEnum::IOR;
    case Fortran::parser::ReductionOperator::Operator::Ieor:
      return fir::ReduceOperationEnum::EIOR;
    }
    llvm_unreachable("illegal reduction operator");
  }

  /// Collect DO CONCURRENT or FORALL loop control information.
  IncrementLoopNestInfo getConcurrentControl(
      const Fortran::parser::ConcurrentHeader &header,
      const std::list<Fortran::parser::LocalitySpec> &localityList = {}) {
    IncrementLoopNestInfo incrementLoopNestInfo;
    for (const Fortran::parser::ConcurrentControl &control :
         std::get<std::list<Fortran::parser::ConcurrentControl>>(header.t))
      incrementLoopNestInfo.emplace_back(
          *std::get<0>(control.t).symbol, std::get<1>(control.t),
          std::get<2>(control.t), std::get<3>(control.t), /*isUnordered=*/true);
    IncrementLoopInfo &info = incrementLoopNestInfo.back();
    info.maskExpr = Fortran::semantics::GetExpr(
        std::get<std::optional<Fortran::parser::ScalarLogicalExpr>>(header.t));
    for (const Fortran::parser::LocalitySpec &x : localityList) {
      if (const auto *localList =
              std::get_if<Fortran::parser::LocalitySpec::Local>(&x.u))
        for (const Fortran::parser::Name &x : localList->v)
          info.localSymList.push_back(x.symbol);
      if (const auto *localInitList =
              std::get_if<Fortran::parser::LocalitySpec::LocalInit>(&x.u))
        for (const Fortran::parser::Name &x : localInitList->v)
          info.localInitSymList.push_back(x.symbol);
      for (IncrementLoopInfo &info : incrementLoopNestInfo) {
        if (const auto *reduceList =
                std::get_if<Fortran::parser::LocalitySpec::Reduce>(&x.u)) {
          fir::ReduceOperationEnum reduce_operation = getReduceOperationEnum(
              std::get<Fortran::parser::ReductionOperator>(reduceList->t));
          for (const Fortran::parser::Name &x :
               std::get<std::list<Fortran::parser::Name>>(reduceList->t)) {
            info.reduceSymList.push_back(
                std::make_pair(reduce_operation, x.symbol));
          }
        }
      }
      if (const auto *sharedList =
              std::get_if<Fortran::parser::LocalitySpec::Shared>(&x.u))
        for (const Fortran::parser::Name &x : sharedList->v)
          info.sharedSymList.push_back(x.symbol);
    }
    return incrementLoopNestInfo;
  }

  /// Create DO CONCURRENT construct symbol bindings and generate LOCAL_INIT
  /// assignments.
  void handleLocalitySpecs(const IncrementLoopInfo &info) {
    Fortran::semantics::SemanticsContext &semanticsContext =
        bridge.getSemanticsContext();
    for (const Fortran::semantics::Symbol *sym : info.localSymList)
      createHostAssociateVarClone(*sym);
    for (const Fortran::semantics::Symbol *sym : info.localInitSymList) {
      createHostAssociateVarClone(*sym);
      const auto *hostDetails =
          sym->detailsIf<Fortran::semantics::HostAssocDetails>();
      assert(hostDetails && "missing locality spec host symbol");
      const Fortran::semantics::Symbol *hostSym = &hostDetails->symbol();
      Fortran::evaluate::ExpressionAnalyzer ea{semanticsContext};
      Fortran::evaluate::Assignment assign{
          ea.Designate(Fortran::evaluate::DataRef{*sym}).value(),
          ea.Designate(Fortran::evaluate::DataRef{*hostSym}).value()};
      if (Fortran::semantics::IsPointer(*sym))
        assign.u = Fortran::evaluate::Assignment::BoundsSpec{};
      genAssignment(assign);
    }
    for (const Fortran::semantics::Symbol *sym : info.sharedSymList) {
      const auto *hostDetails =
          sym->detailsIf<Fortran::semantics::HostAssocDetails>();
      copySymbolBinding(hostDetails->symbol(), *sym);
    }
  }

  /// Generate FIR for a DO construct. There are six variants:
  ///  - unstructured infinite and while loops
  ///  - structured and unstructured increment loops
  ///  - structured and unstructured concurrent loops
  void genFIR(const Fortran::parser::DoConstruct &doConstruct) {
    setCurrentPositionAt(doConstruct);
    // Collect loop nest information.
    // Generate begin loop code directly for infinite and while loops.
    Fortran::lower::pft::Evaluation &eval = getEval();
    bool unstructuredContext = eval.lowerAsUnstructured();
    Fortran::lower::pft::Evaluation &doStmtEval =
        eval.getFirstNestedEvaluation();
    auto *doStmt = doStmtEval.getIf<Fortran::parser::NonLabelDoStmt>();
    const auto &loopControl =
        std::get<std::optional<Fortran::parser::LoopControl>>(doStmt->t);
    mlir::Block *preheaderBlock = doStmtEval.block;
    mlir::Block *beginBlock =
        preheaderBlock ? preheaderBlock : builder->getBlock();
    auto createNextBeginBlock = [&]() {
      // Step beginBlock through unstructured preheader, header, and mask
      // blocks, created in outermost to innermost order.
      return beginBlock = beginBlock->splitBlock(beginBlock->end());
    };
    mlir::Block *headerBlock =
        unstructuredContext ? createNextBeginBlock() : nullptr;
    mlir::Block *bodyBlock = doStmtEval.lexicalSuccessor->block;
    mlir::Block *exitBlock = doStmtEval.parentConstruct->constructExit->block;
    IncrementLoopNestInfo incrementLoopNestInfo;
    const Fortran::parser::ScalarLogicalExpr *whileCondition = nullptr;
    bool infiniteLoop = !loopControl.has_value();
    if (infiniteLoop) {
      assert(unstructuredContext && "infinite loop must be unstructured");
      startBlock(headerBlock);
    } else if ((whileCondition =
                    std::get_if<Fortran::parser::ScalarLogicalExpr>(
                        &loopControl->u))) {
      assert(unstructuredContext && "while loop must be unstructured");
      maybeStartBlock(preheaderBlock); // no block or empty block
      startBlock(headerBlock);
      genConditionalBranch(*whileCondition, bodyBlock, exitBlock);
    } else if (const auto *bounds =
                   std::get_if<Fortran::parser::LoopControl::Bounds>(
                       &loopControl->u)) {
      // Non-concurrent increment loop.
      IncrementLoopInfo &info = incrementLoopNestInfo.emplace_back(
          *bounds->name.thing.symbol, bounds->lower, bounds->upper,
          bounds->step);
      if (unstructuredContext) {
        maybeStartBlock(preheaderBlock);
        info.hasRealControl = info.loopVariableSym->GetType()->IsNumeric(
            Fortran::common::TypeCategory::Real);
        info.headerBlock = headerBlock;
        info.bodyBlock = bodyBlock;
        info.exitBlock = exitBlock;
      }
    } else {
      const auto *concurrent =
          std::get_if<Fortran::parser::LoopControl::Concurrent>(
              &loopControl->u);
      assert(concurrent && "invalid DO loop variant");
      incrementLoopNestInfo = getConcurrentControl(
          std::get<Fortran::parser::ConcurrentHeader>(concurrent->t),
          std::get<std::list<Fortran::parser::LocalitySpec>>(concurrent->t));
      if (unstructuredContext) {
        maybeStartBlock(preheaderBlock);
        for (IncrementLoopInfo &info : incrementLoopNestInfo) {
          // The original loop body provides the body and latch blocks of the
          // innermost dimension. The (first) body block of a non-innermost
          // dimension is the preheader block of the immediately enclosed
          // dimension. The latch block of a non-innermost dimension is the
          // exit block of the immediately enclosed dimension.
          auto createNextExitBlock = [&]() {
            // Create unstructured loop exit blocks, outermost to innermost.
            return exitBlock = insertBlock(exitBlock);
          };
          bool isInnermost = &info == &incrementLoopNestInfo.back();
          bool isOutermost = &info == &incrementLoopNestInfo.front();
          info.headerBlock = isOutermost ? headerBlock : createNextBeginBlock();
          info.bodyBlock = isInnermost ? bodyBlock : createNextBeginBlock();
          info.exitBlock = isOutermost ? exitBlock : createNextExitBlock();
          if (info.maskExpr)
            info.maskBlock = createNextBeginBlock();
        }
      }
    }

    // Increment loop begin code. (Infinite/while code was already generated.)
    if (!infiniteLoop && !whileCondition)
      genFIRIncrementLoopBegin(incrementLoopNestInfo, doStmtEval.dirs);

    // Loop body code.
    auto iter = eval.getNestedEvaluations().begin();
    for (auto end = --eval.getNestedEvaluations().end(); iter != end; ++iter)
      genFIR(*iter, unstructuredContext);

    // An EndDoStmt in unstructured code may start a new block.
    Fortran::lower::pft::Evaluation &endDoEval = *iter;
    assert(endDoEval.getIf<Fortran::parser::EndDoStmt>() && "no enddo stmt");
    if (unstructuredContext)
      maybeStartBlock(endDoEval.block);

    // Loop end code.
    if (infiniteLoop || whileCondition)
      genBranch(headerBlock);
    else
      genFIRIncrementLoopEnd(incrementLoopNestInfo);

    // This call may generate a branch in some contexts.
    genFIR(endDoEval, unstructuredContext);
  }

  /// Generate FIR to evaluate loop control values (lower, upper and step).
  mlir::Value genControlValue(const Fortran::lower::SomeExpr *expr,
                              const IncrementLoopInfo &info,
                              bool *isConst = nullptr) {
    mlir::Location loc = toLocation();
    mlir::Type controlType = info.isStructured() ? builder->getIndexType()
                                                 : info.getLoopVariableType();
    Fortran::lower::StatementContext stmtCtx;
    if (expr) {
      if (isConst)
        *isConst = Fortran::evaluate::IsConstantExpr(*expr);
      return builder->createConvert(loc, controlType,
                                    createFIRExpr(loc, expr, stmtCtx));
    }

    if (isConst)
      *isConst = true;
    if (info.hasRealControl)
      return builder->createRealConstant(loc, controlType, 1u);
    return builder->createIntegerConstant(loc, controlType, 1); // step
  }

  void addLoopAnnotationAttr(IncrementLoopInfo &info) {
    mlir::BoolAttr f = mlir::BoolAttr::get(builder->getContext(), false);
    mlir::LLVM::LoopVectorizeAttr va = mlir::LLVM::LoopVectorizeAttr::get(
        builder->getContext(), /*disable=*/f, {}, {}, {}, {}, {}, {});
    mlir::LLVM::LoopAnnotationAttr la = mlir::LLVM::LoopAnnotationAttr::get(
        builder->getContext(), {}, /*vectorize=*/va, {}, {}, {}, {}, {}, {}, {},
        {}, {}, {}, {}, {}, {});
    info.doLoop.setLoopAnnotationAttr(la);
  }

  /// Generate FIR to begin a structured or unstructured increment loop nest.
  void genFIRIncrementLoopBegin(
      IncrementLoopNestInfo &incrementLoopNestInfo,
      llvm::SmallVectorImpl<const Fortran::parser::CompilerDirective *> &dirs) {
    assert(!incrementLoopNestInfo.empty() && "empty loop nest");
    mlir::Location loc = toLocation();
    for (IncrementLoopInfo &info : incrementLoopNestInfo) {
      info.loopVariable =
          genLoopVariableAddress(loc, *info.loopVariableSym, info.isUnordered);
      mlir::Value lowerValue = genControlValue(info.lowerExpr, info);
      mlir::Value upperValue = genControlValue(info.upperExpr, info);
      bool isConst = true;
      mlir::Value stepValue = genControlValue(
          info.stepExpr, info, info.isStructured() ? nullptr : &isConst);
      // Use a temp variable for unstructured loops with non-const step.
      if (!isConst) {
        info.stepVariable = builder->createTemporary(loc, stepValue.getType());
        builder->create<fir::StoreOp>(loc, stepValue, info.stepVariable);
      }

      // Structured loop - generate fir.do_loop.
      if (info.isStructured()) {
        mlir::Type loopVarType = info.getLoopVariableType();
        mlir::Value loopValue;
        if (info.isUnordered) {
          llvm::SmallVector<mlir::Value> reduceOperands;
          llvm::SmallVector<mlir::Attribute> reduceAttrs;
          // Create DO CONCURRENT reduce operands and attributes
          for (const auto &reduceSym : info.reduceSymList) {
            const fir::ReduceOperationEnum reduce_operation = reduceSym.first;
            const Fortran::semantics::Symbol *sym = reduceSym.second;
            fir::ExtendedValue exv = getSymbolExtendedValue(*sym, nullptr);
            reduceOperands.push_back(fir::getBase(exv));
            auto reduce_attr =
                fir::ReduceAttr::get(builder->getContext(), reduce_operation);
            reduceAttrs.push_back(reduce_attr);
          }
          // The loop variable value is explicitly updated.
          info.doLoop = builder->create<fir::DoLoopOp>(
              loc, lowerValue, upperValue, stepValue, /*unordered=*/true,
              /*finalCountValue=*/false, /*iterArgs=*/std::nullopt,
              llvm::ArrayRef<mlir::Value>(reduceOperands), reduceAttrs);
          builder->setInsertionPointToStart(info.doLoop.getBody());
          loopValue = builder->createConvert(loc, loopVarType,
                                             info.doLoop.getInductionVar());
        } else {
          // The loop variable is a doLoop op argument.
          info.doLoop = builder->create<fir::DoLoopOp>(
              loc, lowerValue, upperValue, stepValue, /*unordered=*/false,
              /*finalCountValue=*/true,
              builder->createConvert(loc, loopVarType, lowerValue));
          builder->setInsertionPointToStart(info.doLoop.getBody());
          loopValue = info.doLoop.getRegionIterArgs()[0];
        }
        // Update the loop variable value in case it has non-index references.
        builder->create<fir::StoreOp>(loc, loopValue, info.loopVariable);
        if (info.maskExpr) {
          Fortran::lower::StatementContext stmtCtx;
          mlir::Value maskCond = createFIRExpr(loc, info.maskExpr, stmtCtx);
          stmtCtx.finalizeAndReset();
          mlir::Value maskCondCast =
              builder->createConvert(loc, builder->getI1Type(), maskCond);
          auto ifOp = builder->create<fir::IfOp>(loc, maskCondCast,
                                                 /*withElseRegion=*/false);
          builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
        }
        if (info.hasLocalitySpecs())
          handleLocalitySpecs(info);

        for (const auto *dir : dirs) {
          Fortran::common::visit(
              Fortran::common::visitors{
                  [&](const Fortran::parser::CompilerDirective::VectorAlways
                          &d) { addLoopAnnotationAttr(info); },
                  [&](const auto &) {}},
              dir->u);
        }
        continue;
      }

      // Unstructured loop preheader - initialize tripVariable and loopVariable.
      mlir::Value tripCount;
      if (info.hasRealControl) {
        auto diff1 =
            builder->create<mlir::arith::SubFOp>(loc, upperValue, lowerValue);
        auto diff2 =
            builder->create<mlir::arith::AddFOp>(loc, diff1, stepValue);
        tripCount = builder->create<mlir::arith::DivFOp>(loc, diff2, stepValue);
        tripCount =
            builder->createConvert(loc, builder->getIndexType(), tripCount);
      } else {
        auto diff1 =
            builder->create<mlir::arith::SubIOp>(loc, upperValue, lowerValue);
        auto diff2 =
            builder->create<mlir::arith::AddIOp>(loc, diff1, stepValue);
        tripCount =
            builder->create<mlir::arith::DivSIOp>(loc, diff2, stepValue);
      }
      if (forceLoopToExecuteOnce) { // minimum tripCount is 1
        mlir::Value one =
            builder->createIntegerConstant(loc, tripCount.getType(), 1);
        auto cond = builder->create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, tripCount, one);
        tripCount =
            builder->create<mlir::arith::SelectOp>(loc, cond, one, tripCount);
      }
      info.tripVariable = builder->createTemporary(loc, tripCount.getType());
      builder->create<fir::StoreOp>(loc, tripCount, info.tripVariable);
      builder->create<fir::StoreOp>(loc, lowerValue, info.loopVariable);

      // Unstructured loop header - generate loop condition and mask.
      // Note - Currently there is no way to tag a loop as a concurrent loop.
      startBlock(info.headerBlock);
      tripCount = builder->create<fir::LoadOp>(loc, info.tripVariable);
      mlir::Value zero =
          builder->createIntegerConstant(loc, tripCount.getType(), 0);
      auto cond = builder->create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sgt, tripCount, zero);
      if (info.maskExpr) {
        genConditionalBranch(cond, info.maskBlock, info.exitBlock);
        startBlock(info.maskBlock);
        mlir::Block *latchBlock = getEval().getLastNestedEvaluation().block;
        assert(latchBlock && "missing masked concurrent loop latch block");
        Fortran::lower::StatementContext stmtCtx;
        mlir::Value maskCond = createFIRExpr(loc, info.maskExpr, stmtCtx);
        stmtCtx.finalizeAndReset();
        genConditionalBranch(maskCond, info.bodyBlock, latchBlock);
      } else {
        genConditionalBranch(cond, info.bodyBlock, info.exitBlock);
        if (&info != &incrementLoopNestInfo.back()) // not innermost
          startBlock(info.bodyBlock); // preheader block of enclosed dimension
      }
      if (info.hasLocalitySpecs()) {
        mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
        builder->setInsertionPointToStart(info.bodyBlock);
        handleLocalitySpecs(info);
        builder->restoreInsertionPoint(insertPt);
      }
    }
  }

  /// Generate FIR to end a structured or unstructured increment loop nest.
  void genFIRIncrementLoopEnd(IncrementLoopNestInfo &incrementLoopNestInfo) {
    assert(!incrementLoopNestInfo.empty() && "empty loop nest");
    mlir::Location loc = toLocation();
    mlir::arith::IntegerOverflowFlags flags{};
    if (getLoweringOptions().getNSWOnLoopVarInc())
      flags = bitEnumSet(flags, mlir::arith::IntegerOverflowFlags::nsw);
    auto iofAttr = mlir::arith::IntegerOverflowFlagsAttr::get(
        builder->getContext(), flags);
    for (auto it = incrementLoopNestInfo.rbegin(),
              rend = incrementLoopNestInfo.rend();
         it != rend; ++it) {
      IncrementLoopInfo &info = *it;
      if (info.isStructured()) {
        // End fir.do_loop.
        if (info.isUnordered) {
          builder->setInsertionPointAfter(info.doLoop);
          continue;
        }
        // Decrement tripVariable.
        builder->setInsertionPointToEnd(info.doLoop.getBody());
        llvm::SmallVector<mlir::Value, 2> results;
        results.push_back(builder->create<mlir::arith::AddIOp>(
            loc, info.doLoop.getInductionVar(), info.doLoop.getStep(),
            iofAttr));
        // Step loopVariable to help optimizations such as vectorization.
        // Induction variable elimination will clean up as necessary.
        mlir::Value step = builder->createConvert(
            loc, info.getLoopVariableType(), info.doLoop.getStep());
        mlir::Value loopVar =
            builder->create<fir::LoadOp>(loc, info.loopVariable);
        results.push_back(
            builder->create<mlir::arith::AddIOp>(loc, loopVar, step, iofAttr));
        builder->create<fir::ResultOp>(loc, results);
        builder->setInsertionPointAfter(info.doLoop);
        // The loop control variable may be used after the loop.
        builder->create<fir::StoreOp>(loc, info.doLoop.getResult(1),
                                      info.loopVariable);
        continue;
      }

      // Unstructured loop - decrement tripVariable and step loopVariable.
      mlir::Value tripCount =
          builder->create<fir::LoadOp>(loc, info.tripVariable);
      mlir::Value one =
          builder->createIntegerConstant(loc, tripCount.getType(), 1);
      tripCount = builder->create<mlir::arith::SubIOp>(loc, tripCount, one);
      builder->create<fir::StoreOp>(loc, tripCount, info.tripVariable);
      mlir::Value value = builder->create<fir::LoadOp>(loc, info.loopVariable);
      mlir::Value step;
      if (info.stepVariable)
        step = builder->create<fir::LoadOp>(loc, info.stepVariable);
      else
        step = genControlValue(info.stepExpr, info);
      if (info.hasRealControl)
        value = builder->create<mlir::arith::AddFOp>(loc, value, step);
      else
        value = builder->create<mlir::arith::AddIOp>(loc, value, step, iofAttr);
      builder->create<fir::StoreOp>(loc, value, info.loopVariable);

      genBranch(info.headerBlock);
      if (&info != &incrementLoopNestInfo.front()) // not outermost
        startBlock(info.exitBlock); // latch block of enclosing dimension
    }
  }

  /// Generate structured or unstructured FIR for an IF construct.
  /// The initial statement may be either an IfStmt or an IfThenStmt.
  void genFIR(const Fortran::parser::IfConstruct &) {
    Fortran::lower::pft::Evaluation &eval = getEval();

    // Structured fir.if nest.
    if (eval.lowerAsStructured()) {
      fir::IfOp topIfOp, currentIfOp;
      for (Fortran::lower::pft::Evaluation &e : eval.getNestedEvaluations()) {
        auto genIfOp = [&](mlir::Value cond) {
          Fortran::lower::pft::Evaluation &succ = *e.controlSuccessor;
          bool hasElse = succ.isA<Fortran::parser::ElseIfStmt>() ||
                         succ.isA<Fortran::parser::ElseStmt>();
          auto ifOp = builder->create<fir::IfOp>(toLocation(), cond,
                                                 /*withElseRegion=*/hasElse);
          builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
          return ifOp;
        };
        setCurrentPosition(e.position);
        if (auto *s = e.getIf<Fortran::parser::IfThenStmt>()) {
          topIfOp = currentIfOp = genIfOp(genIfCondition(s, e.negateCondition));
        } else if (auto *s = e.getIf<Fortran::parser::IfStmt>()) {
          topIfOp = currentIfOp = genIfOp(genIfCondition(s, e.negateCondition));
        } else if (auto *s = e.getIf<Fortran::parser::ElseIfStmt>()) {
          builder->setInsertionPointToStart(
              &currentIfOp.getElseRegion().front());
          currentIfOp = genIfOp(genIfCondition(s));
        } else if (e.isA<Fortran::parser::ElseStmt>()) {
          builder->setInsertionPointToStart(
              &currentIfOp.getElseRegion().front());
        } else if (e.isA<Fortran::parser::EndIfStmt>()) {
          builder->setInsertionPointAfter(topIfOp);
          genFIR(e, /*unstructuredContext=*/false); // may generate branch
        } else {
          genFIR(e, /*unstructuredContext=*/false);
        }
      }
      return;
    }

    // Unstructured branch sequence.
    llvm::SmallVector<Fortran::lower::pft::Evaluation *> exits, fallThroughs;
    collectFinalEvaluations(eval, exits, fallThroughs);

    for (Fortran::lower::pft::Evaluation &e : eval.getNestedEvaluations()) {
      auto genIfBranch = [&](mlir::Value cond) {
        if (e.lexicalSuccessor == e.controlSuccessor) // empty block -> exit
          genConditionalBranch(cond, e.parentConstruct->constructExit,
                               e.controlSuccessor);
        else // non-empty block
          genConditionalBranch(cond, e.lexicalSuccessor, e.controlSuccessor);
      };
      setCurrentPosition(e.position);
      if (auto *s = e.getIf<Fortran::parser::IfThenStmt>()) {
        maybeStartBlock(e.block);
        genIfBranch(genIfCondition(s, e.negateCondition));
      } else if (auto *s = e.getIf<Fortran::parser::IfStmt>()) {
        maybeStartBlock(e.block);
        genIfBranch(genIfCondition(s, e.negateCondition));
      } else if (auto *s = e.getIf<Fortran::parser::ElseIfStmt>()) {
        startBlock(e.block);
        genIfBranch(genIfCondition(s));
      } else {
        genFIR(e);
        if (blockIsUnterminated()) {
          if (llvm::is_contained(exits, &e))
            genConstructExitBranch(*eval.constructExit);
          else if (llvm::is_contained(fallThroughs, &e))
            genBranch(e.lexicalSuccessor->block);
        }
      }
    }
  }

  void genCaseOrRankConstruct() {
    Fortran::lower::pft::Evaluation &eval = getEval();
    Fortran::lower::StatementContext stmtCtx;
    pushActiveConstruct(eval, stmtCtx);

    llvm::SmallVector<Fortran::lower::pft::Evaluation *> exits, fallThroughs;
    collectFinalEvaluations(eval, exits, fallThroughs);

    for (Fortran::lower::pft::Evaluation &e : eval.getNestedEvaluations()) {
      if (e.getIf<Fortran::parser::EndSelectStmt>())
        maybeStartBlock(e.block);
      else
        genFIR(e);
      if (blockIsUnterminated()) {
        if (llvm::is_contained(exits, &e))
          genConstructExitBranch(*eval.constructExit);
        else if (llvm::is_contained(fallThroughs, &e))
          genBranch(e.lexicalSuccessor->block);
      }
    }
    popActiveConstruct();
  }
  void genFIR(const Fortran::parser::CaseConstruct &) {
    genCaseOrRankConstruct();
  }

  template <typename A>
  void genNestedStatement(const Fortran::parser::Statement<A> &stmt) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }

  /// Force the binding of an explicit symbol. This is used to bind and re-bind
  /// a concurrent control symbol to its value.
  void forceControlVariableBinding(const Fortran::semantics::Symbol *sym,
                                   mlir::Value inducVar) {
    mlir::Location loc = toLocation();
    assert(sym && "There must be a symbol to bind");
    mlir::Type toTy = genType(*sym);
    // FIXME: this should be a "per iteration" temporary.
    mlir::Value tmp =
        builder->createTemporary(loc, toTy, toStringRef(sym->name()),
                                 llvm::ArrayRef<mlir::NamedAttribute>{
                                     fir::getAdaptToByRefAttr(*builder)});
    mlir::Value cast = builder->createConvert(loc, toTy, inducVar);
    builder->create<fir::StoreOp>(loc, cast, tmp);
    addSymbol(*sym, tmp, /*force=*/true);
  }

  /// Process a concurrent header for a FORALL. (Concurrent headers for DO
  /// CONCURRENT loops are lowered elsewhere.)
  void genFIR(const Fortran::parser::ConcurrentHeader &header) {
    llvm::SmallVector<mlir::Value> lows;
    llvm::SmallVector<mlir::Value> highs;
    llvm::SmallVector<mlir::Value> steps;
    if (explicitIterSpace.isOutermostForall()) {
      // For the outermost forall, we evaluate the bounds expressions once.
      // Contrastingly, if this forall is nested, the bounds expressions are
      // assumed to be pure, possibly dependent on outer concurrent control
      // variables, possibly variant with respect to arguments, and will be
      // re-evaluated.
      mlir::Location loc = toLocation();
      mlir::Type idxTy = builder->getIndexType();
      Fortran::lower::StatementContext &stmtCtx =
          explicitIterSpace.stmtContext();
      auto lowerExpr = [&](auto &e) {
        return fir::getBase(genExprValue(e, stmtCtx));
      };
      for (const Fortran::parser::ConcurrentControl &ctrl :
           std::get<std::list<Fortran::parser::ConcurrentControl>>(header.t)) {
        const Fortran::lower::SomeExpr *lo =
            Fortran::semantics::GetExpr(std::get<1>(ctrl.t));
        const Fortran::lower::SomeExpr *hi =
            Fortran::semantics::GetExpr(std::get<2>(ctrl.t));
        auto &optStep =
            std::get<std::optional<Fortran::parser::ScalarIntExpr>>(ctrl.t);
        lows.push_back(builder->createConvert(loc, idxTy, lowerExpr(*lo)));
        highs.push_back(builder->createConvert(loc, idxTy, lowerExpr(*hi)));
        steps.push_back(
            optStep.has_value()
                ? builder->createConvert(
                      loc, idxTy,
                      lowerExpr(*Fortran::semantics::GetExpr(*optStep)))
                : builder->createIntegerConstant(loc, idxTy, 1));
      }
    }
    auto lambda = [&, lows, highs, steps]() {
      // Create our iteration space from the header spec.
      mlir::Location loc = toLocation();
      mlir::Type idxTy = builder->getIndexType();
      llvm::SmallVector<fir::DoLoopOp> loops;
      Fortran::lower::StatementContext &stmtCtx =
          explicitIterSpace.stmtContext();
      auto lowerExpr = [&](auto &e) {
        return fir::getBase(genExprValue(e, stmtCtx));
      };
      const bool outermost = !lows.empty();
      std::size_t headerIndex = 0;
      for (const Fortran::parser::ConcurrentControl &ctrl :
           std::get<std::list<Fortran::parser::ConcurrentControl>>(header.t)) {
        const Fortran::semantics::Symbol *ctrlVar =
            std::get<Fortran::parser::Name>(ctrl.t).symbol;
        mlir::Value lb;
        mlir::Value ub;
        mlir::Value by;
        if (outermost) {
          assert(headerIndex < lows.size());
          if (headerIndex == 0)
            explicitIterSpace.resetInnerArgs();
          lb = lows[headerIndex];
          ub = highs[headerIndex];
          by = steps[headerIndex++];
        } else {
          const Fortran::lower::SomeExpr *lo =
              Fortran::semantics::GetExpr(std::get<1>(ctrl.t));
          const Fortran::lower::SomeExpr *hi =
              Fortran::semantics::GetExpr(std::get<2>(ctrl.t));
          auto &optStep =
              std::get<std::optional<Fortran::parser::ScalarIntExpr>>(ctrl.t);
          lb = builder->createConvert(loc, idxTy, lowerExpr(*lo));
          ub = builder->createConvert(loc, idxTy, lowerExpr(*hi));
          by = optStep.has_value()
                   ? builder->createConvert(
                         loc, idxTy,
                         lowerExpr(*Fortran::semantics::GetExpr(*optStep)))
                   : builder->createIntegerConstant(loc, idxTy, 1);
        }
        auto lp = builder->create<fir::DoLoopOp>(
            loc, lb, ub, by, /*unordered=*/true,
            /*finalCount=*/false, explicitIterSpace.getInnerArgs());
        if ((!loops.empty() || !outermost) && !lp.getRegionIterArgs().empty())
          builder->create<fir::ResultOp>(loc, lp.getResults());
        explicitIterSpace.setInnerArgs(lp.getRegionIterArgs());
        builder->setInsertionPointToStart(lp.getBody());
        forceControlVariableBinding(ctrlVar, lp.getInductionVar());
        loops.push_back(lp);
      }
      if (outermost)
        explicitIterSpace.setOuterLoop(loops[0]);
      explicitIterSpace.appendLoops(loops);
      if (const auto &mask =
              std::get<std::optional<Fortran::parser::ScalarLogicalExpr>>(
                  header.t);
          mask.has_value()) {
        mlir::Type i1Ty = builder->getI1Type();
        fir::ExtendedValue maskExv =
            genExprValue(*Fortran::semantics::GetExpr(mask.value()), stmtCtx);
        mlir::Value cond =
            builder->createConvert(loc, i1Ty, fir::getBase(maskExv));
        auto ifOp = builder->create<fir::IfOp>(
            loc, explicitIterSpace.innerArgTypes(), cond,
            /*withElseRegion=*/true);
        builder->create<fir::ResultOp>(loc, ifOp.getResults());
        builder->setInsertionPointToStart(&ifOp.getElseRegion().front());
        builder->create<fir::ResultOp>(loc, explicitIterSpace.getInnerArgs());
        builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
      }
    };
    // Push the lambda to gen the loop nest context.
    explicitIterSpace.pushLoopNest(lambda);
  }

  void genFIR(const Fortran::parser::ForallAssignmentStmt &stmt) {
    Fortran::common::visit([&](const auto &x) { genFIR(x); }, stmt.u);
  }

  void genFIR(const Fortran::parser::EndForallStmt &) {
    if (!lowerToHighLevelFIR())
      cleanupExplicitSpace();
  }

  template <typename A>
  void prepareExplicitSpace(const A &forall) {
    if (!explicitIterSpace.isActive())
      analyzeExplicitSpace(forall);
    localSymbols.pushScope();
    explicitIterSpace.enter();
  }

  /// Cleanup all the FORALL context information when we exit.
  void cleanupExplicitSpace() {
    explicitIterSpace.leave();
    localSymbols.popScope();
  }

  /// Generate FIR for a FORALL statement.
  void genFIR(const Fortran::parser::ForallStmt &stmt) {
    const auto &concurrentHeader =
        std::get<
            Fortran::common::Indirection<Fortran::parser::ConcurrentHeader>>(
            stmt.t)
            .value();
    if (lowerToHighLevelFIR()) {
      mlir::OpBuilder::InsertionGuard guard(*builder);
      Fortran::lower::SymMapScope scope(localSymbols);
      genForallNest(concurrentHeader);
      genFIR(std::get<Fortran::parser::UnlabeledStatement<
                 Fortran::parser::ForallAssignmentStmt>>(stmt.t)
                 .statement);
      return;
    }
    prepareExplicitSpace(stmt);
    genFIR(concurrentHeader);
    genFIR(std::get<Fortran::parser::UnlabeledStatement<
               Fortran::parser::ForallAssignmentStmt>>(stmt.t)
               .statement);
    cleanupExplicitSpace();
  }

  /// Generate FIR for a FORALL construct.
  void genFIR(const Fortran::parser::ForallConstruct &forall) {
    mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
    if (lowerToHighLevelFIR())
      localSymbols.pushScope();
    else
      prepareExplicitSpace(forall);
    genNestedStatement(
        std::get<
            Fortran::parser::Statement<Fortran::parser::ForallConstructStmt>>(
            forall.t));
    for (const Fortran::parser::ForallBodyConstruct &s :
         std::get<std::list<Fortran::parser::ForallBodyConstruct>>(forall.t)) {
      Fortran::common::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::WhereConstruct &b) { genFIR(b); },
              [&](const Fortran::common::Indirection<
                  Fortran::parser::ForallConstruct> &b) { genFIR(b.value()); },
              [&](const auto &b) { genNestedStatement(b); }},
          s.u);
    }
    genNestedStatement(
        std::get<Fortran::parser::Statement<Fortran::parser::EndForallStmt>>(
            forall.t));
    if (lowerToHighLevelFIR()) {
      localSymbols.popScope();
      builder->restoreInsertionPoint(insertPt);
    }
  }

  /// Lower the concurrent header specification.
  void genFIR(const Fortran::parser::ForallConstructStmt &stmt) {
    const auto &concurrentHeader =
        std::get<
            Fortran::common::Indirection<Fortran::parser::ConcurrentHeader>>(
            stmt.t)
            .value();
    if (lowerToHighLevelFIR())
      genForallNest(concurrentHeader);
    else
      genFIR(concurrentHeader);
  }

  /// Generate hlfir.forall and hlfir.forall_mask nest given a Forall
  /// concurrent header
  void genForallNest(const Fortran::parser::ConcurrentHeader &header) {
    mlir::Location loc = getCurrentLocation();
    const bool isOutterForall = !isInsideHlfirForallOrWhere();
    hlfir::ForallOp outerForall;
    auto evaluateControl = [&](const auto &parserExpr, mlir::Region &region,
                               bool isMask = false) {
      if (region.empty())
        builder->createBlock(&region);
      Fortran::lower::StatementContext localStmtCtx;
      const Fortran::semantics::SomeExpr *anlalyzedExpr =
          Fortran::semantics::GetExpr(parserExpr);
      assert(anlalyzedExpr && "expression semantics failed");
      // Generate the controls of outer forall outside of the hlfir.forall
      // region. They do not depend on any previous forall indices (C1123) and
      // no assignment has been made yet that could modify their value. This
      // will simplify hlfir.forall analysis because the SSA integer value
      // yielded will obviously not depend on any variable modified by the
      // forall when produced outside of it.
      // This is not done for the mask because it may (and in usual code, does)
      // depend on the forall indices that have just been defined as
      // hlfir.forall block arguments.
      mlir::OpBuilder::InsertPoint innerInsertionPoint;
      if (outerForall && !isMask) {
        innerInsertionPoint = builder->saveInsertionPoint();
        builder->setInsertionPoint(outerForall);
      }
      mlir::Value exprVal =
          fir::getBase(genExprValue(*anlalyzedExpr, localStmtCtx, &loc));
      localStmtCtx.finalizeAndPop();
      if (isMask)
        exprVal = builder->createConvert(loc, builder->getI1Type(), exprVal);
      if (innerInsertionPoint.isSet())
        builder->restoreInsertionPoint(innerInsertionPoint);
      builder->create<hlfir::YieldOp>(loc, exprVal);
    };
    for (const Fortran::parser::ConcurrentControl &control :
         std::get<std::list<Fortran::parser::ConcurrentControl>>(header.t)) {
      auto forallOp = builder->create<hlfir::ForallOp>(loc);
      if (isOutterForall && !outerForall)
        outerForall = forallOp;
      evaluateControl(std::get<1>(control.t), forallOp.getLbRegion());
      evaluateControl(std::get<2>(control.t), forallOp.getUbRegion());
      if (const auto &optionalStep =
              std::get<std::optional<Fortran::parser::ScalarIntExpr>>(
                  control.t))
        evaluateControl(*optionalStep, forallOp.getStepRegion());
      // Create block argument and map it to a symbol via an hlfir.forall_index
      // op (symbols must be mapped to in memory values).
      const Fortran::semantics::Symbol *controlVar =
          std::get<Fortran::parser::Name>(control.t).symbol;
      assert(controlVar && "symbol analysis failed");
      mlir::Type controlVarType = genType(*controlVar);
      mlir::Block *forallBody = builder->createBlock(&forallOp.getBody(), {},
                                                     {controlVarType}, {loc});
      auto forallIndex = builder->create<hlfir::ForallIndexOp>(
          loc, fir::ReferenceType::get(controlVarType),
          forallBody->getArguments()[0],
          builder->getStringAttr(controlVar->name().ToString()));
      localSymbols.addVariableDefinition(*controlVar, forallIndex,
                                         /*force=*/true);
      auto end = builder->create<fir::FirEndOp>(loc);
      builder->setInsertionPoint(end);
    }

    if (const auto &maskExpr =
            std::get<std::optional<Fortran::parser::ScalarLogicalExpr>>(
                header.t)) {
      // Create hlfir.forall_mask and set insertion point in its body.
      auto forallMaskOp = builder->create<hlfir::ForallMaskOp>(loc);
      evaluateControl(*maskExpr, forallMaskOp.getMaskRegion(), /*isMask=*/true);
      builder->createBlock(&forallMaskOp.getBody());
      auto end = builder->create<fir::FirEndOp>(loc);
      builder->setInsertionPoint(end);
    }
  }

  void attachDirectiveToLoop(const Fortran::parser::CompilerDirective &dir,
                             Fortran::lower::pft::Evaluation *e) {
    while (e->isDirective())
      e = e->lexicalSuccessor;

    if (e->isA<Fortran::parser::NonLabelDoStmt>())
      e->dirs.push_back(&dir);
  }

  void genFIR(const Fortran::parser::CompilerDirective &dir) {
    Fortran::lower::pft::Evaluation &eval = getEval();

    Fortran::common::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::CompilerDirective::VectorAlways &) {
              attachDirectiveToLoop(dir, &eval);
            },
            [&](const auto &) {}},
        dir.u);
  }

  void genFIR(const Fortran::parser::OpenACCConstruct &acc) {
    mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
    localSymbols.pushScope();
    mlir::Value exitCond = genOpenACCConstruct(
        *this, bridge.getSemanticsContext(), getEval(), acc);

    const Fortran::parser::OpenACCLoopConstruct *accLoop =
        std::get_if<Fortran::parser::OpenACCLoopConstruct>(&acc.u);
    const Fortran::parser::OpenACCCombinedConstruct *accCombined =
        std::get_if<Fortran::parser::OpenACCCombinedConstruct>(&acc.u);

    Fortran::lower::pft::Evaluation *curEval = &getEval();

    if (accLoop || accCombined) {
      int64_t collapseValue;
      if (accLoop) {
        const Fortran::parser::AccBeginLoopDirective &beginLoopDir =
            std::get<Fortran::parser::AccBeginLoopDirective>(accLoop->t);
        const Fortran::parser::AccClauseList &clauseList =
            std::get<Fortran::parser::AccClauseList>(beginLoopDir.t);
        collapseValue = Fortran::lower::getCollapseValue(clauseList);
      } else if (accCombined) {
        const Fortran::parser::AccBeginCombinedDirective &beginCombinedDir =
            std::get<Fortran::parser::AccBeginCombinedDirective>(
                accCombined->t);
        const Fortran::parser::AccClauseList &clauseList =
            std::get<Fortran::parser::AccClauseList>(beginCombinedDir.t);
        collapseValue = Fortran::lower::getCollapseValue(clauseList);
      }

      if (curEval->lowerAsStructured()) {
        curEval = &curEval->getFirstNestedEvaluation();
        for (int64_t i = 1; i < collapseValue; i++)
          curEval = &*std::next(curEval->getNestedEvaluations().begin());
      }
    }

    for (Fortran::lower::pft::Evaluation &e : curEval->getNestedEvaluations())
      genFIR(e);
    localSymbols.popScope();
    builder->restoreInsertionPoint(insertPt);

    if (accLoop && exitCond) {
      Fortran::lower::pft::FunctionLikeUnit *funit =
          getEval().getOwningProcedure();
      assert(funit && "not inside main program, function or subroutine");
      mlir::Block *continueBlock =
          builder->getBlock()->splitBlock(builder->getBlock()->end());
      builder->create<mlir::cf::CondBranchOp>(toLocation(), exitCond,
                                              funit->finalBlock, continueBlock);
      builder->setInsertionPointToEnd(continueBlock);
    }
  }

  void genFIR(const Fortran::parser::OpenACCDeclarativeConstruct &accDecl) {
    genOpenACCDeclarativeConstruct(*this, bridge.getSemanticsContext(),
                                   bridge.openAccCtx(), accDecl,
                                   accRoutineInfos);
    for (Fortran::lower::pft::Evaluation &e : getEval().getNestedEvaluations())
      genFIR(e);
  }

  void genFIR(const Fortran::parser::OpenACCRoutineConstruct &acc) {
    // Handled by genFIR(const Fortran::parser::OpenACCDeclarativeConstruct &)
  }

  void genFIR(const Fortran::parser::CUFKernelDoConstruct &kernel) {
    Fortran::lower::SymMapScope scope(localSymbols);
    const Fortran::parser::CUFKernelDoConstruct::Directive &dir =
        std::get<Fortran::parser::CUFKernelDoConstruct::Directive>(kernel.t);

    mlir::Location loc = genLocation(dir.source);

    Fortran::lower::StatementContext stmtCtx;

    unsigned nestedLoops = 1;

    const auto &nLoops =
        std::get<std::optional<Fortran::parser::ScalarIntConstantExpr>>(dir.t);
    if (nLoops)
      nestedLoops = *Fortran::semantics::GetIntValue(*nLoops);

    mlir::IntegerAttr n;
    if (nestedLoops > 1)
      n = builder->getIntegerAttr(builder->getI64Type(), nestedLoops);

    const std::list<Fortran::parser::CUFKernelDoConstruct::StarOrExpr> &grid =
        std::get<1>(dir.t);
    const std::list<Fortran::parser::CUFKernelDoConstruct::StarOrExpr> &block =
        std::get<2>(dir.t);
    const std::optional<Fortran::parser::ScalarIntExpr> &stream =
        std::get<3>(dir.t);
    const std::list<Fortran::parser::CUFReduction> &cufreds =
        std::get<4>(dir.t);

    llvm::SmallVector<mlir::Value> reduceOperands;
    llvm::SmallVector<mlir::Attribute> reduceAttrs;

    for (const Fortran::parser::CUFReduction &cufred : cufreds) {
      fir::ReduceOperationEnum redOpEnum = getReduceOperationEnum(
          std::get<Fortran::parser::ReductionOperator>(cufred.t));
      const std::list<Fortran::parser::Scalar<Fortran::parser::Variable>>
          &scalarvars = std::get<1>(cufred.t);
      for (const Fortran::parser::Scalar<Fortran::parser::Variable> &scalarvar :
           scalarvars) {
        auto reduce_attr =
            fir::ReduceAttr::get(builder->getContext(), redOpEnum);
        reduceAttrs.push_back(reduce_attr);
        const Fortran::parser::Variable &var = scalarvar.thing;
        if (const auto *iDesignator = std::get_if<
                Fortran::common::Indirection<Fortran::parser::Designator>>(
                &var.u)) {
          const Fortran::parser::Designator &designator = iDesignator->value();
          if (const auto *name =
                  Fortran::semantics::getDesignatorNameIfDataRef(designator)) {
            auto val = getSymbolAddress(*name->symbol);
            reduceOperands.push_back(val);
          }
        }
      }
    }

    auto isOnlyStars =
        [&](const std::list<Fortran::parser::CUFKernelDoConstruct::StarOrExpr>
                &list) -> bool {
      for (const Fortran::parser::CUFKernelDoConstruct::StarOrExpr &expr :
           list) {
        if (expr.v)
          return false;
      }
      return true;
    };

    mlir::Value zero =
        builder->createIntegerConstant(loc, builder->getI32Type(), 0);

    llvm::SmallVector<mlir::Value> gridValues;
    if (!isOnlyStars(grid)) {
      for (const Fortran::parser::CUFKernelDoConstruct::StarOrExpr &expr :
           grid) {
        if (expr.v) {
          gridValues.push_back(fir::getBase(
              genExprValue(*Fortran::semantics::GetExpr(*expr.v), stmtCtx)));
        } else {
          gridValues.push_back(zero);
        }
      }
    }
    llvm::SmallVector<mlir::Value> blockValues;
    if (!isOnlyStars(block)) {
      for (const Fortran::parser::CUFKernelDoConstruct::StarOrExpr &expr :
           block) {
        if (expr.v) {
          blockValues.push_back(fir::getBase(
              genExprValue(*Fortran::semantics::GetExpr(*expr.v), stmtCtx)));
        } else {
          blockValues.push_back(zero);
        }
      }
    }
    mlir::Value streamValue;
    if (stream)
      streamValue = builder->createConvert(
          loc, builder->getI32Type(),
          fir::getBase(
              genExprValue(*Fortran::semantics::GetExpr(*stream), stmtCtx)));

    const auto &outerDoConstruct =
        std::get<std::optional<Fortran::parser::DoConstruct>>(kernel.t);

    llvm::SmallVector<mlir::Location> locs;
    locs.push_back(loc);
    llvm::SmallVector<mlir::Value> lbs, ubs, steps;

    mlir::Type idxTy = builder->getIndexType();

    llvm::SmallVector<mlir::Type> ivTypes;
    llvm::SmallVector<mlir::Location> ivLocs;
    llvm::SmallVector<mlir::Value> ivValues;
    Fortran::lower::pft::Evaluation *loopEval =
        &getEval().getFirstNestedEvaluation();
    for (unsigned i = 0; i < nestedLoops; ++i) {
      const Fortran::parser::LoopControl *loopControl;
      mlir::Location crtLoc = loc;
      if (i == 0) {
        loopControl = &*outerDoConstruct->GetLoopControl();
        crtLoc =
            genLocation(Fortran::parser::FindSourceLocation(outerDoConstruct));
      } else {
        auto *doCons = loopEval->getIf<Fortran::parser::DoConstruct>();
        assert(doCons && "expect do construct");
        loopControl = &*doCons->GetLoopControl();
        crtLoc = genLocation(Fortran::parser::FindSourceLocation(*doCons));
      }

      locs.push_back(crtLoc);

      const Fortran::parser::LoopControl::Bounds *bounds =
          std::get_if<Fortran::parser::LoopControl::Bounds>(&loopControl->u);
      assert(bounds && "Expected bounds on the loop construct");

      Fortran::semantics::Symbol &ivSym =
          bounds->name.thing.symbol->GetUltimate();
      ivValues.push_back(getSymbolAddress(ivSym));

      lbs.push_back(builder->createConvert(
          crtLoc, idxTy,
          fir::getBase(genExprValue(*Fortran::semantics::GetExpr(bounds->lower),
                                    stmtCtx))));
      ubs.push_back(builder->createConvert(
          crtLoc, idxTy,
          fir::getBase(genExprValue(*Fortran::semantics::GetExpr(bounds->upper),
                                    stmtCtx))));
      if (bounds->step)
        steps.push_back(fir::getBase(
            genExprValue(*Fortran::semantics::GetExpr(bounds->step), stmtCtx)));
      else // If `step` is not present, assume it is `1`.
        steps.push_back(builder->createIntegerConstant(loc, idxTy, 1));

      ivTypes.push_back(idxTy);
      ivLocs.push_back(crtLoc);
      if (i < nestedLoops - 1)
        loopEval = &*std::next(loopEval->getNestedEvaluations().begin());
    }

    auto op = builder->create<cuf::KernelOp>(
        loc, gridValues, blockValues, streamValue, lbs, ubs, steps, n,
        mlir::ValueRange(reduceOperands), builder->getArrayAttr(reduceAttrs));
    builder->createBlock(&op.getRegion(), op.getRegion().end(), ivTypes,
                         ivLocs);
    mlir::Block &b = op.getRegion().back();
    builder->setInsertionPointToStart(&b);

    Fortran::lower::pft::Evaluation *crtEval = &getEval();
    if (crtEval->lowerAsUnstructured())
      Fortran::lower::createEmptyRegionBlocks<fir::FirEndOp>(
          *builder, crtEval->getNestedEvaluations());
    builder->setInsertionPointToStart(&b);

    for (auto [arg, value] : llvm::zip(
             op.getLoopRegions().front()->front().getArguments(), ivValues)) {
      mlir::Value convArg =
          builder->createConvert(loc, fir::unwrapRefType(value.getType()), arg);
      builder->create<fir::StoreOp>(loc, convArg, value);
    }

    if (crtEval->lowerAsStructured()) {
      crtEval = &crtEval->getFirstNestedEvaluation();
      for (int64_t i = 1; i < nestedLoops; i++)
        crtEval = &*std::next(crtEval->getNestedEvaluations().begin());
    }

    // Generate loop body
    for (Fortran::lower::pft::Evaluation &e : crtEval->getNestedEvaluations())
      genFIR(e);

    builder->create<fir::FirEndOp>(loc);
    builder->setInsertionPointAfter(op);
  }

  void genFIR(const Fortran::parser::OpenMPConstruct &omp) {
    mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
    genOpenMPConstruct(*this, localSymbols, bridge.getSemanticsContext(),
                       getEval(), omp);
    builder->restoreInsertionPoint(insertPt);

    // Register if a target region was found
    ompDeviceCodeFound =
        ompDeviceCodeFound || Fortran::lower::isOpenMPTargetConstruct(omp);
  }

  void genFIR(const Fortran::parser::OpenMPDeclarativeConstruct &ompDecl) {
    mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
    // Register if a declare target construct intended for a target device was
    // found
    ompDeviceCodeFound =
        ompDeviceCodeFound ||
        Fortran::lower::isOpenMPDeviceDeclareTarget(
            *this, bridge.getSemanticsContext(), getEval(), ompDecl);
    Fortran::lower::gatherOpenMPDeferredDeclareTargets(
        *this, bridge.getSemanticsContext(), getEval(), ompDecl,
        ompDeferredDeclareTarget);
    genOpenMPDeclarativeConstruct(
        *this, localSymbols, bridge.getSemanticsContext(), getEval(), ompDecl);
    builder->restoreInsertionPoint(insertPt);
  }

  /// Generate FIR for a SELECT CASE statement.
  /// The selector may have CHARACTER, INTEGER, or LOGICAL type.
  void genFIR(const Fortran::parser::SelectCaseStmt &stmt) {
    Fortran::lower::pft::Evaluation &eval = getEval();
    Fortran::lower::pft::Evaluation *parentConstruct = eval.parentConstruct;
    assert(!activeConstructStack.empty() &&
           &activeConstructStack.back().eval == parentConstruct &&
           "select case construct is not active");
    Fortran::lower::StatementContext &stmtCtx =
        activeConstructStack.back().stmtCtx;
    const Fortran::lower::SomeExpr *expr = Fortran::semantics::GetExpr(
        std::get<Fortran::parser::Scalar<Fortran::parser::Expr>>(stmt.t));
    bool isCharSelector = isCharacterCategory(expr->GetType()->category());
    bool isLogicalSelector = isLogicalCategory(expr->GetType()->category());
    mlir::MLIRContext *context = builder->getContext();
    mlir::Location loc = toLocation();
    auto charValue = [&](const Fortran::lower::SomeExpr *expr) {
      fir::ExtendedValue exv = genExprAddr(*expr, stmtCtx, &loc);
      return exv.match(
          [&](const fir::CharBoxValue &cbv) {
            return fir::factory::CharacterExprHelper{*builder, loc}
                .createEmboxChar(cbv.getAddr(), cbv.getLen());
          },
          [&](auto) {
            fir::emitFatalError(loc, "not a character");
            return mlir::Value{};
          });
    };
    mlir::Value selector;
    if (isCharSelector) {
      selector = charValue(expr);
    } else {
      selector = createFIRExpr(loc, expr, stmtCtx);
      if (isLogicalSelector)
        selector = builder->createConvert(loc, builder->getI1Type(), selector);
    }
    mlir::Type selectType = selector.getType();
    llvm::SmallVector<mlir::Attribute> attrList;
    llvm::SmallVector<mlir::Value> valueList;
    llvm::SmallVector<mlir::Block *> blockList;
    mlir::Block *defaultBlock = parentConstruct->constructExit->block;
    using CaseValue = Fortran::parser::Scalar<Fortran::parser::ConstantExpr>;
    auto addValue = [&](const CaseValue &caseValue) {
      const Fortran::lower::SomeExpr *expr =
          Fortran::semantics::GetExpr(caseValue.thing);
      if (isCharSelector)
        valueList.push_back(charValue(expr));
      else if (isLogicalSelector)
        valueList.push_back(builder->createConvert(
            loc, selectType, createFIRExpr(toLocation(), expr, stmtCtx)));
      else
        valueList.push_back(builder->createIntegerConstant(
            loc, selectType, *Fortran::evaluate::ToInt64(*expr)));
    };
    for (Fortran::lower::pft::Evaluation *e = eval.controlSuccessor; e;
         e = e->controlSuccessor) {
      const auto &caseStmt = e->getIf<Fortran::parser::CaseStmt>();
      assert(e->block && "missing CaseStmt block");
      const auto &caseSelector =
          std::get<Fortran::parser::CaseSelector>(caseStmt->t);
      const auto *caseValueRangeList =
          std::get_if<std::list<Fortran::parser::CaseValueRange>>(
              &caseSelector.u);
      if (!caseValueRangeList) {
        defaultBlock = e->block;
        continue;
      }
      for (const Fortran::parser::CaseValueRange &caseValueRange :
           *caseValueRangeList) {
        blockList.push_back(e->block);
        if (const auto *caseValue = std::get_if<CaseValue>(&caseValueRange.u)) {
          attrList.push_back(fir::PointIntervalAttr::get(context));
          addValue(*caseValue);
          continue;
        }
        const auto &caseRange =
            std::get<Fortran::parser::CaseValueRange::Range>(caseValueRange.u);
        if (caseRange.lower && caseRange.upper) {
          attrList.push_back(fir::ClosedIntervalAttr::get(context));
          addValue(*caseRange.lower);
          addValue(*caseRange.upper);
        } else if (caseRange.lower) {
          attrList.push_back(fir::LowerBoundAttr::get(context));
          addValue(*caseRange.lower);
        } else {
          attrList.push_back(fir::UpperBoundAttr::get(context));
          addValue(*caseRange.upper);
        }
      }
    }
    // Skip a logical default block that can never be referenced.
    if (isLogicalSelector && attrList.size() == 2)
      defaultBlock = parentConstruct->constructExit->block;
    attrList.push_back(mlir::UnitAttr::get(context));
    blockList.push_back(defaultBlock);

    // Generate a fir::SelectCaseOp. Explicit branch code is better for the
    // LOGICAL type. The CHARACTER type does not have downstream SelectOp
    // support. The -no-structured-fir option can be used to force generation
    // of INTEGER type branch code.
    if (!isLogicalSelector && !isCharSelector &&
        !getEval().forceAsUnstructured()) {
      // The selector is in an ssa register. Any temps that may have been
      // generated while evaluating it can be cleaned up now.
      stmtCtx.finalizeAndReset();
      builder->create<fir::SelectCaseOp>(loc, selector, attrList, valueList,
                                         blockList);
      return;
    }

    // Generate a sequence of case value comparisons and branches.
    auto caseValue = valueList.begin();
    auto caseBlock = blockList.begin();
    for (mlir::Attribute attr : attrList) {
      if (mlir::isa<mlir::UnitAttr>(attr)) {
        genBranch(*caseBlock++);
        break;
      }
      auto genCond = [&](mlir::Value rhs,
                         mlir::arith::CmpIPredicate pred) -> mlir::Value {
        if (!isCharSelector)
          return builder->create<mlir::arith::CmpIOp>(loc, pred, selector, rhs);
        fir::factory::CharacterExprHelper charHelper{*builder, loc};
        std::pair<mlir::Value, mlir::Value> lhsVal =
            charHelper.createUnboxChar(selector);
        std::pair<mlir::Value, mlir::Value> rhsVal =
            charHelper.createUnboxChar(rhs);
        return fir::runtime::genCharCompare(*builder, loc, pred, lhsVal.first,
                                            lhsVal.second, rhsVal.first,
                                            rhsVal.second);
      };
      mlir::Block *newBlock = insertBlock(*caseBlock);
      if (mlir::isa<fir::ClosedIntervalAttr>(attr)) {
        mlir::Block *newBlock2 = insertBlock(*caseBlock);
        mlir::Value cond =
            genCond(*caseValue++, mlir::arith::CmpIPredicate::sge);
        genConditionalBranch(cond, newBlock, newBlock2);
        builder->setInsertionPointToEnd(newBlock);
        mlir::Value cond2 =
            genCond(*caseValue++, mlir::arith::CmpIPredicate::sle);
        genConditionalBranch(cond2, *caseBlock++, newBlock2);
        builder->setInsertionPointToEnd(newBlock2);
        continue;
      }
      mlir::arith::CmpIPredicate pred;
      if (mlir::isa<fir::PointIntervalAttr>(attr)) {
        pred = mlir::arith::CmpIPredicate::eq;
      } else if (mlir::isa<fir::LowerBoundAttr>(attr)) {
        pred = mlir::arith::CmpIPredicate::sge;
      } else {
        assert(mlir::isa<fir::UpperBoundAttr>(attr) && "unexpected predicate");
        pred = mlir::arith::CmpIPredicate::sle;
      }
      mlir::Value cond = genCond(*caseValue++, pred);
      genConditionalBranch(cond, *caseBlock++, newBlock);
      builder->setInsertionPointToEnd(newBlock);
    }
    assert(caseValue == valueList.end() && caseBlock == blockList.end() &&
           "select case list mismatch");
  }

  fir::ExtendedValue
  genAssociateSelector(const Fortran::lower::SomeExpr &selector,
                       Fortran::lower::StatementContext &stmtCtx) {
    if (lowerToHighLevelFIR())
      return genExprAddr(selector, stmtCtx);
    return Fortran::lower::isArraySectionWithoutVectorSubscript(selector)
               ? Fortran::lower::createSomeArrayBox(*this, selector,
                                                    localSymbols, stmtCtx)
               : genExprAddr(selector, stmtCtx);
  }

  void genFIR(const Fortran::parser::AssociateConstruct &) {
    Fortran::lower::pft::Evaluation &eval = getEval();
    Fortran::lower::StatementContext stmtCtx;
    pushActiveConstruct(eval, stmtCtx);
    for (Fortran::lower::pft::Evaluation &e : eval.getNestedEvaluations()) {
      setCurrentPosition(e.position);
      if (auto *stmt = e.getIf<Fortran::parser::AssociateStmt>()) {
        if (eval.lowerAsUnstructured())
          maybeStartBlock(e.block);
        localSymbols.pushScope();
        for (const Fortran::parser::Association &assoc :
             std::get<std::list<Fortran::parser::Association>>(stmt->t)) {
          Fortran::semantics::Symbol &sym =
              *std::get<Fortran::parser::Name>(assoc.t).symbol;
          const Fortran::lower::SomeExpr &selector =
              *sym.get<Fortran::semantics::AssocEntityDetails>().expr();
          addSymbol(sym, genAssociateSelector(selector, stmtCtx));
        }
      } else if (e.getIf<Fortran::parser::EndAssociateStmt>()) {
        if (eval.lowerAsUnstructured())
          maybeStartBlock(e.block);
        localSymbols.popScope();
      } else {
        genFIR(e);
      }
    }
    popActiveConstruct();
  }

  void genFIR(const Fortran::parser::BlockConstruct &blockConstruct) {
    Fortran::lower::pft::Evaluation &eval = getEval();
    Fortran::lower::StatementContext stmtCtx;
    pushActiveConstruct(eval, stmtCtx);
    for (Fortran::lower::pft::Evaluation &e : eval.getNestedEvaluations()) {
      setCurrentPosition(e.position);
      if (e.getIf<Fortran::parser::BlockStmt>()) {
        if (eval.lowerAsUnstructured())
          maybeStartBlock(e.block);
        const Fortran::parser::CharBlock &endPosition =
            eval.getLastNestedEvaluation().position;
        localSymbols.pushScope();
        mlir::Value stackPtr = builder->genStackSave(toLocation());
        mlir::Location endLoc = genLocation(endPosition);
        stmtCtx.attachCleanup(
            [=]() { builder->genStackRestore(endLoc, stackPtr); });
        Fortran::semantics::Scope &scope =
            bridge.getSemanticsContext().FindScope(endPosition);
        scopeBlockIdMap.try_emplace(&scope, ++blockId);
        Fortran::lower::AggregateStoreMap storeMap;
        for (const Fortran::lower::pft::Variable &var :
             Fortran::lower::pft::getScopeVariableList(scope)) {
          // Do no instantiate again variables from the block host
          // that appears in specification of block variables.
          if (!var.hasSymbol() || !lookupSymbol(var.getSymbol()))
            instantiateVar(var, storeMap);
        }
      } else if (e.getIf<Fortran::parser::EndBlockStmt>()) {
        if (eval.lowerAsUnstructured())
          maybeStartBlock(e.block);
        localSymbols.popScope();
      } else {
        genFIR(e);
      }
    }
    popActiveConstruct();
  }

  void genFIR(const Fortran::parser::ChangeTeamConstruct &construct) {
    TODO(toLocation(), "coarray: ChangeTeamConstruct");
  }
  void genFIR(const Fortran::parser::ChangeTeamStmt &stmt) {
    TODO(toLocation(), "coarray: ChangeTeamStmt");
  }
  void genFIR(const Fortran::parser::EndChangeTeamStmt &stmt) {
    TODO(toLocation(), "coarray: EndChangeTeamStmt");
  }

  void genFIR(const Fortran::parser::CriticalConstruct &criticalConstruct) {
    setCurrentPositionAt(criticalConstruct);
    TODO(toLocation(), "coarray: CriticalConstruct");
  }
  void genFIR(const Fortran::parser::CriticalStmt &) {
    TODO(toLocation(), "coarray: CriticalStmt");
  }
  void genFIR(const Fortran::parser::EndCriticalStmt &) {
    TODO(toLocation(), "coarray: EndCriticalStmt");
  }

  void genFIR(const Fortran::parser::SelectRankConstruct &selectRankConstruct) {
    setCurrentPositionAt(selectRankConstruct);
    genCaseOrRankConstruct();
  }

  void genFIR(const Fortran::parser::SelectRankStmt &selectRankStmt) {
    // Generate a fir.select_case with the selector rank. The RANK(*) case,
    // if any, is handles with a conditional branch before the fir.select_case.
    mlir::Type rankType = builder->getIntegerType(8);
    mlir::MLIRContext *context = builder->getContext();
    mlir::Location loc = toLocation();
    // Build block list for fir.select_case, and identify RANK(*) block, if any.
    // Default block must be placed last in the fir.select_case block list.
    mlir::Block *rankStarBlock = nullptr;
    Fortran::lower::pft::Evaluation &eval = getEval();
    mlir::Block *defaultBlock = eval.parentConstruct->constructExit->block;
    llvm::SmallVector<mlir::Attribute> attrList;
    llvm::SmallVector<mlir::Value> valueList;
    llvm::SmallVector<mlir::Block *> blockList;
    for (Fortran::lower::pft::Evaluation *e = eval.controlSuccessor; e;
         e = e->controlSuccessor) {
      if (const auto *rankCaseStmt =
              e->getIf<Fortran::parser::SelectRankCaseStmt>()) {
        const auto &rank = std::get<Fortran::parser::SelectRankCaseStmt::Rank>(
            rankCaseStmt->t);
        assert(e->block && "missing SelectRankCaseStmt block");
        Fortran::common::visit(
            Fortran::common::visitors{
                [&](const Fortran::parser::ScalarIntConstantExpr &rankExpr) {
                  blockList.emplace_back(e->block);
                  attrList.emplace_back(fir::PointIntervalAttr::get(context));
                  std::optional<std::int64_t> rankCst =
                      Fortran::evaluate::ToInt64(
                          Fortran::semantics::GetExpr(rankExpr));
                  assert(rankCst.has_value() &&
                         "rank expr must be constant integer");
                  valueList.emplace_back(
                      builder->createIntegerConstant(loc, rankType, *rankCst));
                },
                [&](const Fortran::parser::Star &) {
                  rankStarBlock = e->block;
                },
                [&](const Fortran::parser::Default &) {
                  defaultBlock = e->block;
                }},
            rank.u);
      }
    }
    attrList.push_back(mlir::UnitAttr::get(context));
    blockList.push_back(defaultBlock);

    // Lower selector.
    assert(!activeConstructStack.empty() && "must be inside construct");
    assert(!activeConstructStack.back().selector &&
           "selector should not yet be set");
    Fortran::lower::StatementContext &stmtCtx =
        activeConstructStack.back().stmtCtx;
    const Fortran::lower::SomeExpr *selectorExpr = Fortran::common::visit(
        [](const auto &x) { return Fortran::semantics::GetExpr(x); },
        std::get<Fortran::parser::Selector>(selectRankStmt.t).u);
    assert(selectorExpr && "failed to retrieve selector expr");
    hlfir::Entity selector = Fortran::lower::convertExprToHLFIR(
        loc, *this, *selectorExpr, localSymbols, stmtCtx);
    activeConstructStack.back().selector = selector;

    // Deal with assumed-size first. They must fall into RANK(*) if present, or
    // the default case (F'2023 11.1.10.2.). The selector cannot be an
    // assumed-size if it is allocatable or pointer, so the check is skipped.
    if (!Fortran::evaluate::IsAllocatableOrPointerObject(*selectorExpr)) {
      mlir::Value isAssumedSize = builder->create<fir::IsAssumedSizeOp>(
          loc, builder->getI1Type(), selector);
      // Create new block to hold the fir.select_case for the non assumed-size
      // cases.
      mlir::Block *selectCaseBlock = insertBlock(blockList[0]);
      mlir::Block *assumedSizeBlock =
          rankStarBlock ? rankStarBlock : defaultBlock;
      builder->create<mlir::cf::CondBranchOp>(loc, isAssumedSize,
                                              assumedSizeBlock, std::nullopt,
                                              selectCaseBlock, std::nullopt);
      startBlock(selectCaseBlock);
    }
    // Create fir.select_case for the other rank cases.
    mlir::Value rank = builder->create<fir::BoxRankOp>(loc, rankType, selector);
    stmtCtx.finalizeAndReset();
    builder->create<fir::SelectCaseOp>(loc, rank, attrList, valueList,
                                       blockList);
  }

  // Get associating entity symbol inside case statement scope.
  static const Fortran::semantics::Symbol &
  getAssociatingEntitySymbol(const Fortran::semantics::Scope &scope) {
    const Fortran::semantics::Symbol *assocSym = nullptr;
    for (const auto &sym : scope.GetSymbols()) {
      if (sym->has<Fortran::semantics::AssocEntityDetails>()) {
        assert(!assocSym &&
               "expect only one associating entity symbol in this scope");
        assocSym = &*sym;
      }
    }
    assert(assocSym && "should contain associating entity symbol");
    return *assocSym;
  }

  void genFIR(const Fortran::parser::SelectRankCaseStmt &stmt) {
    assert(!activeConstructStack.empty() &&
           "must be inside select rank construct");
    // Pop previous associating entity mapping, if any, and push scope for new
    // mapping.
    if (activeConstructStack.back().pushedScope)
      localSymbols.popScope();
    localSymbols.pushScope();
    activeConstructStack.back().pushedScope = true;
    const Fortran::semantics::Symbol &assocEntitySymbol =
        getAssociatingEntitySymbol(
            bridge.getSemanticsContext().FindScope(getEval().position));
    const auto &details =
        assocEntitySymbol.get<Fortran::semantics::AssocEntityDetails>();
    assert(!activeConstructStack.empty() &&
           activeConstructStack.back().selector.has_value() &&
           "selector must have been created");
    // Get lowered value for the selector.
    hlfir::Entity selector = *activeConstructStack.back().selector;
    assert(selector.isVariable() && "assumed-rank selector are variables");
    // Cook selector mlir::Value according to rank case and map it to
    // associating entity symbol.
    Fortran::lower::StatementContext stmtCtx;
    mlir::Location loc = toLocation();
    if (details.IsAssumedRank()) {
      fir::ExtendedValue selectorExv = Fortran::lower::translateToExtendedValue(
          loc, *builder, selector, stmtCtx);
      addSymbol(assocEntitySymbol, selectorExv);
    } else if (details.IsAssumedSize()) {
      // Create rank-1 assumed-size from descriptor. Assumed-size are contiguous
      // so a new entity can be built from scratch using the base address, type
      // parameters and dynamic type. The selector cannot be a
      // POINTER/ALLOCATBLE as per F'2023 C1160.
      fir::ExtendedValue newExv;
      llvm::SmallVector assumeSizeExtents{
          builder->createMinusOneInteger(loc, builder->getIndexType())};
      mlir::Value baseAddr =
          hlfir::genVariableRawAddress(loc, *builder, selector);
      mlir::Type eleType =
          fir::unwrapSequenceType(fir::unwrapRefType(baseAddr.getType()));
      mlir::Type rank1Type =
          fir::ReferenceType::get(builder->getVarLenSeqTy(eleType, 1));
      baseAddr = builder->createConvert(loc, rank1Type, baseAddr);
      if (selector.isCharacter()) {
        mlir::Value len = hlfir::genCharLength(loc, *builder, selector);
        newExv = fir::CharArrayBoxValue{baseAddr, len, assumeSizeExtents};
      } else if (selector.isDerivedWithLengthParameters()) {
        TODO(loc, "RANK(*) with parameterized derived type selector");
      } else if (selector.isPolymorphic()) {
        TODO(loc, "RANK(*) with polymorphic selector");
      } else {
        // Simple intrinsic or derived type.
        newExv = fir::ArrayBoxValue{baseAddr, assumeSizeExtents};
      }
      addSymbol(assocEntitySymbol, newExv);
    } else {
      int rank = details.rank().value();
      auto boxTy =
          mlir::cast<fir::BaseBoxType>(fir::unwrapRefType(selector.getType()));
      mlir::Type newBoxType = boxTy.getBoxTypeWithNewShape(rank);
      if (fir::isa_ref_type(selector.getType()))
        newBoxType = fir::ReferenceType::get(newBoxType);
      // Give rank info to value via cast, and get rid of the box if not needed
      // (simple scalars, contiguous arrays... This is done by
      // translateVariableToExtendedValue).
      hlfir::Entity rankedBox{
          builder->createConvert(loc, newBoxType, selector)};
      bool isSimplyContiguous = Fortran::evaluate::IsSimplyContiguous(
          assocEntitySymbol, getFoldingContext());
      fir::ExtendedValue newExv = Fortran::lower::translateToExtendedValue(
          loc, *builder, rankedBox, stmtCtx, isSimplyContiguous);

      // Non deferred length parameters of character allocatable/pointer
      // MutableBoxValue should be properly set before binding it to a symbol in
      // order to get correct assignment semantics.
      if (const fir::MutableBoxValue *mutableBox =
              newExv.getBoxOf<fir::MutableBoxValue>()) {
        if (selector.isCharacter()) {
          auto dynamicType =
              Fortran::evaluate::DynamicType::From(assocEntitySymbol);
          if (!dynamicType.value().HasDeferredTypeParameter()) {
            llvm::SmallVector<mlir::Value> lengthParams;
            hlfir::genLengthParameters(loc, *builder, selector, lengthParams);
            newExv = fir::MutableBoxValue{rankedBox, lengthParams,
                                          mutableBox->getMutableProperties()};
          }
        }
      }
      addSymbol(assocEntitySymbol, newExv);
    }
    // Statements inside rank case are lowered by SelectRankConstruct visit.
  }

  void genFIR(const Fortran::parser::SelectTypeConstruct &selectTypeConstruct) {
    mlir::MLIRContext *context = builder->getContext();
    Fortran::lower::StatementContext stmtCtx;
    fir::ExtendedValue selector;
    llvm::SmallVector<mlir::Attribute> attrList;
    llvm::SmallVector<mlir::Block *> blockList;
    unsigned typeGuardIdx = 0;
    std::size_t defaultAttrPos = std::numeric_limits<size_t>::max();
    bool hasLocalScope = false;
    llvm::SmallVector<const Fortran::semantics::Scope *> typeCaseScopes;

    const auto &typeCaseList =
        std::get<std::list<Fortran::parser::SelectTypeConstruct::TypeCase>>(
            selectTypeConstruct.t);
    for (const auto &typeCase : typeCaseList) {
      const auto &stmt =
          std::get<Fortran::parser::Statement<Fortran::parser::TypeGuardStmt>>(
              typeCase.t);
      const Fortran::semantics::Scope &scope =
          bridge.getSemanticsContext().FindScope(stmt.source);
      typeCaseScopes.push_back(&scope);
    }

    pushActiveConstruct(getEval(), stmtCtx);
    llvm::SmallVector<Fortran::lower::pft::Evaluation *> exits, fallThroughs;
    collectFinalEvaluations(getEval(), exits, fallThroughs);
    Fortran::lower::pft::Evaluation &constructExit = *getEval().constructExit;

    for (Fortran::lower::pft::Evaluation &eval :
         getEval().getNestedEvaluations()) {
      setCurrentPosition(eval.position);
      mlir::Location loc = toLocation();
      if (auto *selectTypeStmt =
              eval.getIf<Fortran::parser::SelectTypeStmt>()) {
        // A genFIR(SelectTypeStmt) call would have unwanted side effects.
        maybeStartBlock(eval.block);
        // Retrieve the selector
        const auto &s = std::get<Fortran::parser::Selector>(selectTypeStmt->t);
        if (const auto *v = std::get_if<Fortran::parser::Variable>(&s.u))
          selector = genExprBox(loc, *Fortran::semantics::GetExpr(*v), stmtCtx);
        else if (const auto *e = std::get_if<Fortran::parser::Expr>(&s.u))
          selector = genExprBox(loc, *Fortran::semantics::GetExpr(*e), stmtCtx);

        // Going through the controlSuccessor first to create the
        // fir.select_type operation.
        mlir::Block *defaultBlock = eval.parentConstruct->constructExit->block;
        for (Fortran::lower::pft::Evaluation *e = eval.controlSuccessor; e;
             e = e->controlSuccessor) {
          const auto &typeGuardStmt =
              e->getIf<Fortran::parser::TypeGuardStmt>();
          const auto &guard =
              std::get<Fortran::parser::TypeGuardStmt::Guard>(typeGuardStmt->t);
          assert(e->block && "missing TypeGuardStmt block");
          // CLASS DEFAULT
          if (std::holds_alternative<Fortran::parser::Default>(guard.u)) {
            defaultBlock = e->block;
            // Keep track of the actual position of the CLASS DEFAULT type guard
            // in the SELECT TYPE construct.
            defaultAttrPos = attrList.size();
            continue;
          }

          blockList.push_back(e->block);
          if (const auto *typeSpec =
                  std::get_if<Fortran::parser::TypeSpec>(&guard.u)) {
            // TYPE IS
            mlir::Type ty;
            if (std::holds_alternative<Fortran::parser::IntrinsicTypeSpec>(
                    typeSpec->u)) {
              const Fortran::semantics::IntrinsicTypeSpec *intrinsic =
                  typeSpec->declTypeSpec->AsIntrinsic();
              int kind =
                  Fortran::evaluate::ToInt64(intrinsic->kind()).value_or(kind);
              llvm::SmallVector<Fortran::lower::LenParameterTy> params;
              ty = genType(intrinsic->category(), kind, params);
            } else {
              const Fortran::semantics::DerivedTypeSpec *derived =
                  typeSpec->declTypeSpec->AsDerived();
              ty = genType(*derived);
            }
            attrList.push_back(fir::ExactTypeAttr::get(ty));
          } else if (const auto *derived =
                         std::get_if<Fortran::parser::DerivedTypeSpec>(
                             &guard.u)) {
            // CLASS IS
            assert(derived->derivedTypeSpec && "derived type spec is null");
            mlir::Type ty = genType(*(derived->derivedTypeSpec));
            attrList.push_back(fir::SubclassAttr::get(ty));
          }
        }
        attrList.push_back(mlir::UnitAttr::get(context));
        blockList.push_back(defaultBlock);
        builder->create<fir::SelectTypeOp>(loc, fir::getBase(selector),
                                           attrList, blockList);

        // If the actual position of CLASS DEFAULT type guard is not the last
        // one, it needs to be put back at its correct position for the rest of
        // the processing. TypeGuardStmt are processed in the same order they
        // appear in the Fortran code.
        if (defaultAttrPos < attrList.size() - 1) {
          auto attrIt = attrList.begin();
          attrIt = attrIt + defaultAttrPos;
          auto blockIt = blockList.begin();
          blockIt = blockIt + defaultAttrPos;
          attrList.insert(attrIt, mlir::UnitAttr::get(context));
          blockList.insert(blockIt, defaultBlock);
          attrList.pop_back();
          blockList.pop_back();
        }
      } else if (auto *typeGuardStmt =
                     eval.getIf<Fortran::parser::TypeGuardStmt>()) {
        // Map the type guard local symbol for the selector to a more precise
        // typed entity in the TypeGuardStmt when necessary.
        genFIR(eval);
        const auto &guard =
            std::get<Fortran::parser::TypeGuardStmt::Guard>(typeGuardStmt->t);
        if (hasLocalScope)
          localSymbols.popScope();
        localSymbols.pushScope();
        hasLocalScope = true;
        assert(attrList.size() >= typeGuardIdx &&
               "TypeGuard attribute missing");
        mlir::Attribute typeGuardAttr = attrList[typeGuardIdx];
        mlir::Block *typeGuardBlock = blockList[typeGuardIdx];
        mlir::OpBuilder::InsertPoint crtInsPt = builder->saveInsertionPoint();
        builder->setInsertionPointToStart(typeGuardBlock);

        auto addAssocEntitySymbol = [&](fir::ExtendedValue exv) {
          for (auto &symbol : typeCaseScopes[typeGuardIdx]->GetSymbols()) {
            if (symbol->GetUltimate()
                    .detailsIf<Fortran::semantics::AssocEntityDetails>()) {
              addSymbol(symbol, exv);
              break;
            }
          }
        };

        mlir::Type baseTy = fir::getBase(selector).getType();
        bool isPointer = fir::isPointerType(baseTy);
        bool isAllocatable = fir::isAllocatableType(baseTy);
        bool isArray =
            mlir::isa<fir::SequenceType>(fir::dyn_cast_ptrOrBoxEleTy(baseTy));
        const fir::BoxValue *selectorBox = selector.getBoxOf<fir::BoxValue>();
        if (std::holds_alternative<Fortran::parser::Default>(guard.u)) {
          // CLASS DEFAULT
          addAssocEntitySymbol(selector);
        } else if (const auto *typeSpec =
                       std::get_if<Fortran::parser::TypeSpec>(&guard.u)) {
          // TYPE IS
          fir::ExactTypeAttr attr =
              mlir::dyn_cast<fir::ExactTypeAttr>(typeGuardAttr);
          mlir::Value exactValue;
          mlir::Type addrTy = attr.getType();
          if (isArray) {
            auto seqTy = mlir::dyn_cast<fir::SequenceType>(
                fir::dyn_cast_ptrOrBoxEleTy(baseTy));
            addrTy = fir::SequenceType::get(seqTy.getShape(), attr.getType());
          }
          if (isPointer)
            addrTy = fir::PointerType::get(addrTy);
          if (isAllocatable)
            addrTy = fir::HeapType::get(addrTy);
          if (std::holds_alternative<Fortran::parser::IntrinsicTypeSpec>(
                  typeSpec->u)) {
            mlir::Type refTy = fir::ReferenceType::get(addrTy);
            if (isPointer || isAllocatable)
              refTy = addrTy;
            exactValue = builder->create<fir::BoxAddrOp>(
                loc, refTy, fir::getBase(selector));
            const Fortran::semantics::IntrinsicTypeSpec *intrinsic =
                typeSpec->declTypeSpec->AsIntrinsic();
            if (isArray) {
              mlir::Value exact = builder->create<fir::ConvertOp>(
                  loc, fir::BoxType::get(addrTy), fir::getBase(selector));
              addAssocEntitySymbol(selectorBox->clone(exact));
            } else if (intrinsic->category() ==
                       Fortran::common::TypeCategory::Character) {
              auto charTy = mlir::dyn_cast<fir::CharacterType>(attr.getType());
              mlir::Value charLen =
                  fir::factory::CharacterExprHelper(*builder, loc)
                      .readLengthFromBox(fir::getBase(selector), charTy);
              addAssocEntitySymbol(fir::CharBoxValue(exactValue, charLen));
            } else {
              addAssocEntitySymbol(exactValue);
            }
          } else if (std::holds_alternative<Fortran::parser::DerivedTypeSpec>(
                         typeSpec->u)) {
            exactValue = builder->create<fir::ConvertOp>(
                loc, fir::BoxType::get(addrTy), fir::getBase(selector));
            addAssocEntitySymbol(selectorBox->clone(exactValue));
          }
        } else if (std::holds_alternative<Fortran::parser::DerivedTypeSpec>(
                       guard.u)) {
          // CLASS IS
          fir::SubclassAttr attr =
              mlir::dyn_cast<fir::SubclassAttr>(typeGuardAttr);
          mlir::Type addrTy = attr.getType();
          if (isArray) {
            auto seqTy = mlir::dyn_cast<fir::SequenceType>(
                fir::dyn_cast_ptrOrBoxEleTy(baseTy));
            addrTy = fir::SequenceType::get(seqTy.getShape(), attr.getType());
          }
          if (isPointer)
            addrTy = fir::PointerType::get(addrTy);
          if (isAllocatable)
            addrTy = fir::HeapType::get(addrTy);
          mlir::Type classTy = fir::ClassType::get(addrTy);
          if (classTy == baseTy) {
            addAssocEntitySymbol(selector);
          } else {
            mlir::Value derived = builder->create<fir::ConvertOp>(
                loc, classTy, fir::getBase(selector));
            addAssocEntitySymbol(selectorBox->clone(derived));
          }
        }
        builder->restoreInsertionPoint(crtInsPt);
        ++typeGuardIdx;
      } else if (eval.getIf<Fortran::parser::EndSelectStmt>()) {
        maybeStartBlock(eval.block);
        if (hasLocalScope)
          localSymbols.popScope();
      } else {
        genFIR(eval);
      }
      if (blockIsUnterminated()) {
        if (llvm::is_contained(exits, &eval))
          genConstructExitBranch(constructExit);
        else if (llvm::is_contained(fallThroughs, &eval))
          genBranch(eval.lexicalSuccessor->block);
      }
    }
    popActiveConstruct();
  }

  //===--------------------------------------------------------------------===//
  // IO statements (see io.h)
  //===--------------------------------------------------------------------===//

  void genFIR(const Fortran::parser::BackspaceStmt &stmt) {
    mlir::Value iostat = genBackspaceStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::CloseStmt &stmt) {
    mlir::Value iostat = genCloseStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::EndfileStmt &stmt) {
    mlir::Value iostat = genEndfileStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::FlushStmt &stmt) {
    mlir::Value iostat = genFlushStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::InquireStmt &stmt) {
    mlir::Value iostat = genInquireStatement(*this, stmt);
    if (const auto *specs =
            std::get_if<std::list<Fortran::parser::InquireSpec>>(&stmt.u))
      genIoConditionBranches(getEval(), *specs, iostat);
  }
  void genFIR(const Fortran::parser::OpenStmt &stmt) {
    mlir::Value iostat = genOpenStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::PrintStmt &stmt) {
    genPrintStatement(*this, stmt);
  }
  void genFIR(const Fortran::parser::ReadStmt &stmt) {
    mlir::Value iostat = genReadStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.controls, iostat);
  }
  void genFIR(const Fortran::parser::RewindStmt &stmt) {
    mlir::Value iostat = genRewindStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::WaitStmt &stmt) {
    mlir::Value iostat = genWaitStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::WriteStmt &stmt) {
    mlir::Value iostat = genWriteStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.controls, iostat);
  }

  template <typename A>
  void genIoConditionBranches(Fortran::lower::pft::Evaluation &eval,
                              const A &specList, mlir::Value iostat) {
    if (!iostat)
      return;

    Fortran::parser::Label endLabel{};
    Fortran::parser::Label eorLabel{};
    Fortran::parser::Label errLabel{};
    bool hasIostat{};
    for (const auto &spec : specList) {
      Fortran::common::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::EndLabel &label) {
                endLabel = label.v;
              },
              [&](const Fortran::parser::EorLabel &label) {
                eorLabel = label.v;
              },
              [&](const Fortran::parser::ErrLabel &label) {
                errLabel = label.v;
              },
              [&](const Fortran::parser::StatVariable &) { hasIostat = true; },
              [](const auto &) {}},
          spec.u);
    }
    if (!endLabel && !eorLabel && !errLabel)
      return;

    // An ERR specifier branch is taken on any positive error value rather than
    // some single specific value. If ERR and IOSTAT specifiers are given and
    // END and EOR specifiers are allowed, the latter two specifiers must have
    // explicit branch targets to allow the ERR branch to be implemented as a
    // default/else target. A label=0 target for an absent END or EOR specifier
    // indicates that these specifiers have a fallthrough target. END and EOR
    // specifiers may appear on READ and WAIT statements.
    bool allSpecifiersRequired = errLabel && hasIostat &&
                                 (eval.isA<Fortran::parser::ReadStmt>() ||
                                  eval.isA<Fortran::parser::WaitStmt>());
    mlir::Value selector =
        builder->createConvert(toLocation(), builder->getIndexType(), iostat);
    llvm::SmallVector<int64_t> valueList;
    llvm::SmallVector<Fortran::parser::Label> labelList;
    if (eorLabel || allSpecifiersRequired) {
      valueList.push_back(Fortran::runtime::io::IostatEor);
      labelList.push_back(eorLabel ? eorLabel : 0);
    }
    if (endLabel || allSpecifiersRequired) {
      valueList.push_back(Fortran::runtime::io::IostatEnd);
      labelList.push_back(endLabel ? endLabel : 0);
    }
    if (errLabel) {
      // Must be last. Value 0 is interpreted as any positive value, or
      // equivalently as any value other than 0, IostatEor, or IostatEnd.
      valueList.push_back(0);
      labelList.push_back(errLabel);
    }
    genMultiwayBranch(selector, valueList, labelList, eval.nonNopSuccessor());
  }

  //===--------------------------------------------------------------------===//
  // Memory allocation and deallocation
  //===--------------------------------------------------------------------===//

  void genFIR(const Fortran::parser::AllocateStmt &stmt) {
    Fortran::lower::genAllocateStmt(*this, stmt, toLocation());
  }

  void genFIR(const Fortran::parser::DeallocateStmt &stmt) {
    Fortran::lower::genDeallocateStmt(*this, stmt, toLocation());
  }

  /// Nullify pointer object list
  ///
  /// For each pointer object, reset the pointer to a disassociated status.
  /// We do this by setting each pointer to null.
  void genFIR(const Fortran::parser::NullifyStmt &stmt) {
    mlir::Location loc = toLocation();
    for (auto &pointerObject : stmt.v) {
      const Fortran::lower::SomeExpr *expr =
          Fortran::semantics::GetExpr(pointerObject);
      assert(expr);
      if (Fortran::evaluate::IsProcedurePointer(*expr)) {
        Fortran::lower::StatementContext stmtCtx;
        hlfir::Entity pptr = Fortran::lower::convertExprToHLFIR(
            loc, *this, *expr, localSymbols, stmtCtx);
        auto boxTy{
            Fortran::lower::getUntypedBoxProcType(builder->getContext())};
        hlfir::Entity nullBoxProc(
            fir::factory::createNullBoxProc(*builder, loc, boxTy));
        builder->createStoreWithConvert(loc, nullBoxProc, pptr);
      } else {
        fir::MutableBoxValue box = genExprMutableBox(loc, *expr);
        fir::factory::disassociateMutableBox(*builder, loc, box);
      }
    }
  }

  //===--------------------------------------------------------------------===//

  void genFIR(const Fortran::parser::NotifyWaitStmt &stmt) {
    genNotifyWaitStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::EventPostStmt &stmt) {
    genEventPostStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::EventWaitStmt &stmt) {
    genEventWaitStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::FormTeamStmt &stmt) {
    genFormTeamStatement(*this, getEval(), stmt);
  }

  void genFIR(const Fortran::parser::LockStmt &stmt) {
    genLockStatement(*this, stmt);
  }

  fir::ExtendedValue
  genInitializerExprValue(const Fortran::lower::SomeExpr &expr,
                          Fortran::lower::StatementContext &stmtCtx) {
    return Fortran::lower::createSomeInitializerExpression(
        toLocation(), *this, expr, localSymbols, stmtCtx);
  }

  /// Return true if the current context is a conditionalized and implied
  /// iteration space.
  bool implicitIterationSpace() { return !implicitIterSpace.empty(); }

  /// Return true if context is currently an explicit iteration space. A scalar
  /// assignment expression may be contextually within a user-defined iteration
  /// space, transforming it into an array expression.
  bool explicitIterationSpace() { return explicitIterSpace.isActive(); }

  /// Generate an array assignment.
  /// This is an assignment expression with rank > 0. The assignment may or may
  /// not be in a WHERE and/or FORALL context.
  /// In a FORALL context, the assignment may be a pointer assignment and the \p
  /// lbounds and \p ubounds parameters should only be used in such a pointer
  /// assignment case. (If both are None then the array assignment cannot be a
  /// pointer assignment.)
  void genArrayAssignment(
      const Fortran::evaluate::Assignment &assign,
      Fortran::lower::StatementContext &localStmtCtx,
      std::optional<llvm::SmallVector<mlir::Value>> lbounds = std::nullopt,
      std::optional<llvm::SmallVector<mlir::Value>> ubounds = std::nullopt) {

    Fortran::lower::StatementContext &stmtCtx =
        explicitIterationSpace()
            ? explicitIterSpace.stmtContext()
            : (implicitIterationSpace() ? implicitIterSpace.stmtContext()
                                        : localStmtCtx);
    if (Fortran::lower::isWholeAllocatable(assign.lhs)) {
      // Assignment to allocatables may require the lhs to be
      // deallocated/reallocated. See Fortran 2018 10.2.1.3 p3
      Fortran::lower::createAllocatableArrayAssignment(
          *this, assign.lhs, assign.rhs, explicitIterSpace, implicitIterSpace,
          localSymbols, stmtCtx);
      return;
    }

    if (lbounds) {
      // Array of POINTER entities, with elemental assignment.
      if (!Fortran::lower::isWholePointer(assign.lhs))
        fir::emitFatalError(toLocation(), "pointer assignment to non-pointer");

      Fortran::lower::createArrayOfPointerAssignment(
          *this, assign.lhs, assign.rhs, explicitIterSpace, implicitIterSpace,
          *lbounds, ubounds, localSymbols, stmtCtx);
      return;
    }

    if (!implicitIterationSpace() && !explicitIterationSpace()) {
      // No masks and the iteration space is implied by the array, so create a
      // simple array assignment.
      Fortran::lower::createSomeArrayAssignment(*this, assign.lhs, assign.rhs,
                                                localSymbols, stmtCtx);
      return;
    }

    // If there is an explicit iteration space, generate an array assignment
    // with a user-specified iteration space and possibly with masks. These
    // assignments may *appear* to be scalar expressions, but the scalar
    // expression is evaluated at all points in the user-defined space much like
    // an ordinary array assignment. More specifically, the semantics inside the
    // FORALL much more closely resembles that of WHERE than a scalar
    // assignment.
    // Otherwise, generate a masked array assignment. The iteration space is
    // implied by the lhs array expression.
    Fortran::lower::createAnyMaskedArrayAssignment(
        *this, assign.lhs, assign.rhs, explicitIterSpace, implicitIterSpace,
        localSymbols, stmtCtx);
  }

#if !defined(NDEBUG)
  static bool isFuncResultDesignator(const Fortran::lower::SomeExpr &expr) {
    const Fortran::semantics::Symbol *sym =
        Fortran::evaluate::GetFirstSymbol(expr);
    return sym && sym->IsFuncResult();
  }
#endif

  inline fir::MutableBoxValue
  genExprMutableBox(mlir::Location loc,
                    const Fortran::lower::SomeExpr &expr) override final {
    if (lowerToHighLevelFIR())
      return Fortran::lower::convertExprToMutableBox(loc, *this, expr,
                                                     localSymbols);
    return Fortran::lower::createMutableBox(loc, *this, expr, localSymbols);
  }

  // Create the [newRank] array with the lower bounds to be passed to the
  // runtime as a descriptor.
  mlir::Value createLboundArray(llvm::ArrayRef<mlir::Value> lbounds,
                                mlir::Location loc) {
    mlir::Type indexTy = builder->getIndexType();
    mlir::Type boundArrayTy = fir::SequenceType::get(
        {static_cast<int64_t>(lbounds.size())}, builder->getI64Type());
    mlir::Value boundArray = builder->create<fir::AllocaOp>(loc, boundArrayTy);
    mlir::Value array = builder->create<fir::UndefOp>(loc, boundArrayTy);
    for (unsigned i = 0; i < lbounds.size(); ++i) {
      array = builder->create<fir::InsertValueOp>(
          loc, boundArrayTy, array, lbounds[i],
          builder->getArrayAttr({builder->getIntegerAttr(
              builder->getIndexType(), static_cast<int>(i))}));
    }
    builder->create<fir::StoreOp>(loc, array, boundArray);
    mlir::Type boxTy = fir::BoxType::get(boundArrayTy);
    mlir::Value ext =
        builder->createIntegerConstant(loc, indexTy, lbounds.size());
    llvm::SmallVector<mlir::Value> shapes = {ext};
    mlir::Value shapeOp = builder->genShape(loc, shapes);
    return builder->create<fir::EmboxOp>(loc, boxTy, boundArray, shapeOp);
  }

  // Generate pointer assignment with possibly empty bounds-spec. R1035: a
  // bounds-spec is a lower bound value.
  void genPointerAssignment(
      mlir::Location loc, const Fortran::evaluate::Assignment &assign,
      const Fortran::evaluate::Assignment::BoundsSpec &lbExprs) {
    Fortran::lower::StatementContext stmtCtx;

    if (!lowerToHighLevelFIR() &&
        Fortran::evaluate::IsProcedureDesignator(assign.rhs))
      TODO(loc, "procedure pointer assignment");
    if (Fortran::evaluate::IsProcedurePointer(assign.lhs)) {
      hlfir::Entity lhs = Fortran::lower::convertExprToHLFIR(
          loc, *this, assign.lhs, localSymbols, stmtCtx);
      if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
              assign.rhs)) {
        // rhs is null(). rhs being null(pptr) is handled in genNull.
        auto boxTy{
            Fortran::lower::getUntypedBoxProcType(builder->getContext())};
        hlfir::Entity rhs(
            fir::factory::createNullBoxProc(*builder, loc, boxTy));
        builder->createStoreWithConvert(loc, rhs, lhs);
        return;
      }
      hlfir::Entity rhs(getBase(Fortran::lower::convertExprToAddress(
          loc, *this, assign.rhs, localSymbols, stmtCtx)));
      builder->createStoreWithConvert(loc, rhs, lhs);
      return;
    }

    std::optional<Fortran::evaluate::DynamicType> lhsType =
        assign.lhs.GetType();
    // Delegate pointer association to unlimited polymorphic pointer
    // to the runtime. element size, type code, attribute and of
    // course base_addr might need to be updated.
    if (lhsType && lhsType->IsPolymorphic()) {
      if (!lowerToHighLevelFIR() && explicitIterationSpace())
        TODO(loc, "polymorphic pointer assignment in FORALL");
      llvm::SmallVector<mlir::Value> lbounds;
      for (const Fortran::evaluate::ExtentExpr &lbExpr : lbExprs)
        lbounds.push_back(
            fir::getBase(genExprValue(toEvExpr(lbExpr), stmtCtx)));
      fir::MutableBoxValue lhsMutableBox = genExprMutableBox(loc, assign.lhs);
      if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
              assign.rhs)) {
        fir::factory::disassociateMutableBox(*builder, loc, lhsMutableBox);
        return;
      }
      mlir::Value lhs = lhsMutableBox.getAddr();
      mlir::Value rhs = fir::getBase(genExprBox(loc, assign.rhs, stmtCtx));
      if (!lbounds.empty()) {
        mlir::Value boundsDesc = createLboundArray(lbounds, loc);
        Fortran::lower::genPointerAssociateLowerBounds(*builder, loc, lhs, rhs,
                                                       boundsDesc);
        return;
      }
      Fortran::lower::genPointerAssociate(*builder, loc, lhs, rhs);
      return;
    }

    llvm::SmallVector<mlir::Value> lbounds;
    for (const Fortran::evaluate::ExtentExpr &lbExpr : lbExprs)
      lbounds.push_back(fir::getBase(genExprValue(toEvExpr(lbExpr), stmtCtx)));
    if (!lowerToHighLevelFIR() && explicitIterationSpace()) {
      // Pointer assignment in FORALL context. Copy the rhs box value
      // into the lhs box variable.
      genArrayAssignment(assign, stmtCtx, lbounds);
      return;
    }
    fir::MutableBoxValue lhs = genExprMutableBox(loc, assign.lhs);
    Fortran::lower::associateMutableBox(*this, loc, lhs, assign.rhs, lbounds,
                                        stmtCtx);
  }

  // Create the 2 x newRank array with the bounds to be passed to the runtime as
  // a descriptor.
  mlir::Value createBoundArray(llvm::ArrayRef<mlir::Value> lbounds,
                               llvm::ArrayRef<mlir::Value> ubounds,
                               mlir::Location loc) {
    assert(lbounds.size() && ubounds.size());
    mlir::Type indexTy = builder->getIndexType();
    mlir::Type boundArrayTy = fir::SequenceType::get(
        {2, static_cast<int64_t>(lbounds.size())}, builder->getI64Type());
    mlir::Value boundArray = builder->create<fir::AllocaOp>(loc, boundArrayTy);
    mlir::Value array = builder->create<fir::UndefOp>(loc, boundArrayTy);
    for (unsigned i = 0; i < lbounds.size(); ++i) {
      array = builder->create<fir::InsertValueOp>(
          loc, boundArrayTy, array, lbounds[i],
          builder->getArrayAttr(
              {builder->getIntegerAttr(builder->getIndexType(), 0),
               builder->getIntegerAttr(builder->getIndexType(),
                                       static_cast<int>(i))}));
      array = builder->create<fir::InsertValueOp>(
          loc, boundArrayTy, array, ubounds[i],
          builder->getArrayAttr(
              {builder->getIntegerAttr(builder->getIndexType(), 1),
               builder->getIntegerAttr(builder->getIndexType(),
                                       static_cast<int>(i))}));
    }
    builder->create<fir::StoreOp>(loc, array, boundArray);
    mlir::Type boxTy = fir::BoxType::get(boundArrayTy);
    mlir::Value ext =
        builder->createIntegerConstant(loc, indexTy, lbounds.size());
    mlir::Value c2 = builder->createIntegerConstant(loc, indexTy, 2);
    llvm::SmallVector<mlir::Value> shapes = {c2, ext};
    mlir::Value shapeOp = builder->genShape(loc, shapes);
    return builder->create<fir::EmboxOp>(loc, boxTy, boundArray, shapeOp);
  }

  // Pointer assignment with bounds-remapping. R1036: a bounds-remapping is a
  // pair, lower bound and upper bound.
  void genPointerAssignment(
      mlir::Location loc, const Fortran::evaluate::Assignment &assign,
      const Fortran::evaluate::Assignment::BoundsRemapping &boundExprs) {
    Fortran::lower::StatementContext stmtCtx;
    llvm::SmallVector<mlir::Value> lbounds;
    llvm::SmallVector<mlir::Value> ubounds;
    for (const std::pair<Fortran::evaluate::ExtentExpr,
                         Fortran::evaluate::ExtentExpr> &pair : boundExprs) {
      const Fortran::evaluate::ExtentExpr &lbExpr = pair.first;
      const Fortran::evaluate::ExtentExpr &ubExpr = pair.second;
      lbounds.push_back(fir::getBase(genExprValue(toEvExpr(lbExpr), stmtCtx)));
      ubounds.push_back(fir::getBase(genExprValue(toEvExpr(ubExpr), stmtCtx)));
    }

    std::optional<Fortran::evaluate::DynamicType> lhsType =
        assign.lhs.GetType();
    std::optional<Fortran::evaluate::DynamicType> rhsType =
        assign.rhs.GetType();
    // Polymorphic lhs/rhs need more care. See F2018 10.2.2.3.
    if ((lhsType && lhsType->IsPolymorphic()) ||
        (rhsType && rhsType->IsPolymorphic())) {
      if (!lowerToHighLevelFIR() && explicitIterationSpace())
        TODO(loc, "polymorphic pointer assignment in FORALL");

      fir::MutableBoxValue lhsMutableBox = genExprMutableBox(loc, assign.lhs);
      if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
              assign.rhs)) {
        fir::factory::disassociateMutableBox(*builder, loc, lhsMutableBox);
        return;
      }
      mlir::Value lhs = lhsMutableBox.getAddr();
      mlir::Value rhs = fir::getBase(genExprBox(loc, assign.rhs, stmtCtx));
      mlir::Value boundsDesc = createBoundArray(lbounds, ubounds, loc);
      Fortran::lower::genPointerAssociateRemapping(*builder, loc, lhs, rhs,
                                                   boundsDesc);
      return;
    }
    if (!lowerToHighLevelFIR() && explicitIterationSpace()) {
      // Pointer assignment in FORALL context. Copy the rhs box value
      // into the lhs box variable.
      genArrayAssignment(assign, stmtCtx, lbounds, ubounds);
      return;
    }
    fir::MutableBoxValue lhs = genExprMutableBox(loc, assign.lhs);
    if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
            assign.rhs)) {
      fir::factory::disassociateMutableBox(*builder, loc, lhs);
      return;
    }
    if (lowerToHighLevelFIR()) {
      fir::ExtendedValue rhs = genExprAddr(assign.rhs, stmtCtx);
      fir::factory::associateMutableBoxWithRemap(*builder, loc, lhs, rhs,
                                                 lbounds, ubounds);
      return;
    }
    // Legacy lowering below.
    // Do not generate a temp in case rhs is an array section.
    fir::ExtendedValue rhs =
        Fortran::lower::isArraySectionWithoutVectorSubscript(assign.rhs)
            ? Fortran::lower::createSomeArrayBox(*this, assign.rhs,
                                                 localSymbols, stmtCtx)
            : genExprAddr(assign.rhs, stmtCtx);
    fir::factory::associateMutableBoxWithRemap(*builder, loc, lhs, rhs, lbounds,
                                               ubounds);
    if (explicitIterationSpace()) {
      mlir::ValueRange inners = explicitIterSpace.getInnerArgs();
      if (!inners.empty())
        builder->create<fir::ResultOp>(loc, inners);
    }
  }

  /// Given converted LHS and RHS of the assignment, materialize any
  /// implicit conversion of the RHS to the LHS type. The front-end
  /// usually already makes those explicit, except for non-standard
  /// LOGICAL <-> INTEGER, or if the LHS is a whole allocatable
  /// (making the conversion explicit in the front-end would prevent
  /// propagation of the LHS lower bound in the reallocation).
  /// If array temporaries or values are created, the cleanups are
  /// added in the statement context.
  hlfir::Entity genImplicitConvert(const Fortran::evaluate::Assignment &assign,
                                   hlfir::Entity rhs, bool preserveLowerBounds,
                                   Fortran::lower::StatementContext &stmtCtx) {
    mlir::Location loc = toLocation();
    auto &builder = getFirOpBuilder();
    mlir::Type toType = genType(assign.lhs);
    auto valueAndPair = hlfir::genTypeAndKindConvert(loc, builder, rhs, toType,
                                                     preserveLowerBounds);
    if (valueAndPair.second)
      stmtCtx.attachCleanup(*valueAndPair.second);
    return hlfir::Entity{valueAndPair.first};
  }

  bool firstDummyIsPointerOrAllocatable(
      const Fortran::evaluate::ProcedureRef &userDefinedAssignment) {
    using DummyAttr = Fortran::evaluate::characteristics::DummyDataObject::Attr;
    if (auto procedure =
            Fortran::evaluate::characteristics::Procedure::Characterize(
                userDefinedAssignment.proc(), getFoldingContext(),
                /*emitError=*/false))
      if (!procedure->dummyArguments.empty())
        if (const auto *dataArg = std::get_if<
                Fortran::evaluate::characteristics::DummyDataObject>(
                &procedure->dummyArguments[0].u))
          return dataArg->attrs.test(DummyAttr::Pointer) ||
                 dataArg->attrs.test(DummyAttr::Allocatable);
    return false;
  }

  void genCUDADataTransfer(fir::FirOpBuilder &builder, mlir::Location loc,
                           const Fortran::evaluate::Assignment &assign,
                           hlfir::Entity &lhs, hlfir::Entity &rhs) {
    bool lhsIsDevice = Fortran::evaluate::HasCUDADeviceAttrs(assign.lhs);
    bool rhsIsDevice = Fortran::evaluate::HasCUDADeviceAttrs(assign.rhs);

    auto getRefFromValue = [](mlir::Value val) -> mlir::Value {
      if (auto loadOp =
              mlir::dyn_cast_or_null<fir::LoadOp>(val.getDefiningOp()))
        return loadOp.getMemref();
      if (!mlir::isa<fir::BaseBoxType>(val.getType()))
        return val;
      if (auto declOp =
              mlir::dyn_cast_or_null<hlfir::DeclareOp>(val.getDefiningOp())) {
        if (!declOp.getShape())
          return val;
        if (mlir::isa<fir::ReferenceType>(declOp.getMemref().getType()))
          return declOp.getResults()[1];
      }
      return val;
    };

    auto getShapeFromDecl = [](mlir::Value val) -> mlir::Value {
      if (!mlir::isa<fir::BaseBoxType>(val.getType()))
        return {};
      if (auto declOp =
              mlir::dyn_cast_or_null<hlfir::DeclareOp>(val.getDefiningOp()))
        return declOp.getShape();
      return {};
    };

    mlir::Value rhsVal = getRefFromValue(rhs.getBase());
    mlir::Value lhsVal = getRefFromValue(lhs.getBase());
    // Get shape from the rhs if available otherwise get it from lhs.
    mlir::Value shape = getShapeFromDecl(rhs.getBase());
    if (!shape)
      shape = getShapeFromDecl(lhs.getBase());

    // device = host
    if (lhsIsDevice && !rhsIsDevice) {
      auto transferKindAttr = cuf::DataTransferKindAttr::get(
          builder.getContext(), cuf::DataTransferKind::HostDevice);
      if (!rhs.isVariable()) {
        mlir::Value base = rhs;
        if (auto convertOp =
                mlir::dyn_cast<fir::ConvertOp>(rhs.getDefiningOp()))
          base = convertOp.getValue();
        // Special case if the rhs is a constant.
        if (matchPattern(base.getDefiningOp(), mlir::m_Constant())) {
          builder.create<cuf::DataTransferOp>(loc, base, lhsVal, shape,
                                              transferKindAttr);
        } else {
          auto associate = hlfir::genAssociateExpr(
              loc, builder, rhs, rhs.getType(), ".cuf_host_tmp");
          builder.create<cuf::DataTransferOp>(loc, associate.getBase(), lhsVal,
                                              shape, transferKindAttr);
          builder.create<hlfir::EndAssociateOp>(loc, associate);
        }
      } else {
        builder.create<cuf::DataTransferOp>(loc, rhsVal, lhsVal, shape,
                                            transferKindAttr);
      }
      return;
    }

    // host = device
    if (!lhsIsDevice && rhsIsDevice) {
      auto transferKindAttr = cuf::DataTransferKindAttr::get(
          builder.getContext(), cuf::DataTransferKind::DeviceHost);
      builder.create<cuf::DataTransferOp>(loc, rhsVal, lhsVal, shape,
                                          transferKindAttr);
      return;
    }

    // device = device
    if (lhsIsDevice && rhsIsDevice) {
      assert(rhs.isVariable() && "CUDA Fortran assignment rhs is not legal");
      auto transferKindAttr = cuf::DataTransferKindAttr::get(
          builder.getContext(), cuf::DataTransferKind::DeviceDevice);
      builder.create<cuf::DataTransferOp>(loc, rhsVal, lhsVal, shape,
                                          transferKindAttr);
      return;
    }
    llvm_unreachable("Unhandled CUDA data transfer");
  }

  llvm::SmallVector<mlir::Value>
  genCUDAImplicitDataTransfer(fir::FirOpBuilder &builder, mlir::Location loc,
                              const Fortran::evaluate::Assignment &assign) {
    llvm::SmallVector<mlir::Value> temps;
    localSymbols.pushScope();
    auto transferKindAttr = cuf::DataTransferKindAttr::get(
        builder.getContext(), cuf::DataTransferKind::DeviceHost);
    [[maybe_unused]] unsigned nbDeviceResidentObject = 0;
    for (const Fortran::semantics::Symbol &sym :
         Fortran::evaluate::CollectSymbols(assign.rhs)) {
      if (const auto *details =
              sym.GetUltimate()
                  .detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
        if (details->cudaDataAttr() &&
            *details->cudaDataAttr() != Fortran::common::CUDADataAttr::Pinned) {
          if (sym.owner().IsDerivedType() && IsAllocatable(sym.GetUltimate()))
            TODO(loc, "Device resident allocatable derived-type component");
          // TODO: This should probably being checked in semantic and give a
          // proper error.
          assert(
              nbDeviceResidentObject <= 1 &&
              "Only one reference to the device resident object is supported");
          auto addr = getSymbolAddress(sym);
          hlfir::Entity entity{addr};
          auto [temp, cleanup] =
              hlfir::createTempFromMold(loc, builder, entity);
          auto needCleanup = fir::getIntIfConstant(cleanup);
          if (needCleanup && *needCleanup) {
            if (auto declareOp =
                    mlir::dyn_cast<hlfir::DeclareOp>(temp.getDefiningOp()))
              temps.push_back(declareOp.getMemref());
            else
              temps.push_back(temp);
          }
          addSymbol(sym,
                    hlfir::translateToExtendedValue(loc, builder, temp).first,
                    /*forced=*/true);
          builder.create<cuf::DataTransferOp>(
              loc, addr, temp, /*shape=*/mlir::Value{}, transferKindAttr);
          ++nbDeviceResidentObject;
        }
      }
    }
    return temps;
  }

  void genDataAssignment(
      const Fortran::evaluate::Assignment &assign,
      const Fortran::evaluate::ProcedureRef *userDefinedAssignment) {
    mlir::Location loc = getCurrentLocation();
    fir::FirOpBuilder &builder = getFirOpBuilder();

    bool isInDeviceContext = Fortran::lower::isCudaDeviceContext(builder);

    bool isCUDATransfer = (Fortran::evaluate::HasCUDADeviceAttrs(assign.lhs) ||
                           Fortran::evaluate::HasCUDADeviceAttrs(assign.rhs)) &&
                          !isInDeviceContext;
    bool hasCUDAImplicitTransfer =
        Fortran::evaluate::HasCUDAImplicitTransfer(assign.rhs);
    llvm::SmallVector<mlir::Value> implicitTemps;
    if (hasCUDAImplicitTransfer && !isInDeviceContext)
      implicitTemps = genCUDAImplicitDataTransfer(builder, loc, assign);

    // Gather some information about the assignment that will impact how it is
    // lowered.
    const bool isWholeAllocatableAssignment =
        !userDefinedAssignment && !isInsideHlfirWhere() &&
        Fortran::lower::isWholeAllocatable(assign.lhs);
    const bool isUserDefAssignToPointerOrAllocatable =
        userDefinedAssignment &&
        firstDummyIsPointerOrAllocatable(*userDefinedAssignment);
    std::optional<Fortran::evaluate::DynamicType> lhsType =
        assign.lhs.GetType();
    const bool keepLhsLengthInAllocatableAssignment =
        isWholeAllocatableAssignment && lhsType.has_value() &&
        lhsType->category() == Fortran::common::TypeCategory::Character &&
        !lhsType->HasDeferredTypeParameter();
    const bool lhsHasVectorSubscripts =
        Fortran::evaluate::HasVectorSubscript(assign.lhs);

    // Helper to generate the code evaluating the right-hand side.
    auto evaluateRhs = [&](Fortran::lower::StatementContext &stmtCtx) {
      hlfir::Entity rhs = Fortran::lower::convertExprToHLFIR(
          loc, *this, assign.rhs, localSymbols, stmtCtx);
      // Load trivial scalar RHS to allow the loads to be hoisted outside of
      // loops early if possible. This also dereferences pointer and
      // allocatable RHS: the target is being assigned from.
      rhs = hlfir::loadTrivialScalar(loc, builder, rhs);
      // In intrinsic assignments, the LHS type may not match the RHS type, in
      // which case an implicit conversion of the LHS must be done. The
      // front-end usually makes it explicit, unless it cannot (whole
      // allocatable LHS or Logical<->Integer assignment extension). Recognize
      // any type mismatches here and insert explicit scalar convert or
      // ElementalOp for array assignment. Preserve the RHS lower bounds on the
      // converted entity in case of assignment to whole allocatables so to
      // propagate the lower bounds to the LHS in case of reallocation.
      if (!userDefinedAssignment)
        rhs = genImplicitConvert(assign, rhs, isWholeAllocatableAssignment,
                                 stmtCtx);
      return rhs;
    };

    // Helper to generate the code evaluating the left-hand side.
    auto evaluateLhs = [&](Fortran::lower::StatementContext &stmtCtx) {
      hlfir::Entity lhs = Fortran::lower::convertExprToHLFIR(
          loc, *this, assign.lhs, localSymbols, stmtCtx);
      // Dereference pointer LHS: the target is being assigned to.
      // Same for allocatables outside of whole allocatable assignments.
      if (!isWholeAllocatableAssignment &&
          !isUserDefAssignToPointerOrAllocatable)
        lhs = hlfir::derefPointersAndAllocatables(loc, builder, lhs);
      return lhs;
    };

    if (!isInsideHlfirForallOrWhere() && !lhsHasVectorSubscripts &&
        !userDefinedAssignment) {
      Fortran::lower::StatementContext localStmtCtx;
      hlfir::Entity rhs = evaluateRhs(localStmtCtx);
      hlfir::Entity lhs = evaluateLhs(localStmtCtx);
      if (isCUDATransfer && !hasCUDAImplicitTransfer)
        genCUDADataTransfer(builder, loc, assign, lhs, rhs);
      else
        builder.create<hlfir::AssignOp>(loc, rhs, lhs,
                                        isWholeAllocatableAssignment,
                                        keepLhsLengthInAllocatableAssignment);
      if (hasCUDAImplicitTransfer && !isInDeviceContext) {
        localSymbols.popScope();
        for (mlir::Value temp : implicitTemps)
          builder.create<fir::FreeMemOp>(loc, temp);
      }
      return;
    }
    // Assignments inside Forall, Where, or assignments to a vector subscripted
    // left-hand side requires using an hlfir.region_assign in HLFIR. The
    // right-hand side and left-hand side must be evaluated inside the
    // hlfir.region_assign regions.
    auto regionAssignOp = builder.create<hlfir::RegionAssignOp>(loc);

    // Lower RHS in its own region.
    builder.createBlock(&regionAssignOp.getRhsRegion());
    Fortran::lower::StatementContext rhsContext;
    hlfir::Entity rhs = evaluateRhs(rhsContext);
    auto rhsYieldOp = builder.create<hlfir::YieldOp>(loc, rhs);
    Fortran::lower::genCleanUpInRegionIfAny(
        loc, builder, rhsYieldOp.getCleanup(), rhsContext);
    // Lower LHS in its own region.
    builder.createBlock(&regionAssignOp.getLhsRegion());
    Fortran::lower::StatementContext lhsContext;
    mlir::Value lhsYield = nullptr;
    if (!lhsHasVectorSubscripts) {
      hlfir::Entity lhs = evaluateLhs(lhsContext);
      auto lhsYieldOp = builder.create<hlfir::YieldOp>(loc, lhs);
      Fortran::lower::genCleanUpInRegionIfAny(
          loc, builder, lhsYieldOp.getCleanup(), lhsContext);
      lhsYield = lhs;
    } else {
      hlfir::ElementalAddrOp elementalAddr =
          Fortran::lower::convertVectorSubscriptedExprToElementalAddr(
              loc, *this, assign.lhs, localSymbols, lhsContext);
      Fortran::lower::genCleanUpInRegionIfAny(
          loc, builder, elementalAddr.getCleanup(), lhsContext);
      lhsYield = elementalAddr.getYieldOp().getEntity();
    }
    assert(lhsYield && "must have been set");

    // Add "realloc" flag to hlfir.region_assign.
    if (isWholeAllocatableAssignment)
      TODO(loc, "assignment to a whole allocatable inside FORALL");

    // Generate the hlfir.region_assign userDefinedAssignment region.
    if (userDefinedAssignment) {
      mlir::Type rhsType = rhs.getType();
      mlir::Type lhsType = lhsYield.getType();
      if (userDefinedAssignment->IsElemental()) {
        rhsType = hlfir::getEntityElementType(rhs);
        lhsType = hlfir::getEntityElementType(hlfir::Entity{lhsYield});
      }
      builder.createBlock(&regionAssignOp.getUserDefinedAssignment(),
                          mlir::Region::iterator{}, {rhsType, lhsType},
                          {loc, loc});
      auto end = builder.create<fir::FirEndOp>(loc);
      builder.setInsertionPoint(end);
      hlfir::Entity lhsBlockArg{regionAssignOp.getUserAssignmentLhs()};
      hlfir::Entity rhsBlockArg{regionAssignOp.getUserAssignmentRhs()};
      Fortran::lower::convertUserDefinedAssignmentToHLFIR(
          loc, *this, *userDefinedAssignment, lhsBlockArg, rhsBlockArg,
          localSymbols);
    }
    builder.setInsertionPointAfter(regionAssignOp);
  }

  /// Shared for both assignments and pointer assignments.
  void genAssignment(const Fortran::evaluate::Assignment &assign) {
    mlir::Location loc = toLocation();
    if (lowerToHighLevelFIR()) {
      Fortran::common::visit(
          Fortran::common::visitors{
              [&](const Fortran::evaluate::Assignment::Intrinsic &) {
                genDataAssignment(assign, /*userDefinedAssignment=*/nullptr);
              },
              [&](const Fortran::evaluate::ProcedureRef &procRef) {
                genDataAssignment(assign, /*userDefinedAssignment=*/&procRef);
              },
              [&](const Fortran::evaluate::Assignment::BoundsSpec &lbExprs) {
                if (isInsideHlfirForallOrWhere())
                  TODO(loc, "pointer assignment inside FORALL");
                genPointerAssignment(loc, assign, lbExprs);
              },
              [&](const Fortran::evaluate::Assignment::BoundsRemapping
                      &boundExprs) {
                if (isInsideHlfirForallOrWhere())
                  TODO(loc, "pointer assignment inside FORALL");
                genPointerAssignment(loc, assign, boundExprs);
              },
          },
          assign.u);
      return;
    }
    if (explicitIterationSpace()) {
      Fortran::lower::createArrayLoads(*this, explicitIterSpace, localSymbols);
      explicitIterSpace.genLoopNest();
    }
    Fortran::lower::StatementContext stmtCtx;
    Fortran::common::visit(
        Fortran::common::visitors{
            // [1] Plain old assignment.
            [&](const Fortran::evaluate::Assignment::Intrinsic &) {
              const Fortran::semantics::Symbol *sym =
                  Fortran::evaluate::GetLastSymbol(assign.lhs);

              if (!sym)
                TODO(loc, "assignment to pointer result of function reference");

              std::optional<Fortran::evaluate::DynamicType> lhsType =
                  assign.lhs.GetType();
              assert(lhsType && "lhs cannot be typeless");
              std::optional<Fortran::evaluate::DynamicType> rhsType =
                  assign.rhs.GetType();

              // Assignment to/from polymorphic entities are done with the
              // runtime.
              if (lhsType->IsPolymorphic() ||
                  lhsType->IsUnlimitedPolymorphic() ||
                  (rhsType && (rhsType->IsPolymorphic() ||
                               rhsType->IsUnlimitedPolymorphic()))) {
                mlir::Value lhs;
                if (Fortran::lower::isWholeAllocatable(assign.lhs))
                  lhs = genExprMutableBox(loc, assign.lhs).getAddr();
                else
                  lhs = fir::getBase(genExprBox(loc, assign.lhs, stmtCtx));
                mlir::Value rhs =
                    fir::getBase(genExprBox(loc, assign.rhs, stmtCtx));
                if ((lhsType->IsPolymorphic() ||
                     lhsType->IsUnlimitedPolymorphic()) &&
                    Fortran::lower::isWholeAllocatable(assign.lhs))
                  fir::runtime::genAssignPolymorphic(*builder, loc, lhs, rhs);
                else
                  fir::runtime::genAssign(*builder, loc, lhs, rhs);
                return;
              }

              // Note: No ad-hoc handling for pointers is required here. The
              // target will be assigned as per 2018 10.2.1.3 p2. genExprAddr
              // on a pointer returns the target address and not the address of
              // the pointer variable.

              if (assign.lhs.Rank() > 0 || explicitIterationSpace()) {
                if (isDerivedCategory(lhsType->category()) &&
                    Fortran::semantics::IsFinalizable(
                        lhsType->GetDerivedTypeSpec()))
                  TODO(loc, "derived-type finalization with array assignment");
                // Array assignment
                // See Fortran 2018 10.2.1.3 p5, p6, and p7
                genArrayAssignment(assign, stmtCtx);
                return;
              }

              // Scalar assignment
              const bool isNumericScalar =
                  isNumericScalarCategory(lhsType->category());
              const bool isVector =
                  isDerivedCategory(lhsType->category()) &&
                  lhsType->GetDerivedTypeSpec().IsVectorType();
              fir::ExtendedValue rhs = (isNumericScalar || isVector)
                                           ? genExprValue(assign.rhs, stmtCtx)
                                           : genExprAddr(assign.rhs, stmtCtx);
              const bool lhsIsWholeAllocatable =
                  Fortran::lower::isWholeAllocatable(assign.lhs);
              std::optional<fir::factory::MutableBoxReallocation> lhsRealloc;
              std::optional<fir::MutableBoxValue> lhsMutableBox;

              // Set flag to know if the LHS needs finalization. Polymorphic,
              // unlimited polymorphic assignment will be done with genAssign.
              // Assign runtime function performs the finalization.
              bool needFinalization = !lhsType->IsPolymorphic() &&
                                      !lhsType->IsUnlimitedPolymorphic() &&
                                      (isDerivedCategory(lhsType->category()) &&
                                       Fortran::semantics::IsFinalizable(
                                           lhsType->GetDerivedTypeSpec()));

              auto lhs = [&]() -> fir::ExtendedValue {
                if (lhsIsWholeAllocatable) {
                  lhsMutableBox = genExprMutableBox(loc, assign.lhs);
                  // Finalize if needed.
                  if (needFinalization) {
                    mlir::Value isAllocated =
                        fir::factory::genIsAllocatedOrAssociatedTest(
                            *builder, loc, *lhsMutableBox);
                    builder->genIfThen(loc, isAllocated)
                        .genThen([&]() {
                          fir::runtime::genDerivedTypeDestroy(
                              *builder, loc, fir::getBase(*lhsMutableBox));
                        })
                        .end();
                    needFinalization = false;
                  }

                  llvm::SmallVector<mlir::Value> lengthParams;
                  if (const fir::CharBoxValue *charBox = rhs.getCharBox())
                    lengthParams.push_back(charBox->getLen());
                  else if (fir::isDerivedWithLenParameters(rhs))
                    TODO(loc, "assignment to derived type allocatable with "
                              "LEN parameters");
                  lhsRealloc = fir::factory::genReallocIfNeeded(
                      *builder, loc, *lhsMutableBox,
                      /*shape=*/std::nullopt, lengthParams);
                  return lhsRealloc->newValue;
                }
                return genExprAddr(assign.lhs, stmtCtx);
              }();

              if (isNumericScalar || isVector) {
                // Fortran 2018 10.2.1.3 p8 and p9
                // Conversions should have been inserted by semantic analysis,
                // but they can be incorrect between the rhs and lhs. Correct
                // that here.
                mlir::Value addr = fir::getBase(lhs);
                mlir::Value val = fir::getBase(rhs);
                // A function with multiple entry points returning different
                // types tags all result variables with one of the largest
                // types to allow them to share the same storage. Assignment
                // to a result variable of one of the other types requires
                // conversion to the actual type.
                mlir::Type toTy = genType(assign.lhs);

                // If Cray pointee, need to handle the address
                // Array is handled in genCoordinateOp.
                if (sym->test(Fortran::semantics::Symbol::Flag::CrayPointee) &&
                    sym->Rank() == 0) {
                  // get the corresponding Cray pointer

                  const Fortran::semantics::Symbol &ptrSym =
                      Fortran::semantics::GetCrayPointer(*sym);
                  fir::ExtendedValue ptr =
                      getSymbolExtendedValue(ptrSym, nullptr);
                  mlir::Value ptrVal = fir::getBase(ptr);
                  mlir::Type ptrTy = genType(ptrSym);

                  fir::ExtendedValue pte =
                      getSymbolExtendedValue(*sym, nullptr);
                  mlir::Value pteVal = fir::getBase(pte);
                  mlir::Value cnvrt = Fortran::lower::addCrayPointerInst(
                      loc, *builder, ptrVal, ptrTy, pteVal.getType());
                  addr = builder->create<fir::LoadOp>(loc, cnvrt);
                }
                mlir::Value cast =
                    isVector ? val
                             : builder->convertWithSemantics(loc, toTy, val);
                if (fir::dyn_cast_ptrEleTy(addr.getType()) != toTy) {
                  assert(isFuncResultDesignator(assign.lhs) && "type mismatch");
                  addr = builder->createConvert(
                      toLocation(), builder->getRefType(toTy), addr);
                }
                builder->create<fir::StoreOp>(loc, cast, addr);
              } else if (isCharacterCategory(lhsType->category())) {
                // Fortran 2018 10.2.1.3 p10 and p11
                fir::factory::CharacterExprHelper{*builder, loc}.createAssign(
                    lhs, rhs);
              } else if (isDerivedCategory(lhsType->category())) {
                // Handle parent component.
                if (Fortran::lower::isParentComponent(assign.lhs)) {
                  if (!mlir::isa<fir::BaseBoxType>(fir::getBase(lhs).getType()))
                    lhs = fir::getBase(builder->createBox(loc, lhs));
                  lhs = Fortran::lower::updateBoxForParentComponent(*this, lhs,
                                                                    assign.lhs);
                }

                // Fortran 2018 10.2.1.3 p13 and p14
                // Recursively gen an assignment on each element pair.
                fir::factory::genRecordAssignment(*builder, loc, lhs, rhs,
                                                  needFinalization);
              } else {
                llvm_unreachable("unknown category");
              }
              if (lhsIsWholeAllocatable) {
                assert(lhsRealloc.has_value());
                fir::factory::finalizeRealloc(*builder, loc, *lhsMutableBox,
                                              /*lbounds=*/std::nullopt,
                                              /*takeLboundsIfRealloc=*/false,
                                              *lhsRealloc);
              }
            },

            // [2] User defined assignment. If the context is a scalar
            // expression then call the procedure.
            [&](const Fortran::evaluate::ProcedureRef &procRef) {
              Fortran::lower::StatementContext &ctx =
                  explicitIterationSpace() ? explicitIterSpace.stmtContext()
                                           : stmtCtx;
              Fortran::lower::createSubroutineCall(
                  *this, procRef, explicitIterSpace, implicitIterSpace,
                  localSymbols, ctx, /*isUserDefAssignment=*/true);
            },

            [&](const Fortran::evaluate::Assignment::BoundsSpec &lbExprs) {
              return genPointerAssignment(loc, assign, lbExprs);
            },
            [&](const Fortran::evaluate::Assignment::BoundsRemapping
                    &boundExprs) {
              return genPointerAssignment(loc, assign, boundExprs);
            },
        },
        assign.u);
    if (explicitIterationSpace())
      Fortran::lower::createArrayMergeStores(*this, explicitIterSpace);
  }

  // Is the insertion point of the builder directly or indirectly set
  // inside any operation of type "Op"?
  template <typename... Op>
  bool isInsideOp() const {
    mlir::Block *block = builder->getInsertionBlock();
    mlir::Operation *op = block ? block->getParentOp() : nullptr;
    while (op) {
      if (mlir::isa<Op...>(op))
        return true;
      op = op->getParentOp();
    }
    return false;
  }
  bool isInsideHlfirForallOrWhere() const {
    return isInsideOp<hlfir::ForallOp, hlfir::WhereOp>();
  }
  bool isInsideHlfirWhere() const { return isInsideOp<hlfir::WhereOp>(); }

  void genFIR(const Fortran::parser::WhereConstruct &c) {
    mlir::Location loc = getCurrentLocation();
    hlfir::WhereOp whereOp;

    if (!lowerToHighLevelFIR()) {
      implicitIterSpace.growStack();
    } else {
      whereOp = builder->create<hlfir::WhereOp>(loc);
      builder->createBlock(&whereOp.getMaskRegion());
    }

    // Lower the where mask. For HLFIR, this is done in the hlfir.where mask
    // region.
    genNestedStatement(
        std::get<
            Fortran::parser::Statement<Fortran::parser::WhereConstructStmt>>(
            c.t));

    // Lower WHERE body. For HLFIR, this is done in the hlfir.where body
    // region.
    if (whereOp)
      builder->createBlock(&whereOp.getBody());

    for (const auto &body :
         std::get<std::list<Fortran::parser::WhereBodyConstruct>>(c.t))
      genFIR(body);
    for (const auto &e :
         std::get<std::list<Fortran::parser::WhereConstruct::MaskedElsewhere>>(
             c.t))
      genFIR(e);
    if (const auto &e =
            std::get<std::optional<Fortran::parser::WhereConstruct::Elsewhere>>(
                c.t);
        e.has_value())
      genFIR(*e);
    genNestedStatement(
        std::get<Fortran::parser::Statement<Fortran::parser::EndWhereStmt>>(
            c.t));

    if (whereOp) {
      // For HLFIR, create fir.end terminator in the last hlfir.elsewhere, or
      // in the hlfir.where if it had no elsewhere.
      builder->create<fir::FirEndOp>(loc);
      builder->setInsertionPointAfter(whereOp);
    }
  }
  void genFIR(const Fortran::parser::WhereBodyConstruct &body) {
    Fortran::common::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Statement<
                Fortran::parser::AssignmentStmt> &stmt) {
              genNestedStatement(stmt);
            },
            [&](const Fortran::parser::Statement<Fortran::parser::WhereStmt>
                    &stmt) { genNestedStatement(stmt); },
            [&](const Fortran::common::Indirection<
                Fortran::parser::WhereConstruct> &c) { genFIR(c.value()); },
        },
        body.u);
  }

  /// Lower a Where or Elsewhere mask into an hlfir mask region.
  void lowerWhereMaskToHlfir(mlir::Location loc,
                             const Fortran::semantics::SomeExpr *maskExpr) {
    assert(maskExpr && "mask semantic analysis failed");
    Fortran::lower::StatementContext maskContext;
    hlfir::Entity mask = Fortran::lower::convertExprToHLFIR(
        loc, *this, *maskExpr, localSymbols, maskContext);
    mask = hlfir::loadTrivialScalar(loc, *builder, mask);
    auto yieldOp = builder->create<hlfir::YieldOp>(loc, mask);
    Fortran::lower::genCleanUpInRegionIfAny(loc, *builder, yieldOp.getCleanup(),
                                            maskContext);
  }
  void genFIR(const Fortran::parser::WhereConstructStmt &stmt) {
    const Fortran::semantics::SomeExpr *maskExpr = Fortran::semantics::GetExpr(
        std::get<Fortran::parser::LogicalExpr>(stmt.t));
    if (lowerToHighLevelFIR())
      lowerWhereMaskToHlfir(getCurrentLocation(), maskExpr);
    else
      implicitIterSpace.append(maskExpr);
  }
  void genFIR(const Fortran::parser::WhereConstruct::MaskedElsewhere &ew) {
    mlir::Location loc = getCurrentLocation();
    hlfir::ElseWhereOp elsewhereOp;
    if (lowerToHighLevelFIR()) {
      elsewhereOp = builder->create<hlfir::ElseWhereOp>(loc);
      // Lower mask in the mask region.
      builder->createBlock(&elsewhereOp.getMaskRegion());
    }
    genNestedStatement(
        std::get<
            Fortran::parser::Statement<Fortran::parser::MaskedElsewhereStmt>>(
            ew.t));

    // For HLFIR, lower the body in the hlfir.elsewhere body region.
    if (elsewhereOp)
      builder->createBlock(&elsewhereOp.getBody());

    for (const auto &body :
         std::get<std::list<Fortran::parser::WhereBodyConstruct>>(ew.t))
      genFIR(body);
  }
  void genFIR(const Fortran::parser::MaskedElsewhereStmt &stmt) {
    const auto *maskExpr = Fortran::semantics::GetExpr(
        std::get<Fortran::parser::LogicalExpr>(stmt.t));
    if (lowerToHighLevelFIR())
      lowerWhereMaskToHlfir(getCurrentLocation(), maskExpr);
    else
      implicitIterSpace.append(maskExpr);
  }
  void genFIR(const Fortran::parser::WhereConstruct::Elsewhere &ew) {
    if (lowerToHighLevelFIR()) {
      auto elsewhereOp =
          builder->create<hlfir::ElseWhereOp>(getCurrentLocation());
      builder->createBlock(&elsewhereOp.getBody());
    }
    genNestedStatement(
        std::get<Fortran::parser::Statement<Fortran::parser::ElsewhereStmt>>(
            ew.t));
    for (const auto &body :
         std::get<std::list<Fortran::parser::WhereBodyConstruct>>(ew.t))
      genFIR(body);
  }
  void genFIR(const Fortran::parser::ElsewhereStmt &stmt) {
    if (!lowerToHighLevelFIR())
      implicitIterSpace.append(nullptr);
  }
  void genFIR(const Fortran::parser::EndWhereStmt &) {
    if (!lowerToHighLevelFIR())
      implicitIterSpace.shrinkStack();
  }

  void genFIR(const Fortran::parser::WhereStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    const auto &assign = std::get<Fortran::parser::AssignmentStmt>(stmt.t);
    const auto *mask = Fortran::semantics::GetExpr(
        std::get<Fortran::parser::LogicalExpr>(stmt.t));
    if (lowerToHighLevelFIR()) {
      mlir::Location loc = getCurrentLocation();
      auto whereOp = builder->create<hlfir::WhereOp>(loc);
      builder->createBlock(&whereOp.getMaskRegion());
      lowerWhereMaskToHlfir(loc, mask);
      builder->createBlock(&whereOp.getBody());
      genAssignment(*assign.typedAssignment->v);
      builder->create<fir::FirEndOp>(loc);
      builder->setInsertionPointAfter(whereOp);
      return;
    }
    implicitIterSpace.growStack();
    implicitIterSpace.append(mask);
    genAssignment(*assign.typedAssignment->v);
    implicitIterSpace.shrinkStack();
  }

  void genFIR(const Fortran::parser::PointerAssignmentStmt &stmt) {
    genAssignment(*stmt.typedAssignment->v);
  }

  void genFIR(const Fortran::parser::AssignmentStmt &stmt) {
    genAssignment(*stmt.typedAssignment->v);
  }

  void genFIR(const Fortran::parser::SyncAllStmt &stmt) {
    genSyncAllStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::SyncImagesStmt &stmt) {
    genSyncImagesStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::SyncMemoryStmt &stmt) {
    genSyncMemoryStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::SyncTeamStmt &stmt) {
    genSyncTeamStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::UnlockStmt &stmt) {
    genUnlockStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::AssignStmt &stmt) {
    const Fortran::semantics::Symbol &symbol =
        *std::get<Fortran::parser::Name>(stmt.t).symbol;
    mlir::Location loc = toLocation();
    mlir::Value labelValue = builder->createIntegerConstant(
        loc, genType(symbol), std::get<Fortran::parser::Label>(stmt.t));
    builder->create<fir::StoreOp>(loc, labelValue, getSymbolAddress(symbol));
  }

  void genFIR(const Fortran::parser::FormatStmt &) {
    // do nothing.

    // FORMAT statements have no semantics. They may be lowered if used by a
    // data transfer statement.
  }

  void genFIR(const Fortran::parser::PauseStmt &stmt) {
    genPauseStatement(*this, stmt);
  }

  // call FAIL IMAGE in runtime
  void genFIR(const Fortran::parser::FailImageStmt &stmt) {
    genFailImageStatement(*this);
  }

  // call STOP, ERROR STOP in runtime
  void genFIR(const Fortran::parser::StopStmt &stmt) {
    genStopStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::ReturnStmt &stmt) {
    Fortran::lower::pft::FunctionLikeUnit *funit =
        getEval().getOwningProcedure();
    assert(funit && "not inside main program, function or subroutine");
    for (auto it = activeConstructStack.rbegin(),
              rend = activeConstructStack.rend();
         it != rend; ++it) {
      it->stmtCtx.finalizeAndKeep();
    }
    if (funit->isMainProgram()) {
      bridge.fctCtx().finalizeAndKeep();
      genExitRoutine();
      return;
    }
    mlir::Location loc = toLocation();
    if (stmt.v) {
      // Alternate return statement - If this is a subroutine where some
      // alternate entries have alternate returns, but the active entry point
      // does not, ignore the alternate return value. Otherwise, assign it
      // to the compiler-generated result variable.
      const Fortran::semantics::Symbol &symbol = funit->getSubprogramSymbol();
      if (Fortran::semantics::HasAlternateReturns(symbol)) {
        Fortran::lower::StatementContext stmtCtx;
        const Fortran::lower::SomeExpr *expr =
            Fortran::semantics::GetExpr(*stmt.v);
        assert(expr && "missing alternate return expression");
        mlir::Value altReturnIndex = builder->createConvert(
            loc, builder->getIndexType(), createFIRExpr(loc, expr, stmtCtx));
        builder->create<fir::StoreOp>(loc, altReturnIndex,
                                      getAltReturnResult(symbol));
      }
    }
    // Branch to the last block of the SUBROUTINE, which has the actual return.
    if (!funit->finalBlock) {
      mlir::OpBuilder::InsertPoint insPt = builder->saveInsertionPoint();
      Fortran::lower::setInsertionPointAfterOpenACCLoopIfInside(*builder);
      funit->finalBlock = builder->createBlock(&builder->getRegion());
      builder->restoreInsertionPoint(insPt);
    }

    if (Fortran::lower::isInOpenACCLoop(*builder))
      Fortran::lower::genEarlyReturnInOpenACCLoop(*builder, loc);
    else
      builder->create<mlir::cf::BranchOp>(loc, funit->finalBlock);
  }

  void genFIR(const Fortran::parser::CycleStmt &) {
    genConstructExitBranch(*getEval().controlSuccessor);
  }
  void genFIR(const Fortran::parser::ExitStmt &) {
    genConstructExitBranch(*getEval().controlSuccessor);
  }
  void genFIR(const Fortran::parser::GotoStmt &) {
    genConstructExitBranch(*getEval().controlSuccessor);
  }

  // Nop statements - No code, or code is generated at the construct level.
  // But note that the genFIR call immediately below that wraps one of these
  // calls does block management, possibly starting a new block, and possibly
  // generating a branch to end a block. So these calls may still be required
  // for that functionality.
  void genFIR(const Fortran::parser::AssociateStmt &) {}       // nop
  void genFIR(const Fortran::parser::BlockStmt &) {}           // nop
  void genFIR(const Fortran::parser::CaseStmt &) {}            // nop
  void genFIR(const Fortran::parser::ContinueStmt &) {}        // nop
  void genFIR(const Fortran::parser::ElseIfStmt &) {}          // nop
  void genFIR(const Fortran::parser::ElseStmt &) {}            // nop
  void genFIR(const Fortran::parser::EndAssociateStmt &) {}    // nop
  void genFIR(const Fortran::parser::EndBlockStmt &) {}        // nop
  void genFIR(const Fortran::parser::EndDoStmt &) {}           // nop
  void genFIR(const Fortran::parser::EndFunctionStmt &) {}     // nop
  void genFIR(const Fortran::parser::EndIfStmt &) {}           // nop
  void genFIR(const Fortran::parser::EndMpSubprogramStmt &) {} // nop
  void genFIR(const Fortran::parser::EndProgramStmt &) {}      // nop
  void genFIR(const Fortran::parser::EndSelectStmt &) {}       // nop
  void genFIR(const Fortran::parser::EndSubroutineStmt &) {}   // nop
  void genFIR(const Fortran::parser::EntryStmt &) {}           // nop
  void genFIR(const Fortran::parser::IfStmt &) {}              // nop
  void genFIR(const Fortran::parser::IfThenStmt &) {}          // nop
  void genFIR(const Fortran::parser::NonLabelDoStmt &) {}      // nop
  void genFIR(const Fortran::parser::OmpEndLoopDirective &) {} // nop
  void genFIR(const Fortran::parser::SelectTypeStmt &) {}      // nop
  void genFIR(const Fortran::parser::TypeGuardStmt &) {}       // nop

  /// Generate FIR for Evaluation \p eval.
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              bool unstructuredContext = true) {
    // Start a new unstructured block when applicable. When transitioning
    // from unstructured to structured code, unstructuredContext is true,
    // which accounts for the possibility that the structured code could be
    // a target that starts a new block.
    if (unstructuredContext)
      maybeStartBlock(eval.isConstruct() && eval.lowerAsStructured()
                          ? eval.getFirstNestedEvaluation().block
                          : eval.block);

    // Generate evaluation specific code. Even nop calls should usually reach
    // here in case they start a new block or require generation of a generic
    // end-of-block branch. An alternative is to add special case code
    // elsewhere, such as in the genFIR code for a parent construct.
    setCurrentEval(eval);
    setCurrentPosition(eval.position);
    eval.visit([&](const auto &stmt) { genFIR(stmt); });
  }

  /// Map mlir function block arguments to the corresponding Fortran dummy
  /// variables. When the result is passed as a hidden argument, the Fortran
  /// result is also mapped. The symbol map is used to hold this mapping.
  void mapDummiesAndResults(Fortran::lower::pft::FunctionLikeUnit &funit,
                            const Fortran::lower::CalleeInterface &callee) {
    assert(builder && "require a builder object at this point");
    using PassBy = Fortran::lower::CalleeInterface::PassEntityBy;
    auto mapPassedEntity = [&](const auto arg, bool isResult = false) {
      if (arg.passBy == PassBy::AddressAndLength) {
        if (callee.characterize().IsBindC())
          return;
        // TODO: now that fir call has some attributes regarding character
        // return, PassBy::AddressAndLength should be retired.
        mlir::Location loc = toLocation();
        fir::factory::CharacterExprHelper charHelp{*builder, loc};
        mlir::Value box =
            charHelp.createEmboxChar(arg.firArgument, arg.firLength);
        mapBlockArgToDummyOrResult(arg.entity->get(), box, isResult);
      } else {
        if (arg.entity.has_value()) {
          mapBlockArgToDummyOrResult(arg.entity->get(), arg.firArgument,
                                     isResult);
        } else {
          assert(funit.parentHasTupleHostAssoc() && "expect tuple argument");
        }
      }
    };
    for (const Fortran::lower::CalleeInterface::PassedEntity &arg :
         callee.getPassedArguments())
      mapPassedEntity(arg);
    if (lowerToHighLevelFIR() && !callee.getPassedArguments().empty()) {
      mlir::Value scopeOp = builder->create<fir::DummyScopeOp>(toLocation());
      setDummyArgsScope(scopeOp);
    }
    if (std::optional<Fortran::lower::CalleeInterface::PassedEntity>
            passedResult = callee.getPassedResult()) {
      mapPassedEntity(*passedResult, /*isResult=*/true);
      // FIXME: need to make sure things are OK here. addSymbol may not be OK
      if (funit.primaryResult &&
          passedResult->entity->get() != *funit.primaryResult)
        mapBlockArgToDummyOrResult(
            *funit.primaryResult, getSymbolAddress(passedResult->entity->get()),
            /*isResult=*/true);
    }
  }

  /// Instantiate variable \p var and add it to the symbol map.
  /// See ConvertVariable.cpp.
  void instantiateVar(const Fortran::lower::pft::Variable &var,
                      Fortran::lower::AggregateStoreMap &storeMap) {
    Fortran::lower::instantiateVariable(*this, var, localSymbols, storeMap);
    if (var.hasSymbol())
      genOpenMPSymbolProperties(*this, var);
  }

  /// Where applicable, save the exception state and halting and rounding
  /// modes at function entry and restore them at function exits.
  void manageFPEnvironment(Fortran::lower::pft::FunctionLikeUnit &funit) {
    mlir::Location loc = toLocation();
    mlir::Location endLoc =
        toLocation(Fortran::lower::pft::stmtSourceLoc(funit.endStmt));
    if (funit.hasIeeeAccess) {
      // Subject to F18 Clause 17.1p3, 17.3p3 states: If a flag is signaling
      // on entry to a procedure [...], the processor will set it to quiet
      // on entry and restore it to signaling on return. If a flag signals
      // during execution of a procedure, the processor shall not set it to
      // quiet on return.
      mlir::func::FuncOp testExcept = fir::factory::getFetestexcept(*builder);
      mlir::func::FuncOp clearExcept = fir::factory::getFeclearexcept(*builder);
      mlir::func::FuncOp raiseExcept = fir::factory::getFeraiseexcept(*builder);
      mlir::Value ones = builder->createIntegerConstant(
          loc, testExcept.getFunctionType().getInput(0), -1);
      mlir::Value exceptSet =
          builder->create<fir::CallOp>(loc, testExcept, ones).getResult(0);
      builder->create<fir::CallOp>(loc, clearExcept, exceptSet);
      bridge.fctCtx().attachCleanup([=]() {
        builder->create<fir::CallOp>(endLoc, raiseExcept, exceptSet);
      });
    }
    if (funit.mayModifyHaltingMode) {
      // F18 Clause 17.6p1: In a procedure [...], the processor shall not
      // change the halting mode on entry, and on return shall ensure that
      // the halting mode is the same as it was on entry.
      mlir::func::FuncOp getExcept = fir::factory::getFegetexcept(*builder);
      mlir::func::FuncOp disableExcept =
          fir::factory::getFedisableexcept(*builder);
      mlir::func::FuncOp enableExcept =
          fir::factory::getFeenableexcept(*builder);
      mlir::Value exceptSet =
          builder->create<fir::CallOp>(loc, getExcept).getResult(0);
      mlir::Value ones = builder->createIntegerConstant(
          loc, disableExcept.getFunctionType().getInput(0), -1);
      bridge.fctCtx().attachCleanup([=]() {
        builder->create<fir::CallOp>(endLoc, disableExcept, ones);
        builder->create<fir::CallOp>(endLoc, enableExcept, exceptSet);
      });
    }
    if (funit.mayModifyRoundingMode) {
      // F18 Clause 17.4.5: In a procedure [...], the processor shall not
      // change the rounding modes on entry, and on return shall ensure that
      // the rounding modes are the same as they were on entry.
      mlir::func::FuncOp getRounding =
          fir::factory::getLlvmGetRounding(*builder);
      mlir::func::FuncOp setRounding =
          fir::factory::getLlvmSetRounding(*builder);
      mlir::Value roundingMode =
          builder->create<fir::CallOp>(loc, getRounding).getResult(0);
      bridge.fctCtx().attachCleanup([=]() {
        builder->create<fir::CallOp>(endLoc, setRounding, roundingMode);
      });
    }
  }

  /// Start translation of a function.
  void startNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    assert(!builder && "expected nullptr");
    bridge.fctCtx().pushScope();
    bridge.openAccCtx().pushScope();
    const Fortran::semantics::Scope &scope = funit.getScope();
    LLVM_DEBUG(llvm::dbgs() << "\n[bridge - startNewFunction]";
               if (auto *sym = scope.symbol()) llvm::dbgs() << " " << *sym;
               llvm::dbgs() << "\n");
    Fortran::lower::CalleeInterface callee(funit, *this);
    mlir::func::FuncOp func = callee.addEntryBlockAndMapArguments();
    builder =
        new fir::FirOpBuilder(func, bridge.getKindMap(), &mlirSymbolTable);
    assert(builder && "FirOpBuilder did not instantiate");
    builder->setFastMathFlags(bridge.getLoweringOptions().getMathOptions());
    builder->setInsertionPointToStart(&func.front());
    if (funit.parent.isA<Fortran::lower::pft::FunctionLikeUnit>()) {
      // Give internal linkage to internal functions. There are no name clash
      // risks, but giving global linkage to internal procedure will break the
      // static link register in shared libraries because of the system calls.
      // Also, it should be possible to eliminate the procedure code if all the
      // uses have been inlined.
      fir::factory::setInternalLinkage(func);
    } else {
      func.setVisibility(mlir::SymbolTable::Visibility::Public);
    }
    assert(blockId == 0 && "invalid blockId");
    assert(activeConstructStack.empty() && "invalid construct stack state");

    // Manage floating point exception, halting mode, and rounding mode
    // settings at function entry and exit.
    if (!funit.isMainProgram())
      manageFPEnvironment(funit);

    mapDummiesAndResults(funit, callee);

    // Map host associated symbols from parent procedure if any.
    if (funit.parentHasHostAssoc())
      funit.parentHostAssoc().internalProcedureBindings(*this, localSymbols);

    // Non-primary results of a function with multiple entry points.
    // These result values share storage with the primary result.
    llvm::SmallVector<Fortran::lower::pft::Variable> deferredFuncResultList;

    // Backup actual argument for entry character results with different
    // lengths. It needs to be added to the non-primary results symbol before
    // mapSymbolAttributes is called.
    Fortran::lower::SymbolBox resultArg;
    if (std::optional<Fortran::lower::CalleeInterface::PassedEntity>
            passedResult = callee.getPassedResult())
      resultArg = lookupSymbol(passedResult->entity->get());

    Fortran::lower::AggregateStoreMap storeMap;

    // Map all containing submodule and module equivalences and variables, in
    // case they are referenced. It might be better to limit this to variables
    // that are actually referenced, although that is more complicated when
    // there are equivalenced variables.
    auto &scopeVariableListMap =
        Fortran::lower::pft::getScopeVariableListMap(funit);
    for (auto *scp = &scope.parent(); !scp->IsGlobal(); scp = &scp->parent())
      if (scp->kind() == Fortran::semantics::Scope::Kind::Module)
        for (const auto &var : Fortran::lower::pft::getScopeVariableList(
                 *scp, scopeVariableListMap))
          if (!var.isRuntimeTypeInfoData())
            instantiateVar(var, storeMap);

    // Map function equivalences and variables.
    mlir::Value primaryFuncResultStorage;
    for (const Fortran::lower::pft::Variable &var :
         Fortran::lower::pft::getScopeVariableList(scope)) {
      // Always instantiate aggregate storage blocks.
      if (var.isAggregateStore()) {
        instantiateVar(var, storeMap);
        continue;
      }
      const Fortran::semantics::Symbol &sym = var.getSymbol();
      if (funit.parentHasHostAssoc()) {
        // Never instantiate host associated variables, as they are already
        // instantiated from an argument tuple. Instead, just bind the symbol
        // to the host variable, which must be in the map.
        const Fortran::semantics::Symbol &ultimate = sym.GetUltimate();
        if (funit.parentHostAssoc().isAssociated(ultimate)) {
          copySymbolBinding(ultimate, sym);
          continue;
        }
      }
      if (!sym.IsFuncResult() || !funit.primaryResult) {
        instantiateVar(var, storeMap);
      } else if (&sym == funit.primaryResult) {
        instantiateVar(var, storeMap);
        primaryFuncResultStorage = getSymbolAddress(sym);
      } else {
        deferredFuncResultList.push_back(var);
      }
    }

    // TODO: should use same mechanism as equivalence?
    // One blocking point is character entry returns that need special handling
    // since they are not locally allocated but come as argument. CHARACTER(*)
    // is not something that fits well with equivalence lowering.
    for (const Fortran::lower::pft::Variable &altResult :
         deferredFuncResultList) {
      Fortran::lower::StatementContext stmtCtx;
      if (std::optional<Fortran::lower::CalleeInterface::PassedEntity>
              passedResult = callee.getPassedResult()) {
        mapBlockArgToDummyOrResult(altResult.getSymbol(), resultArg.getAddr(),
                                   /*isResult=*/true);
        Fortran::lower::mapSymbolAttributes(*this, altResult, localSymbols,
                                            stmtCtx);
      } else {
        // catch cases where the allocation for the function result storage type
        // doesn't match the type of this symbol
        mlir::Value preAlloc = primaryFuncResultStorage;
        mlir::Type resTy = primaryFuncResultStorage.getType();
        mlir::Type symTy = genType(altResult);
        mlir::Type wrappedSymTy = fir::ReferenceType::get(symTy);
        if (resTy != wrappedSymTy) {
          // check size of the pointed to type so we can't overflow by writing
          // double precision to a single precision allocation, etc
          LLVM_ATTRIBUTE_UNUSED auto getBitWidth = [this](mlir::Type ty) {
            // 15.6.2.6.3: differering result types should be integer, real,
            // complex or logical
            if (auto cmplx = mlir::dyn_cast_or_null<fir::ComplexType>(ty)) {
              fir::KindTy kind = cmplx.getFKind();
              return 2 * builder->getKindMap().getRealBitsize(kind);
            }
            if (auto logical = mlir::dyn_cast_or_null<fir::LogicalType>(ty)) {
              fir::KindTy kind = logical.getFKind();
              return builder->getKindMap().getLogicalBitsize(kind);
            }
            return ty.getIntOrFloatBitWidth();
          };
          assert(getBitWidth(fir::unwrapRefType(resTy)) >= getBitWidth(symTy));

          // convert the storage to the symbol type so that the hlfir.declare
          // gets the correct type for this symbol
          preAlloc = builder->create<fir::ConvertOp>(getCurrentLocation(),
                                                     wrappedSymTy, preAlloc);
        }

        Fortran::lower::mapSymbolAttributes(*this, altResult, localSymbols,
                                            stmtCtx, preAlloc);
      }
    }

    // If this is a host procedure with host associations, then create the tuple
    // of pointers for passing to the internal procedures.
    if (!funit.getHostAssoc().empty())
      funit.getHostAssoc().hostProcedureBindings(*this, localSymbols);

    // Unregister all dummy symbols, so that their cloning (e.g. for OpenMP
    // privatization) does not create the cloned hlfir.declare operations
    // with dummy_scope operands.
    resetRegisteredDummySymbols();

    // Create most function blocks in advance.
    createEmptyBlocks(funit.evaluationList);

    // Reinstate entry block as the current insertion point.
    builder->setInsertionPointToEnd(&func.front());

    if (callee.hasAlternateReturns()) {
      // Create a local temp to hold the alternate return index.
      // Give it an integer index type and the subroutine name (for dumps).
      // Attach it to the subroutine symbol in the localSymbols map.
      // Initialize it to zero, the "fallthrough" alternate return value.
      const Fortran::semantics::Symbol &symbol = funit.getSubprogramSymbol();
      mlir::Location loc = toLocation();
      mlir::Type idxTy = builder->getIndexType();
      mlir::Value altResult =
          builder->createTemporary(loc, idxTy, toStringRef(symbol.name()));
      addSymbol(symbol, altResult);
      mlir::Value zero = builder->createIntegerConstant(loc, idxTy, 0);
      builder->create<fir::StoreOp>(loc, zero, altResult);
    }

    if (Fortran::lower::pft::Evaluation *alternateEntryEval =
            funit.getEntryEval())
      genBranch(alternateEntryEval->lexicalSuccessor->block);
  }

  /// Create global blocks for the current function. This eliminates the
  /// distinction between forward and backward targets when generating
  /// branches. A block is "global" if it can be the target of a GOTO or
  /// other source code branch. A block that can only be targeted by a
  /// compiler generated branch is "local". For example, a DO loop preheader
  /// block containing loop initialization code is global. A loop header
  /// block, which is the target of the loop back edge, is local. Blocks
  /// belong to a region. Any block within a nested region must be replaced
  /// with a block belonging to that region. Branches may not cross region
  /// boundaries.
  void createEmptyBlocks(
      std::list<Fortran::lower::pft::Evaluation> &evaluationList) {
    mlir::Region *region = &builder->getRegion();
    for (Fortran::lower::pft::Evaluation &eval : evaluationList) {
      if (eval.isNewBlock)
        eval.block = builder->createBlock(region);
      if (eval.isConstruct() || eval.isDirective()) {
        if (eval.lowerAsUnstructured()) {
          createEmptyBlocks(eval.getNestedEvaluations());
        } else if (eval.hasNestedEvaluations()) {
          // A structured construct that is a target starts a new block.
          Fortran::lower::pft::Evaluation &constructStmt =
              eval.getFirstNestedEvaluation();
          if (constructStmt.isNewBlock)
            constructStmt.block = builder->createBlock(region);
        }
      }
    }
  }

  /// Return the predicate: "current block does not have a terminator branch".
  bool blockIsUnterminated() {
    mlir::Block *currentBlock = builder->getBlock();
    return currentBlock->empty() ||
           !currentBlock->back().hasTrait<mlir::OpTrait::IsTerminator>();
  }

  /// Unconditionally switch code insertion to a new block.
  void startBlock(mlir::Block *newBlock) {
    assert(newBlock && "missing block");
    // Default termination for the current block is a fallthrough branch to
    // the new block.
    if (blockIsUnterminated())
      genBranch(newBlock);
    // Some blocks may be re/started more than once, and might not be empty.
    // If the new block already has (only) a terminator, set the insertion
    // point to the start of the block. Otherwise set it to the end.
    builder->setInsertionPointToStart(newBlock);
    if (blockIsUnterminated())
      builder->setInsertionPointToEnd(newBlock);
  }

  /// Conditionally switch code insertion to a new block.
  void maybeStartBlock(mlir::Block *newBlock) {
    if (newBlock)
      startBlock(newBlock);
  }

  void eraseDeadCodeAndBlocks(mlir::RewriterBase &rewriter,
                              llvm::MutableArrayRef<mlir::Region> regions) {
    // WARNING: Do not add passes that can do folding or code motion here
    // because they might cross omp.target region boundaries, which can result
    // in incorrect code. Optimization passes like these must be added after
    // OMP early outlining has been done.
    (void)mlir::eraseUnreachableBlocks(rewriter, regions);
    (void)mlir::runRegionDCE(rewriter, regions);
  }

  /// Finish translation of a function.
  void endNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    setCurrentPosition(Fortran::lower::pft::stmtSourceLoc(funit.endStmt));
    if (funit.isMainProgram()) {
      bridge.openAccCtx().finalizeAndPop();
      bridge.fctCtx().finalizeAndPop();
      genExitRoutine();
    } else {
      genFIRProcedureExit(funit, funit.getSubprogramSymbol());
    }
    funit.finalBlock = nullptr;
    LLVM_DEBUG(llvm::dbgs() << "\n[bridge - endNewFunction";
               if (auto *sym = funit.scope->symbol()) llvm::dbgs()
               << " " << sym->name();
               llvm::dbgs() << "] generated IR:\n\n"
                            << *builder->getFunction() << '\n');
    // Eliminate dead code as a prerequisite to calling other IR passes.
    // FIXME: This simplification should happen in a normal pass, not here.
    mlir::IRRewriter rewriter(*builder);
    (void)eraseDeadCodeAndBlocks(rewriter, {builder->getRegion()});
    delete builder;
    builder = nullptr;
    hostAssocTuple = mlir::Value{};
    localSymbols.clear();
    blockId = 0;
    dummyArgsScope = mlir::Value{};
    resetRegisteredDummySymbols();
  }

  /// Helper to generate GlobalOps when the builder is not positioned in any
  /// region block. This is required because the FirOpBuilder assumes it is
  /// always positioned inside a region block when creating globals, the easiest
  /// way comply is to create a dummy function and to throw it afterwards.
  void createGlobalOutsideOfFunctionLowering(
      const std::function<void()> &createGlobals) {
    // FIXME: get rid of the bogus function context and instantiate the
    // globals directly into the module.
    mlir::MLIRContext *context = &getMLIRContext();
    mlir::SymbolTable *symbolTable = getMLIRSymbolTable();
    mlir::func::FuncOp func = fir::FirOpBuilder::createFunction(
        mlir::UnknownLoc::get(context), getModuleOp(),
        fir::NameUniquer::doGenerated("Sham"),
        mlir::FunctionType::get(context, std::nullopt, std::nullopt),
        symbolTable);
    func.addEntryBlock();
    builder = new fir::FirOpBuilder(func, bridge.getKindMap(), symbolTable);
    assert(builder && "FirOpBuilder did not instantiate");
    builder->setFastMathFlags(bridge.getLoweringOptions().getMathOptions());
    createGlobals();
    if (mlir::Region *region = func.getCallableRegion())
      region->dropAllReferences();
    func.erase();
    delete builder;
    builder = nullptr;
    localSymbols.clear();
    resetRegisteredDummySymbols();
  }

  /// Instantiate the data from a BLOCK DATA unit.
  void lowerBlockData(Fortran::lower::pft::BlockDataUnit &bdunit) {
    createGlobalOutsideOfFunctionLowering([&]() {
      Fortran::lower::AggregateStoreMap fakeMap;
      for (const auto &[_, sym] : bdunit.symTab) {
        if (sym->has<Fortran::semantics::ObjectEntityDetails>()) {
          Fortran::lower::pft::Variable var(*sym, true);
          instantiateVar(var, fakeMap);
        }
      }
    });
  }

  /// Create fir::Global for all the common blocks that appear in the program.
  void
  lowerCommonBlocks(const Fortran::semantics::CommonBlockList &commonBlocks) {
    createGlobalOutsideOfFunctionLowering(
        [&]() { Fortran::lower::defineCommonBlocks(*this, commonBlocks); });
  }

  /// Create intrinsic module array constant definitions.
  void createIntrinsicModuleDefinitions(Fortran::lower::pft::Program &pft) {
    // The intrinsic module scope, if present, is the first scope.
    const Fortran::semantics::Scope *intrinsicModuleScope = nullptr;
    for (Fortran::lower::pft::Program::Units &u : pft.getUnits()) {
      Fortran::common::visit(
          Fortran::common::visitors{
              [&](Fortran::lower::pft::FunctionLikeUnit &f) {
                intrinsicModuleScope = &f.getScope().parent();
              },
              [&](Fortran::lower::pft::ModuleLikeUnit &m) {
                intrinsicModuleScope = &m.getScope().parent();
              },
              [&](Fortran::lower::pft::BlockDataUnit &b) {},
              [&](Fortran::lower::pft::CompilerDirectiveUnit &d) {},
              [&](Fortran::lower::pft::OpenACCDirectiveUnit &d) {},
          },
          u);
      if (intrinsicModuleScope) {
        while (!intrinsicModuleScope->IsGlobal())
          intrinsicModuleScope = &intrinsicModuleScope->parent();
        intrinsicModuleScope = &intrinsicModuleScope->children().front();
        break;
      }
    }
    if (!intrinsicModuleScope || !intrinsicModuleScope->IsIntrinsicModules())
      return;
    for (const auto &scope : intrinsicModuleScope->children()) {
      llvm::StringRef modName = toStringRef(scope.symbol()->name());
      if (modName != "__fortran_ieee_exceptions")
        continue;
      for (auto &var : Fortran::lower::pft::getScopeVariableList(scope)) {
        const Fortran::semantics::Symbol &sym = var.getSymbol();
        if (sym.test(Fortran::semantics::Symbol::Flag::CompilerCreated))
          continue;
        const auto *object =
            sym.detailsIf<Fortran::semantics::ObjectEntityDetails>();
        if (object && object->IsArray() && object->init())
          Fortran::lower::createIntrinsicModuleGlobal(*this, var);
      }
    }
  }

  /// Lower a procedure (nest).
  void lowerFunc(Fortran::lower::pft::FunctionLikeUnit &funit) {
    setCurrentPosition(funit.getStartingSourceLoc());
    for (int entryIndex = 0, last = funit.entryPointList.size();
         entryIndex < last; ++entryIndex) {
      funit.setActiveEntry(entryIndex);
      startNewFunction(funit); // the entry point for lowering this procedure
      for (Fortran::lower::pft::Evaluation &eval : funit.evaluationList)
        genFIR(eval);
      endNewFunction(funit);
    }
    funit.setActiveEntry(0);
    for (Fortran::lower::pft::ContainedUnit &unit : funit.containedUnitList)
      if (auto *f = std::get_if<Fortran::lower::pft::FunctionLikeUnit>(&unit))
        lowerFunc(*f); // internal procedure
  }

  /// Lower module variable definitions to fir::globalOp and OpenMP/OpenACC
  /// declarative construct.
  void lowerModuleDeclScope(Fortran::lower::pft::ModuleLikeUnit &mod) {
    setCurrentPosition(mod.getStartingSourceLoc());
    createGlobalOutsideOfFunctionLowering([&]() {
      auto &scopeVariableListMap =
          Fortran::lower::pft::getScopeVariableListMap(mod);
      for (const auto &var : Fortran::lower::pft::getScopeVariableList(
               mod.getScope(), scopeVariableListMap)) {
        // Only define the variables owned by this module.
        const Fortran::semantics::Scope *owningScope = var.getOwningScope();
        if (!owningScope || mod.getScope() == *owningScope)
          Fortran::lower::defineModuleVariable(*this, var);
      }
      for (auto &eval : mod.evaluationList)
        genFIR(eval);
    });
  }

  /// Lower functions contained in a module.
  void lowerMod(Fortran::lower::pft::ModuleLikeUnit &mod) {
    for (Fortran::lower::pft::ContainedUnit &unit : mod.containedUnitList)
      if (auto *f = std::get_if<Fortran::lower::pft::FunctionLikeUnit>(&unit))
        lowerFunc(*f);
  }

  void setCurrentPosition(const Fortran::parser::CharBlock &position) {
    if (position != Fortran::parser::CharBlock{})
      currentPosition = position;
  }

  /// Set current position at the location of \p parseTreeNode. Note that the
  /// position is updated automatically when visiting statements, but not when
  /// entering higher level nodes like constructs or procedures. This helper is
  /// intended to cover the latter cases.
  template <typename A>
  void setCurrentPositionAt(const A &parseTreeNode) {
    setCurrentPosition(Fortran::parser::FindSourceLocation(parseTreeNode));
  }

  //===--------------------------------------------------------------------===//
  // Utility methods
  //===--------------------------------------------------------------------===//

  /// Convert a parser CharBlock to a Location
  mlir::Location toLocation(const Fortran::parser::CharBlock &cb) {
    return genLocation(cb);
  }

  mlir::Location toLocation() { return toLocation(currentPosition); }
  void setCurrentEval(Fortran::lower::pft::Evaluation &eval) {
    evalPtr = &eval;
  }
  Fortran::lower::pft::Evaluation &getEval() {
    assert(evalPtr);
    return *evalPtr;
  }

  std::optional<Fortran::evaluate::Shape>
  getShape(const Fortran::lower::SomeExpr &expr) {
    return Fortran::evaluate::GetShape(foldingContext, expr);
  }

  //===--------------------------------------------------------------------===//
  // Analysis on a nested explicit iteration space.
  //===--------------------------------------------------------------------===//

  void analyzeExplicitSpace(const Fortran::parser::ConcurrentHeader &header) {
    explicitIterSpace.pushLevel();
    for (const Fortran::parser::ConcurrentControl &ctrl :
         std::get<std::list<Fortran::parser::ConcurrentControl>>(header.t)) {
      const Fortran::semantics::Symbol *ctrlVar =
          std::get<Fortran::parser::Name>(ctrl.t).symbol;
      explicitIterSpace.addSymbol(ctrlVar);
    }
    if (const auto &mask =
            std::get<std::optional<Fortran::parser::ScalarLogicalExpr>>(
                header.t);
        mask.has_value())
      analyzeExplicitSpace(*Fortran::semantics::GetExpr(*mask));
  }
  template <bool LHS = false, typename A>
  void analyzeExplicitSpace(const Fortran::evaluate::Expr<A> &e) {
    explicitIterSpace.exprBase(&e, LHS);
  }
  void analyzeExplicitSpace(const Fortran::evaluate::Assignment *assign) {
    auto analyzeAssign = [&](const Fortran::lower::SomeExpr &lhs,
                             const Fortran::lower::SomeExpr &rhs) {
      analyzeExplicitSpace</*LHS=*/true>(lhs);
      analyzeExplicitSpace(rhs);
    };
    Fortran::common::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::ProcedureRef &procRef) {
              // Ensure the procRef expressions are the one being visited.
              assert(procRef.arguments().size() == 2);
              const Fortran::lower::SomeExpr *lhs =
                  procRef.arguments()[0].value().UnwrapExpr();
              const Fortran::lower::SomeExpr *rhs =
                  procRef.arguments()[1].value().UnwrapExpr();
              assert(lhs && rhs &&
                     "user defined assignment arguments must be expressions");
              analyzeAssign(*lhs, *rhs);
            },
            [&](const auto &) { analyzeAssign(assign->lhs, assign->rhs); }},
        assign->u);
    explicitIterSpace.endAssign();
  }
  void analyzeExplicitSpace(const Fortran::parser::ForallAssignmentStmt &stmt) {
    Fortran::common::visit([&](const auto &s) { analyzeExplicitSpace(s); },
                           stmt.u);
  }
  void analyzeExplicitSpace(const Fortran::parser::AssignmentStmt &s) {
    analyzeExplicitSpace(s.typedAssignment->v.operator->());
  }
  void analyzeExplicitSpace(const Fortran::parser::PointerAssignmentStmt &s) {
    analyzeExplicitSpace(s.typedAssignment->v.operator->());
  }
  void analyzeExplicitSpace(const Fortran::parser::WhereConstruct &c) {
    analyzeExplicitSpace(
        std::get<
            Fortran::parser::Statement<Fortran::parser::WhereConstructStmt>>(
            c.t)
            .statement);
    for (const Fortran::parser::WhereBodyConstruct &body :
         std::get<std::list<Fortran::parser::WhereBodyConstruct>>(c.t))
      analyzeExplicitSpace(body);
    for (const Fortran::parser::WhereConstruct::MaskedElsewhere &e :
         std::get<std::list<Fortran::parser::WhereConstruct::MaskedElsewhere>>(
             c.t))
      analyzeExplicitSpace(e);
    if (const auto &e =
            std::get<std::optional<Fortran::parser::WhereConstruct::Elsewhere>>(
                c.t);
        e.has_value())
      analyzeExplicitSpace(e.operator->());
  }
  void analyzeExplicitSpace(const Fortran::parser::WhereConstructStmt &ws) {
    const Fortran::lower::SomeExpr *exp = Fortran::semantics::GetExpr(
        std::get<Fortran::parser::LogicalExpr>(ws.t));
    addMaskVariable(exp);
    analyzeExplicitSpace(*exp);
  }
  void analyzeExplicitSpace(
      const Fortran::parser::WhereConstruct::MaskedElsewhere &ew) {
    analyzeExplicitSpace(
        std::get<
            Fortran::parser::Statement<Fortran::parser::MaskedElsewhereStmt>>(
            ew.t)
            .statement);
    for (const Fortran::parser::WhereBodyConstruct &e :
         std::get<std::list<Fortran::parser::WhereBodyConstruct>>(ew.t))
      analyzeExplicitSpace(e);
  }
  void analyzeExplicitSpace(const Fortran::parser::WhereBodyConstruct &body) {
    Fortran::common::visit(
        Fortran::common::visitors{
            [&](const Fortran::common::Indirection<
                Fortran::parser::WhereConstruct> &wc) {
              analyzeExplicitSpace(wc.value());
            },
            [&](const auto &s) { analyzeExplicitSpace(s.statement); }},
        body.u);
  }
  void analyzeExplicitSpace(const Fortran::parser::MaskedElsewhereStmt &stmt) {
    const Fortran::lower::SomeExpr *exp = Fortran::semantics::GetExpr(
        std::get<Fortran::parser::LogicalExpr>(stmt.t));
    addMaskVariable(exp);
    analyzeExplicitSpace(*exp);
  }
  void
  analyzeExplicitSpace(const Fortran::parser::WhereConstruct::Elsewhere *ew) {
    for (const Fortran::parser::WhereBodyConstruct &e :
         std::get<std::list<Fortran::parser::WhereBodyConstruct>>(ew->t))
      analyzeExplicitSpace(e);
  }
  void analyzeExplicitSpace(const Fortran::parser::WhereStmt &stmt) {
    const Fortran::lower::SomeExpr *exp = Fortran::semantics::GetExpr(
        std::get<Fortran::parser::LogicalExpr>(stmt.t));
    addMaskVariable(exp);
    analyzeExplicitSpace(*exp);
    const std::optional<Fortran::evaluate::Assignment> &assign =
        std::get<Fortran::parser::AssignmentStmt>(stmt.t).typedAssignment->v;
    assert(assign.has_value() && "WHERE has no statement");
    analyzeExplicitSpace(assign.operator->());
  }
  void analyzeExplicitSpace(const Fortran::parser::ForallStmt &forall) {
    analyzeExplicitSpace(
        std::get<
            Fortran::common::Indirection<Fortran::parser::ConcurrentHeader>>(
            forall.t)
            .value());
    analyzeExplicitSpace(std::get<Fortran::parser::UnlabeledStatement<
                             Fortran::parser::ForallAssignmentStmt>>(forall.t)
                             .statement);
    analyzeExplicitSpacePop();
  }
  void
  analyzeExplicitSpace(const Fortran::parser::ForallConstructStmt &forall) {
    analyzeExplicitSpace(
        std::get<
            Fortran::common::Indirection<Fortran::parser::ConcurrentHeader>>(
            forall.t)
            .value());
  }
  void analyzeExplicitSpace(const Fortran::parser::ForallConstruct &forall) {
    analyzeExplicitSpace(
        std::get<
            Fortran::parser::Statement<Fortran::parser::ForallConstructStmt>>(
            forall.t)
            .statement);
    for (const Fortran::parser::ForallBodyConstruct &s :
         std::get<std::list<Fortran::parser::ForallBodyConstruct>>(forall.t)) {
      Fortran::common::visit(
          Fortran::common::visitors{
              [&](const Fortran::common::Indirection<
                  Fortran::parser::ForallConstruct> &b) {
                analyzeExplicitSpace(b.value());
              },
              [&](const Fortran::parser::WhereConstruct &w) {
                analyzeExplicitSpace(w);
              },
              [&](const auto &b) { analyzeExplicitSpace(b.statement); }},
          s.u);
    }
    analyzeExplicitSpacePop();
  }

  void analyzeExplicitSpacePop() { explicitIterSpace.popLevel(); }

  void addMaskVariable(Fortran::lower::FrontEndExpr exp) {
    // Note: use i8 to store bool values. This avoids round-down behavior found
    // with sequences of i1. That is, an array of i1 will be truncated in size
    // and be too small. For example, a buffer of type fir.array<7xi1> will have
    // 0 size.
    mlir::Type i64Ty = builder->getIntegerType(64);
    mlir::TupleType ty = fir::factory::getRaggedArrayHeaderType(*builder);
    mlir::Type buffTy = ty.getType(1);
    mlir::Type shTy = ty.getType(2);
    mlir::Location loc = toLocation();
    mlir::Value hdr = builder->createTemporary(loc, ty);
    // FIXME: Is there a way to create a `zeroinitializer` in LLVM-IR dialect?
    // For now, explicitly set lazy ragged header to all zeros.
    // auto nilTup = builder->createNullConstant(loc, ty);
    // builder->create<fir::StoreOp>(loc, nilTup, hdr);
    mlir::Type i32Ty = builder->getIntegerType(32);
    mlir::Value zero = builder->createIntegerConstant(loc, i32Ty, 0);
    mlir::Value zero64 = builder->createIntegerConstant(loc, i64Ty, 0);
    mlir::Value flags = builder->create<fir::CoordinateOp>(
        loc, builder->getRefType(i64Ty), hdr, zero);
    builder->create<fir::StoreOp>(loc, zero64, flags);
    mlir::Value one = builder->createIntegerConstant(loc, i32Ty, 1);
    mlir::Value nullPtr1 = builder->createNullConstant(loc, buffTy);
    mlir::Value var = builder->create<fir::CoordinateOp>(
        loc, builder->getRefType(buffTy), hdr, one);
    builder->create<fir::StoreOp>(loc, nullPtr1, var);
    mlir::Value two = builder->createIntegerConstant(loc, i32Ty, 2);
    mlir::Value nullPtr2 = builder->createNullConstant(loc, shTy);
    mlir::Value shape = builder->create<fir::CoordinateOp>(
        loc, builder->getRefType(shTy), hdr, two);
    builder->create<fir::StoreOp>(loc, nullPtr2, shape);
    implicitIterSpace.addMaskVariable(exp, var, shape, hdr);
    explicitIterSpace.outermostContext().attachCleanup(
        [builder = this->builder, hdr, loc]() {
          fir::runtime::genRaggedArrayDeallocate(loc, *builder, hdr);
        });
  }

  void createRuntimeTypeInfoGlobals() {}

  bool lowerToHighLevelFIR() const {
    return bridge.getLoweringOptions().getLowerToHighLevelFIR();
  }

  // Returns the mangling prefix for the given constant expression.
  std::string getConstantExprManglePrefix(mlir::Location loc,
                                          const Fortran::lower::SomeExpr &expr,
                                          mlir::Type eleTy) {
    return Fortran::common::visit(
        [&](const auto &x) -> std::string {
          using T = std::decay_t<decltype(x)>;
          if constexpr (Fortran::common::HasMember<
                            T, Fortran::lower::CategoryExpression>) {
            if constexpr (T::Result::category ==
                          Fortran::common::TypeCategory::Derived) {
              if (const auto *constant =
                      std::get_if<Fortran::evaluate::Constant<
                          Fortran::evaluate::SomeDerived>>(&x.u))
                return Fortran::lower::mangle::mangleArrayLiteral(eleTy,
                                                                  *constant);
              fir::emitFatalError(loc,
                                  "non a constant derived type expression");
            } else {
              return Fortran::common::visit(
                  [&](const auto &someKind) -> std::string {
                    using T = std::decay_t<decltype(someKind)>;
                    using TK = Fortran::evaluate::Type<T::Result::category,
                                                       T::Result::kind>;
                    if (const auto *constant =
                            std::get_if<Fortran::evaluate::Constant<TK>>(
                                &someKind.u)) {
                      return Fortran::lower::mangle::mangleArrayLiteral(
                          nullptr, *constant);
                    }
                    fir::emitFatalError(
                        loc, "not a Fortran::evaluate::Constant<T> expression");
                    return {};
                  },
                  x.u);
            }
          } else {
            fir::emitFatalError(loc, "unexpected expression");
          }
        },
        expr.u);
  }

  /// Performing OpenACC lowering action that were deferred to the end of
  /// lowering.
  void finalizeOpenACCLowering() {
    Fortran::lower::finalizeOpenACCRoutineAttachment(getModuleOp(),
                                                     accRoutineInfos);
  }

  /// Performing OpenMP lowering actions that were deferred to the end of
  /// lowering.
  void finalizeOpenMPLowering(
      const Fortran::semantics::Symbol *globalOmpRequiresSymbol) {
    if (!ompDeferredDeclareTarget.empty()) {
      bool deferredDeviceFuncFound =
          Fortran::lower::markOpenMPDeferredDeclareTargetFunctions(
              getModuleOp().getOperation(), ompDeferredDeclareTarget, *this);
      ompDeviceCodeFound = ompDeviceCodeFound || deferredDeviceFuncFound;
    }

    // Set the module attribute related to OpenMP requires directives
    if (ompDeviceCodeFound)
      Fortran::lower::genOpenMPRequires(getModuleOp().getOperation(),
                                        globalOmpRequiresSymbol);
  }

  /// Record fir.dummy_scope operation for this function.
  /// It will be used to set dummy_scope operand of the hlfir.declare
  /// operations.
  void setDummyArgsScope(mlir::Value val) {
    assert(!dummyArgsScope && val);
    dummyArgsScope = val;
  }

  /// Record the given symbol as a dummy argument of this function.
  void registerDummySymbol(Fortran::semantics::SymbolRef symRef) {
    auto *sym = &*symRef;
    registeredDummySymbols.insert(sym);
  }

  /// Reset all registered dummy symbols.
  void resetRegisteredDummySymbols() { registeredDummySymbols.clear(); }

  //===--------------------------------------------------------------------===//

  Fortran::lower::LoweringBridge &bridge;
  Fortran::evaluate::FoldingContext foldingContext;
  fir::FirOpBuilder *builder = nullptr;
  Fortran::lower::pft::Evaluation *evalPtr = nullptr;
  Fortran::lower::SymMap localSymbols;
  Fortran::parser::CharBlock currentPosition;
  TypeInfoConverter typeInfoConverter;

  // Stack to manage object deallocation and finalization at construct exits.
  llvm::SmallVector<ConstructContext> activeConstructStack;

  /// BLOCK name mangling component map
  int blockId = 0;
  Fortran::lower::mangle::ScopeBlockIdMap scopeBlockIdMap;

  /// FORALL statement/construct context
  Fortran::lower::ExplicitIterSpace explicitIterSpace;

  /// WHERE statement/construct mask expression stack
  Fortran::lower::ImplicitIterSpace implicitIterSpace;

  /// Tuple of host associated variables
  mlir::Value hostAssocTuple;

  /// Value of fir.dummy_scope operation for this function.
  mlir::Value dummyArgsScope;

  /// A set of dummy argument symbols for this function.
  /// The set is only preserved during the instatiation
  /// of variables for this function.
  llvm::SmallPtrSet<const Fortran::semantics::Symbol *, 16>
      registeredDummySymbols;

  /// A map of unique names for constant expressions.
  /// The names are used for representing the constant expressions
  /// with global constant initialized objects.
  /// The names are usually prefixed by a mangling string based
  /// on the element type of the constant expression, but the element
  /// type is not used as a key into the map (so the assumption is that
  /// the equivalent constant expressions are prefixed using the same
  /// element type).
  llvm::DenseMap<const Fortran::lower::SomeExpr *, std::string> literalNamesMap;

  /// Storage for Constant expressions used as keys for literalNamesMap.
  llvm::SmallVector<std::unique_ptr<Fortran::lower::SomeExpr>>
      literalExprsStorage;

  /// A counter for uniquing names in `literalNamesMap`.
  std::uint64_t uniqueLitId = 0;

  /// Deferred OpenACC routine attachment.
  Fortran::lower::AccRoutineInfoMappingList accRoutineInfos;

  /// Whether an OpenMP target region or declare target function/subroutine
  /// intended for device offloading has been detected
  bool ompDeviceCodeFound = false;

  /// Keeps track of symbols defined as declare target that could not be
  /// processed at the time of lowering the declare target construct, such
  /// as certain cases where interfaces are declared but not defined within
  /// a module.
  llvm::SmallVector<Fortran::lower::OMPDeferredDeclareTargetInfo>
      ompDeferredDeclareTarget;

  const Fortran::lower::ExprToValueMap *exprValueOverrides{nullptr};

  /// Stack of derived type under construction to avoid infinite loops when
  /// dealing with recursive derived types. This is held in the bridge because
  /// the state needs to be maintained between data and function type lowering
  /// utilities to deal with procedure pointer components whose arguments have
  /// the type of the containing derived type.
  Fortran::lower::TypeConstructionStack typeConstructionStack;
  /// MLIR symbol table of the fir.global/func.func operations. Note that it is
  /// not guaranteed to contain all operations of the ModuleOp with Symbol
  /// attribute since mlirSymbolTable must pro-actively be maintained when
  /// new Symbol operations are created.
  mlir::SymbolTable mlirSymbolTable;
};

} // namespace

Fortran::evaluate::FoldingContext
Fortran::lower::LoweringBridge::createFoldingContext() {
  return {getDefaultKinds(), getIntrinsicTable(), getTargetCharacteristics(),
          getLanguageFeatures(), tempNames};
}

void Fortran::lower::LoweringBridge::lower(
    const Fortran::parser::Program &prg,
    const Fortran::semantics::SemanticsContext &semanticsContext) {
  std::unique_ptr<Fortran::lower::pft::Program> pft =
      Fortran::lower::createPFT(prg, semanticsContext);
  if (dumpBeforeFir)
    Fortran::lower::dumpPFT(llvm::errs(), *pft);
  FirConverter converter{*this};
  converter.run(*pft);
}

void Fortran::lower::LoweringBridge::parseSourceFile(llvm::SourceMgr &srcMgr) {
  mlir::OwningOpRef<mlir::ModuleOp> owningRef =
      mlir::parseSourceFile<mlir::ModuleOp>(srcMgr, &context);
  module.reset(new mlir::ModuleOp(owningRef.get().getOperation()));
  owningRef.release();
}

Fortran::lower::LoweringBridge::LoweringBridge(
    mlir::MLIRContext &context,
    Fortran::semantics::SemanticsContext &semanticsContext,
    const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds,
    const Fortran::evaluate::IntrinsicProcTable &intrinsics,
    const Fortran::evaluate::TargetCharacteristics &targetCharacteristics,
    const Fortran::parser::AllCookedSources &cooked, llvm::StringRef triple,
    fir::KindMapping &kindMap,
    const Fortran::lower::LoweringOptions &loweringOptions,
    const std::vector<Fortran::lower::EnvironmentDefault> &envDefaults,
    const Fortran::common::LanguageFeatureControl &languageFeatures,
    const llvm::TargetMachine &targetMachine, const llvm::StringRef tuneCPU)
    : semanticsContext{semanticsContext}, defaultKinds{defaultKinds},
      intrinsics{intrinsics}, targetCharacteristics{targetCharacteristics},
      cooked{&cooked}, context{context}, kindMap{kindMap},
      loweringOptions{loweringOptions}, envDefaults{envDefaults},
      languageFeatures{languageFeatures} {
  // Register the diagnostic handler.
  context.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
    llvm::raw_ostream &os = llvm::errs();
    switch (diag.getSeverity()) {
    case mlir::DiagnosticSeverity::Error:
      os << "error: ";
      break;
    case mlir::DiagnosticSeverity::Remark:
      os << "info: ";
      break;
    case mlir::DiagnosticSeverity::Warning:
      os << "warning: ";
      break;
    default:
      break;
    }
    if (!mlir::isa<mlir::UnknownLoc>(diag.getLocation()))
      os << diag.getLocation() << ": ";
    os << diag << '\n';
    os.flush();
    return mlir::success();
  });

  auto getPathLocation = [&semanticsContext, &context]() -> mlir::Location {
    std::optional<std::string> path;
    const auto &allSources{semanticsContext.allCookedSources().allSources()};
    if (auto initial{allSources.GetFirstFileProvenance()};
        initial && !initial->empty()) {
      if (const auto *sourceFile{allSources.GetSourceFile(initial->start())}) {
        path = sourceFile->path();
      }
    }

    if (path.has_value()) {
      llvm::SmallString<256> curPath(*path);
      llvm::sys::fs::make_absolute(curPath);
      llvm::sys::path::remove_dots(curPath);
      return mlir::FileLineColLoc::get(&context, curPath.str(), /*line=*/0,
                                       /*col=*/0);
    } else {
      return mlir::UnknownLoc::get(&context);
    }
  };

  // Create the module and attach the attributes.
  module = std::make_unique<mlir::ModuleOp>(
      mlir::ModuleOp::create(getPathLocation()));
  assert(module.get() && "module was not created");
  fir::setTargetTriple(*module.get(), triple);
  fir::setKindMapping(*module.get(), kindMap);
  fir::setTargetCPU(*module.get(), targetMachine.getTargetCPU());
  fir::setTuneCPU(*module.get(), tuneCPU);
  fir::setTargetFeatures(*module.get(), targetMachine.getTargetFeatureString());
  fir::support::setMLIRDataLayout(*module.get(),
                                  targetMachine.createDataLayout());
  fir::setIdent(*module.get(), Fortran::common::getFlangFullVersion());
}

void Fortran::lower::genCleanUpInRegionIfAny(
    mlir::Location loc, fir::FirOpBuilder &builder, mlir::Region &region,
    Fortran::lower::StatementContext &context) {
  if (!context.hasCode())
    return;
  mlir::OpBuilder::InsertPoint insertPt = builder.saveInsertionPoint();
  if (region.empty())
    builder.createBlock(&region);
  else
    builder.setInsertionPointToEnd(&region.front());
  context.finalizeAndPop();
  hlfir::YieldOp::ensureTerminator(region, builder, loc);
  builder.restoreInsertionPoint(insertPt);
}
