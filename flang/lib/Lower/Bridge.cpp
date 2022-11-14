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
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/Coarray.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/ConvertVariable.h"
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
#include "flang/Optimizer/Builder/Runtime/Character.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Runtime/EnvironmentDefaults.h"
#include "flang/Optimizer/Builder/Runtime/Ragged.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Runtime/iostat.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

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
      : loopVariableSym{sym}, lowerExpr{Fortran::semantics::GetExpr(lower)},
        upperExpr{Fortran::semantics::GetExpr(upper)},
        stepExpr{Fortran::semantics::GetExpr(step)}, isUnordered{isUnordered} {}

  IncrementLoopInfo(IncrementLoopInfo &&) = default;
  IncrementLoopInfo &operator=(IncrementLoopInfo &&x) { return x; }

  bool isStructured() const { return !headerBlock; }

  /// \return true if for this do loop its do-variable's value
  /// is represented as the block argument of the do loop's
  /// region. In this case the data type of the block argument
  /// matches the original data type of the do-variable as written
  /// in user code, and the value is adjusted using the step value
  /// on each iteration of the do loop.
  ///
  /// When do-variable's data type is an integer type shorter
  /// than IndexType, processing the do-variable separately
  /// from the do loop's iteration index allows getting rid
  /// of type casts, which can make backend optimizations easier.
  /// In particular, computing the do variable value from
  /// the iteration index may introduce chains like trunc->arith->sext,
  /// which may be optimized into sequences of shift operations
  /// in InstCombine, which then prevents vectorizer from recognizing
  /// unit-strided accesses.
  ///
  /// We could have disabled the extra iteration variable usage
  /// for cases when its data type is not shorter than IndexType,
  /// but this requires having proper DataLayout set up for the enclosing
  /// module. This is currently blocked by llvm-project#57230 issue.
  bool doVarIsALoopArg() const { return isStructured() && !isUnordered; }

  mlir::Type getLoopVariableType() const {
    assert(loopVariable && "must be set");
    return fir::unwrapRefType(loopVariable.getType());
  }

  // Data members common to both structured and unstructured loops.
  const Fortran::semantics::Symbol &loopVariableSym;
  const Fortran::lower::SomeExpr *lowerExpr;
  const Fortran::lower::SomeExpr *upperExpr;
  const Fortran::lower::SomeExpr *stepExpr;
  const Fortran::lower::SomeExpr *maskExpr = nullptr;
  bool isUnordered; // do concurrent, forall
  llvm::SmallVector<const Fortran::semantics::Symbol *> localInitSymList;
  llvm::SmallVector<const Fortran::semantics::Symbol *> sharedSymList;
  mlir::Value loopVariable = nullptr;
  mlir::Value stepValue = nullptr; // possible uses in multiple blocks

  // Data members for structured loops.
  fir::DoLoopOp doLoop = nullptr;

  // Data members for unstructured loops.
  bool hasRealControl = false;
  mlir::Value tripVariable = nullptr;
  mlir::Block *headerBlock = nullptr; // loop entry and test block
  mlir::Block *maskBlock = nullptr;   // concurrent loop mask block
  mlir::Block *bodyBlock = nullptr;   // first loop body block
  mlir::Block *exitBlock = nullptr;   // loop exit target block
};

/// Helper class to generate the runtime type info global data. This data
/// is required to describe the derived type to the runtime so that it can
/// operate over it. It must be ensured this data will be generated for every
/// derived type lowered in the current translated unit. However, this data
/// cannot be generated before FuncOp have been created for functions since the
/// initializers may take their address (e.g for type bound procedures). This
/// class allows registering all the required runtime type info while it is not
/// possible to create globals, and to generate this data after function
/// lowering.
class RuntimeTypeInfoConverter {
  /// Store the location and symbols of derived type info to be generated.
  /// The location of the derived type instantiation is also stored because
  /// runtime type descriptor symbol are compiler generated and cannot be mapped
  /// to user code on their own.
  struct TypeInfoSymbol {
    Fortran::semantics::SymbolRef symbol;
    mlir::Location loc;
  };

public:
  void registerTypeInfoSymbol(Fortran::lower::AbstractConverter &converter,
                              mlir::Location loc,
                              Fortran::semantics::SymbolRef typeInfoSym) {
    if (seen.contains(typeInfoSym))
      return;
    seen.insert(typeInfoSym);
    if (!skipRegistration) {
      registeredTypeInfoSymbols.emplace_back(TypeInfoSymbol{typeInfoSym, loc});
      return;
    }
    // Once the registration is closed, symbols cannot be added to the
    // registeredTypeInfoSymbols list because it may be iterated over.
    // However, after registration is closed, it is safe to directly generate
    // the globals because all FuncOps whose addresses may be required by the
    // initializers have been generated.
    Fortran::lower::createRuntimeTypeInfoGlobal(converter, loc,
                                                typeInfoSym.get());
  }

  void createTypeInfoGlobals(Fortran::lower::AbstractConverter &converter) {
    skipRegistration = true;
    for (const TypeInfoSymbol &info : registeredTypeInfoSymbols)
      Fortran::lower::createRuntimeTypeInfoGlobal(converter, info.loc,
                                                  info.symbol.get());
    registeredTypeInfoSymbols.clear();
  }

private:
  /// Store the runtime type descriptors that will be required for the
  /// derived type that have been converted to FIR derived types.
  llvm::SmallVector<TypeInfoSymbol> registeredTypeInfoSymbols;
  /// Create derived type runtime info global immediately without storing the
  /// symbol in registeredTypeInfoSymbols.
  bool skipRegistration = false;
  /// Track symbols symbols processed during and after the registration
  /// to avoid infinite loops between type conversions and global variable
  /// creation.
  llvm::SmallSetVector<Fortran::semantics::SymbolRef, 64> seen;
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
        bridge{bridge}, foldingContext{bridge.createFoldingContext()} {}
  virtual ~FirConverter() = default;

  /// Convert the PFT to FIR.
  void run(Fortran::lower::pft::Program &pft) {
    // Preliminary translation pass.

    // - Lower common blocks from the PFT common block list that contains a
    // consolidated list of the common blocks (with the initialization if any in
    // the Program, and with the common block biggest size in all its
    // appearance). This is done before lowering any scope declarations because
    // it is not know at the local scope level what MLIR type common blocks
    // should have to suit all its usage in the compilation unit.
    lowerCommonBlocks(pft.getCommonBlocks());

    //  - Declare all functions that have definitions so that definition
    //    signatures prevail over call site signatures.
    //  - Define module variables and OpenMP/OpenACC declarative construct so
    //    that they are available before lowering any function that may use
    //    them.
    bool hasMainProgram = false;
    for (Fortran::lower::pft::Program::Units &u : pft.getUnits()) {
      std::visit(Fortran::common::visitors{
                     [&](Fortran::lower::pft::FunctionLikeUnit &f) {
                       if (f.isMainProgram())
                         hasMainProgram = true;
                       declareFunction(f);
                     },
                     [&](Fortran::lower::pft::ModuleLikeUnit &m) {
                       lowerModuleDeclScope(m);
                       for (Fortran::lower::pft::FunctionLikeUnit &f :
                            m.nestedFunctions)
                         declareFunction(f);
                     },
                     [&](Fortran::lower::pft::BlockDataUnit &b) {},
                     [&](Fortran::lower::pft::CompilerDirectiveUnit &d) {},
                 },
                 u);
    }

    // Primary translation pass.
    for (Fortran::lower::pft::Program::Units &u : pft.getUnits()) {
      std::visit(
          Fortran::common::visitors{
              [&](Fortran::lower::pft::FunctionLikeUnit &f) { lowerFunc(f); },
              [&](Fortran::lower::pft::ModuleLikeUnit &m) { lowerMod(m); },
              [&](Fortran::lower::pft::BlockDataUnit &b) {},
              [&](Fortran::lower::pft::CompilerDirectiveUnit &d) {
                setCurrentPosition(
                    d.get<Fortran::parser::CompilerDirective>().source);
                mlir::emitWarning(toLocation(),
                                  "ignoring all compiler directives");
              },
          },
          u);
    }

    /// Once all the code has been translated, create runtime type info
    /// global data structure for the derived types that have been
    /// processed.
    createGlobalOutsideOfFunctionLowering(
        [&]() { runtimeTypeInfoConverter.createTypeInfoGlobals(*this); });

    // Create the list of any environment defaults for the runtime to set. The
    // runtime default list is only created if there is a main program to ensure
    // it only happens once and to provide consistent results if multiple files
    // are compiled separately.
    if (hasMainProgram)
      createGlobalOutsideOfFunctionLowering([&]() {
        // FIXME: Ideally, this would create a call to a runtime function
        // accepting the list of environment defaults. That way, we would not
        // need to add an extern pointer to the runtime and said pointer would
        // not need to be generated even if no defaults are specified.
        // However, generating main or changing when the runtime reads
        // environment variables is required to do so.
        fir::runtime::genEnvironmentDefaults(*builder, toLocation(),
                                             bridge.getEnvironmentDefaults());
      });
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
    for (Fortran::lower::pft::FunctionLikeUnit &f : funit.nestedFunctions)
      collectHostAssociatedVariables(f, escapeHost);
    funit.setHostAssociatedSymbols(escapeHost);

    // Declare internal procedures
    for (Fortran::lower::pft::FunctionLikeUnit &f : funit.nestedFunctions)
      declareFunction(f);
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
        const Fortran::semantics::Scope &ultimateScope = ultimate.owner();
        if (ultimateScope.kind() ==
                Fortran::semantics::Scope::Kind::MainProgram ||
            ultimateScope.kind() == Fortran::semantics::Scope::Kind::Subprogram)
          if (ultimateScope != *internalScope &&
              ultimateScope.Contains(*internalScope)) {
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
  getSymbolExtendedValue(const Fortran::semantics::Symbol &sym) override final {
    Fortran::lower::SymbolBox sb = localSymbols.lookupSymbol(sym);
    assert(sb && "symbol box not found");
    return sb.toExtendedValue();
  }

  mlir::Value impliedDoBinding(llvm::StringRef name) override final {
    mlir::Value val = localSymbols.lookupImpliedDo(name);
    if (!val)
      fir::emitFatalError(toLocation(), "ac-do-variable has no binding");
    return val;
  }

  void copySymbolBinding(Fortran::lower::SymbolRef src,
                         Fortran::lower::SymbolRef target) override final {
    localSymbols.addSymbol(target, lookupSymbol(src).toExtendedValue());
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
    localSymbols.addSymbol(sym, exval, /*forced=*/true);
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
    auto iter = owningProc.labelEvaluationMap.find(label);
    if (iter == owningProc.labelEvaluationMap.end())
      return nullptr;
    return iter->second;
  }

  fir::ExtendedValue
  translateToExtendedValue(mlir::Location loc,
                           hlfir::EntityWithAttributes entity,
                           Fortran::lower::StatementContext &context) {
    auto [exv, exvCleanup] =
        hlfir::translateToExtendedValue(loc, getFirOpBuilder(), entity);
    if (exvCleanup)
      context.attachCleanup(*exvCleanup);
    return exv;
  }

  fir::ExtendedValue
  genExprAddr(const Fortran::lower::SomeExpr &expr,
              Fortran::lower::StatementContext &context,
              mlir::Location *locPtr = nullptr) override final {
    mlir::Location loc = locPtr ? *locPtr : toLocation();
    if (bridge.getLoweringOptions().getLowerToHighLevelFIR()) {
      hlfir::EntityWithAttributes loweredExpr =
          Fortran::lower::convertExprToHLFIR(loc, *this, expr, localSymbols,
                                             context);
      if (fir::FortranVariableOpInterface variable =
              loweredExpr.getIfVariable())
        if (!variable.isBox())
          return translateToExtendedValue(loc, loweredExpr, context);
      TODO(loc, "lower expr that is not a scalar or explicit shape array "
                "variable to HLFIR address");
    }
    return Fortran::lower::createSomeExtendedAddress(loc, *this, expr,
                                                     localSymbols, context);
  }

  fir::ExtendedValue
  genExprValue(const Fortran::lower::SomeExpr &expr,
               Fortran::lower::StatementContext &context,
               mlir::Location *locPtr = nullptr) override final {
    mlir::Location loc = locPtr ? *locPtr : toLocation();
    if (bridge.getLoweringOptions().getLowerToHighLevelFIR()) {
      hlfir::EntityWithAttributes loweredExpr =
          Fortran::lower::convertExprToHLFIR(loc, *this, expr, localSymbols,
                                             context);
      fir::ExtendedValue exv =
          translateToExtendedValue(loc, loweredExpr, context);
      // Load scalar references to integer, logical, real, or complex value
      // to an mlir value, dereference allocatable and pointers, and get rid
      // of fir.box that are no needed or create a copy into contiguous memory.
      return exv.match(
          [&](const fir::UnboxedValue &box) -> fir::ExtendedValue {
            if (mlir::Type elementType = fir::dyn_cast_ptrEleTy(box.getType()))
              if (fir::isa_trivial(elementType))
                return getFirOpBuilder().create<fir::LoadOp>(loc, box);
            return box;
          },
          [&](const fir::CharBoxValue &box) -> fir::ExtendedValue {
            return box;
          },
          [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue {
            return box;
          },
          [&](const fir::CharArrayBoxValue &box) -> fir::ExtendedValue {
            return box;
          },
          [&](const auto &) -> fir::ExtendedValue {
            TODO(loc, "lower descriptor designator to HLFIR value");
          });
    }
    return Fortran::lower::createSomeExtendedExpression(loc, *this, expr,
                                                        localSymbols, context);
  }

  fir::ExtendedValue
  genExprBox(mlir::Location loc, const Fortran::lower::SomeExpr &expr,
             Fortran::lower::StatementContext &stmtCtx) override final {
    if (bridge.getLoweringOptions().getLowerToHighLevelFIR()) {
      hlfir::EntityWithAttributes loweredExpr =
          Fortran::lower::convertExprToHLFIR(loc, *this, expr, localSymbols,
                                             stmtCtx);
      if (fir::FortranVariableOpInterface variable =
              loweredExpr.getIfVariable())
        if (variable.isBoxValue() || !variable.isBoxAddress()) {
          auto exv = translateToExtendedValue(loc, loweredExpr, stmtCtx);
          return fir::factory::createBoxValue(getFirOpBuilder(), loc, exv);
        }
      TODO(loc,
           "lower expression value or pointer and allocatable to HLFIR box");
    }
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
        llvm::None);
  }

  bool createHostAssociateVarClone(
      const Fortran::semantics::Symbol &sym) override final {
    mlir::Location loc = genLocation(sym.name());
    mlir::Type symType = genType(sym);
    const auto *details = sym.detailsIf<Fortran::semantics::HostAssocDetails>();
    assert(details && "No host-association found");
    const Fortran::semantics::Symbol &hsym = details->symbol();
    Fortran::lower::SymbolBox hsb = lookupSymbol(hsym);

    auto allocate = [&](llvm::ArrayRef<mlir::Value> shape,
                        llvm::ArrayRef<mlir::Value> typeParams) -> mlir::Value {
      mlir::Value allocVal = builder->allocateLocal(
          loc, symType, mangleName(sym), toStringRef(sym.GetUltimate().name()),
          /*pinned=*/true, shape, typeParams,
          sym.GetUltimate().attrs().test(Fortran::semantics::Attr::TARGET));
      return allocVal;
    };

    fir::ExtendedValue hexv = getExtendedValue(hsb);
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
          return fir::MutableBoxValue(allocate({}, {}),
                                      box.nonDeferredLenParams(), {});
        },
        [&](const auto &) -> fir::ExtendedValue {
          mlir::Value temp =
              allocate(fir::factory::getExtents(loc, *builder, hexv),
                       fir::factory::getTypeParams(loc, *builder, hexv));
          return fir::substBase(hexv, temp);
        });

    // Replace all uses of the original with the clone/copy,
    // esepcially for loop bounds (that uses the variable being privatised)
    // since loop bounds use old values that need to be fixed by using the
    // new copied value.
    // Not able to use replaceAllUsesWith() because uses outside
    // the loop body should not use the clone.
    // FIXME: Call privatization before the loop operation.
    mlir::Region &curRegion = getFirOpBuilder().getRegion();
    mlir::Value oldVal = fir::getBase(hexv);
    mlir::Value cloneVal = fir::getBase(exv);
    for (auto &oper : curRegion.getOps()) {
      for (unsigned int ii = 0; ii < oper.getNumOperands(); ++ii) {
        if (oper.getOperand(ii) == oldVal) {
          oper.setOperand(ii, cloneVal);
        }
      }
    }
    return bindIfNewSymbol(sym, exv);
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
    fir::ExtendedValue hexv = getExtendedValue(hsb);

    // 2) Fetch the copied one that will mask the original.
    Fortran::lower::SymbolBox sb = shallowLookupSymbol(sym);
    assert(sb && "Host-associated symbol box not found");
    assert(hsb.getAddr() != sb.getAddr() &&
           "Host and associated symbol boxes are the same");
    fir::ExtendedValue exv = getExtendedValue(sb);

    // 3) Perform the assignment.
    mlir::OpBuilder::InsertPoint insPt = builder->saveInsertionPoint();
    if (copyAssignIP && copyAssignIP->isSet())
      builder->restoreInsertionPoint(*copyAssignIP);
    else
      builder->setInsertionPointAfter(fir::getBase(exv).getDefiningOp());

    fir::ExtendedValue lhs, rhs;
    if (copyAssignIP && copyAssignIP->isSet() &&
        sym.test(Fortran::semantics::Symbol::Flag::OmpLastPrivate)) {
      // lastprivate case
      lhs = hexv;
      rhs = exv;
    } else {
      lhs = exv;
      rhs = hexv;
    }

    mlir::Location loc = genLocation(sym.name());
    mlir::Type symType = genType(sym);
    if (auto seqTy = symType.dyn_cast<fir::SequenceType>()) {
      Fortran::lower::StatementContext stmtCtx;
      Fortran::lower::createSomeArrayAssignment(*this, lhs, rhs, localSymbols,
                                                stmtCtx);
      stmtCtx.finalize();
    } else if (hexv.getBoxOf<fir::CharBoxValue>()) {
      fir::factory::CharacterExprHelper{*builder, loc}.createAssign(lhs, rhs);
    } else if (hexv.getBoxOf<fir::MutableBoxValue>()) {
      TODO(loc, "firstprivatisation of allocatable variables");
    } else {
      auto loadVal = builder->create<fir::LoadOp>(loc, fir::getBase(rhs));
      builder->create<fir::StoreOp>(loc, loadVal, fir::getBase(lhs));
    }

    if (copyAssignIP && copyAssignIP->isSet() &&
        sym.test(Fortran::semantics::Symbol::Flag::OmpLastPrivate))
      builder->restoreInsertionPoint(insPt);
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
            if (checkHostAssociatedSymbols)
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

  /// Generate a `Location` from the `CharBlock`.
  mlir::Location
  genLocation(const Fortran::parser::CharBlock &block) override final {
    if (const Fortran::parser::AllCookedSources *cooked =
            bridge.getCookedSource()) {
      if (std::optional<std::pair<Fortran::parser::SourcePosition,
                                  Fortran::parser::SourcePosition>>
              loc = cooked->GetSourcePositionRange(block)) {
        // loc is a pair (begin, end); use the beginning position
        Fortran::parser::SourcePosition &filePos = loc->first;
        return mlir::FileLineColLoc::get(&getMLIRContext(), filePos.file.path(),
                                         filePos.line, filePos.column);
      }
    }
    return genUnknownLocation();
  }

  fir::FirOpBuilder &getFirOpBuilder() override final { return *builder; }

  mlir::ModuleOp &getModuleOp() override final { return bridge.getModule(); }

  mlir::MLIRContext &getMLIRContext() override final {
    return bridge.getMLIRContext();
  }
  std::string
  mangleName(const Fortran::semantics::Symbol &symbol) override final {
    return Fortran::lower::mangle::mangleName(symbol);
  }

  const fir::KindMapping &getKindMap() override final {
    return bridge.getKindMap();
  }

  mlir::Value hostAssocTupleValue() override final { return hostAssocTuple; }

  /// Record a binding for the ssa-value of the tuple for this function.
  void bindHostAssocTuple(mlir::Value val) override final {
    assert(!hostAssocTuple && val);
    hostAssocTuple = val;
  }

  void registerRuntimeTypeInfo(
      mlir::Location loc,
      Fortran::lower::SymbolRef typeInfoSym) override final {
    runtimeTypeInfoConverter.registerTypeInfoSymbol(*this, loc, typeInfoSym);
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
  lookupSymbol(const Fortran::semantics::Symbol &sym) {
    if (Fortran::lower::SymbolBox v = localSymbols.lookupSymbol(sym))
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
  lookupOneLevelUpSymbol(const Fortran::semantics::Symbol &sym) {
    if (Fortran::lower::SymbolBox v = localSymbols.lookupOneLevelUpSymbol(sym))
      return v;
    return {};
  }

  /// Add the symbol to the local map and return `true`. If the symbol is
  /// already in the map and \p forced is `false`, the map is not updated.
  /// Instead the value `false` is returned.
  bool addSymbol(const Fortran::semantics::SymbolRef sym, mlir::Value val,
                 bool forced = false) {
    if (!forced && lookupSymbol(sym))
      return false;
    localSymbols.addSymbol(sym, val, forced);
    return true;
  }

  bool addCharSymbol(const Fortran::semantics::SymbolRef sym, mlir::Value val,
                     mlir::Value len, bool forced = false) {
    if (!forced && lookupSymbol(sym))
      return false;
    // TODO: ensure val type is fir.array<len x fir.char<kind>> like. Insert
    // cast if needed.
    localSymbols.addCharSymbol(sym, val, len, forced);
    return true;
  }

  fir::ExtendedValue getExtendedValue(Fortran::lower::SymbolBox sb) {
    return sb.match(
        [&](const Fortran::lower::SymbolBox::PointerOrAllocatable &box) {
          return fir::factory::genMutableBoxRead(*builder, getCurrentLocation(),
                                                 box);
        },
        [&sb](auto &) { return sb.toExtendedValue(); });
  }

  /// Generate the address of loop variable \p sym.
  /// If \p sym is not mapped yet, allocate local storage for it.
  mlir::Value genLoopVariableAddress(mlir::Location loc,
                                     const Fortran::semantics::Symbol &sym,
                                     bool isUnordered) {
    if (isUnordered || sym.has<Fortran::semantics::HostAssocDetails>() ||
        sym.has<Fortran::semantics::UseDetails>()) {
      if (!shallowLookupSymbol(sym)) {
        // Do concurrent loop variables are not mapped yet since they are local
        // to the Do concurrent scope (same for OpenMP loops).
        auto newVal = builder->createTemporary(loc, genType(sym),
                                               toStringRef(sym.name()));
        bindIfNewSymbol(sym, newVal);
        return newVal;
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

  /// Insert a new block before \p block.  Leave the insertion point unchanged.
  mlir::Block *insertBlock(mlir::Block *block) {
    mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
    mlir::Block *newBlock = builder->createBlock(block);
    builder->restoreInsertionPoint(insertPt);
    return newBlock;
  }

  mlir::Block *blockOfLabel(Fortran::lower::pft::Evaluation &eval,
                            Fortran::parser::Label label) {
    const Fortran::lower::pft::LabelEvalMap &labelEvaluationMap =
        eval.getOwningProcedure()->labelEvaluationMap;
    const auto iter = labelEvaluationMap.find(label);
    assert(iter != labelEvaluationMap.end() && "label missing from map");
    mlir::Block *block = iter->second->block;
    assert(block && "missing labeled evaluation block");
    return block;
  }

  void genFIRBranch(mlir::Block *targetBlock) {
    assert(targetBlock && "missing unconditional target block");
    builder->create<mlir::cf::BranchOp>(toLocation(), targetBlock);
  }

  void genFIRConditionalBranch(mlir::Value cond, mlir::Block *trueTarget,
                               mlir::Block *falseTarget) {
    assert(trueTarget && "missing conditional branch true block");
    assert(falseTarget && "missing conditional branch false block");
    mlir::Location loc = toLocation();
    mlir::Value bcc = builder->createConvert(loc, builder->getI1Type(), cond);
    builder->create<mlir::cf::CondBranchOp>(loc, bcc, trueTarget, llvm::None,
                                            falseTarget, llvm::None);
  }
  void genFIRConditionalBranch(mlir::Value cond,
                               Fortran::lower::pft::Evaluation *trueTarget,
                               Fortran::lower::pft::Evaluation *falseTarget) {
    genFIRConditionalBranch(cond, trueTarget->block, falseTarget->block);
  }
  void genFIRConditionalBranch(const Fortran::parser::ScalarLogicalExpr &expr,
                               mlir::Block *trueTarget,
                               mlir::Block *falseTarget) {
    Fortran::lower::StatementContext stmtCtx;
    mlir::Value cond =
        createFIRExpr(toLocation(), Fortran::semantics::GetExpr(expr), stmtCtx);
    stmtCtx.finalize();
    genFIRConditionalBranch(cond, trueTarget, falseTarget);
  }
  void genFIRConditionalBranch(const Fortran::parser::ScalarLogicalExpr &expr,
                               Fortran::lower::pft::Evaluation *trueTarget,
                               Fortran::lower::pft::Evaluation *falseTarget) {
    Fortran::lower::StatementContext stmtCtx;
    mlir::Value cond =
        createFIRExpr(toLocation(), Fortran::semantics::GetExpr(expr), stmtCtx);
    stmtCtx.finalize();
    genFIRConditionalBranch(cond, trueTarget->block, falseTarget->block);
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
  void genFIR(const Fortran::parser::EndProgramStmt &) { genExitRoutine(); }

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
        [&](const auto &) -> mlir::Value {
          mlir::Value resultRef = resultSymBox.getAddr();
          mlir::Type resultType = genType(resultSym);
          mlir::Type resultRefType = builder->getRefType(resultType);
          // A function with multiple entry points returning different types
          // tags all result variables with one of the largest types to allow
          // them to share the same storage.  Convert this to the actual type.
          if (resultRef.getType() != resultRefType)
            resultRef = builder->createConvert(loc, resultRefType, resultRef);
          return builder->create<fir::LoadOp>(loc, resultRef);
        });
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
      builder->create<mlir::func::ReturnOp>(toLocation(), retval);
    } else {
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
    stmtCtx.finalize();
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
    setCurrentPosition(stmt.v.source);
    assert(stmt.typedCall && "Call was not analyzed");
    // Call statement lowering shares code with function call lowering.
    mlir::Value res = Fortran::lower::createSubroutineCall(
        *this, *stmt.typedCall, explicitIterSpace, implicitIterSpace,
        localSymbols, stmtCtx, /*isUserDefAssignment=*/false);
    if (!res)
      return; // "Normal" subroutine call.
    // Call with alternate return specifiers.
    // The call returns an index that selects an alternate return branch target.
    llvm::SmallVector<int64_t> indexList;
    llvm::SmallVector<mlir::Block *> blockList;
    int64_t index = 0;
    for (const Fortran::parser::ActualArgSpec &arg :
         std::get<std::list<Fortran::parser::ActualArgSpec>>(stmt.v.t)) {
      const auto &actual = std::get<Fortran::parser::ActualArg>(arg.t);
      if (const auto *altReturn =
              std::get_if<Fortran::parser::AltReturnSpec>(&actual.u)) {
        indexList.push_back(++index);
        blockList.push_back(blockOfLabel(eval, altReturn->v));
      }
    }
    blockList.push_back(eval.nonNopSuccessor().block); // default = fallthrough
    stmtCtx.finalize();
    builder->create<fir::SelectOp>(toLocation(), res, indexList, blockList);
  }

  void genFIR(const Fortran::parser::ComputedGotoStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    Fortran::lower::pft::Evaluation &eval = getEval();
    mlir::Value selectExpr =
        createFIRExpr(toLocation(),
                      Fortran::semantics::GetExpr(
                          std::get<Fortran::parser::ScalarIntExpr>(stmt.t)),
                      stmtCtx);
    stmtCtx.finalize();
    llvm::SmallVector<int64_t> indexList;
    llvm::SmallVector<mlir::Block *> blockList;
    int64_t index = 0;
    for (Fortran::parser::Label label :
         std::get<std::list<Fortran::parser::Label>>(stmt.t)) {
      indexList.push_back(++index);
      blockList.push_back(blockOfLabel(eval, label));
    }
    blockList.push_back(eval.nonNopSuccessor().block); // default
    builder->create<fir::SelectOp>(toLocation(), selectExpr, indexList,
                                   blockList);
  }

  void genFIR(const Fortran::parser::ArithmeticIfStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    Fortran::lower::pft::Evaluation &eval = getEval();
    mlir::Value expr = createFIRExpr(
        toLocation(),
        Fortran::semantics::GetExpr(std::get<Fortran::parser::Expr>(stmt.t)),
        stmtCtx);
    stmtCtx.finalize();
    mlir::Type exprType = expr.getType();
    mlir::Location loc = toLocation();
    if (exprType.isSignlessInteger()) {
      // Arithmetic expression has Integer type.  Generate a SelectCaseOp
      // with ranges {(-inf:-1], 0=default, [1:inf)}.
      mlir::MLIRContext *context = builder->getContext();
      llvm::SmallVector<mlir::Attribute> attrList;
      llvm::SmallVector<mlir::Value> valueList;
      llvm::SmallVector<mlir::Block *> blockList;
      attrList.push_back(fir::UpperBoundAttr::get(context));
      valueList.push_back(builder->createIntegerConstant(loc, exprType, -1));
      blockList.push_back(blockOfLabel(eval, std::get<1>(stmt.t)));
      attrList.push_back(fir::LowerBoundAttr::get(context));
      valueList.push_back(builder->createIntegerConstant(loc, exprType, 1));
      blockList.push_back(blockOfLabel(eval, std::get<3>(stmt.t)));
      attrList.push_back(mlir::UnitAttr::get(context)); // 0 is the "default"
      blockList.push_back(blockOfLabel(eval, std::get<2>(stmt.t)));
      builder->create<fir::SelectCaseOp>(loc, expr, attrList, valueList,
                                         blockList);
      return;
    }
    // Arithmetic expression has Real type.  Generate
    //   sum = expr + expr  [ raise an exception if expr is a NaN ]
    //   if (sum < 0.0) goto L1 else if (sum > 0.0) goto L3 else goto L2
    auto sum = builder->create<mlir::arith::AddFOp>(loc, expr, expr);
    auto zero = builder->create<mlir::arith::ConstantOp>(
        loc, exprType, builder->getFloatAttr(exprType, 0.0));
    auto cond1 = builder->create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OLT, sum, zero);
    mlir::Block *elseIfBlock =
        builder->getBlock()->splitBlock(builder->getInsertionPoint());
    genFIRConditionalBranch(cond1, blockOfLabel(eval, std::get<1>(stmt.t)),
                            elseIfBlock);
    startBlock(elseIfBlock);
    auto cond2 = builder->create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OGT, sum, zero);
    genFIRConditionalBranch(cond2, blockOfLabel(eval, std::get<3>(stmt.t)),
                            blockOfLabel(eval, std::get<2>(stmt.t)));
  }

  void genFIR(const Fortran::parser::AssignedGotoStmt &stmt) {
    // Program requirement 1990 8.2.4 -
    //
    //   At the time of execution of an assigned GOTO statement, the integer
    //   variable must be defined with the value of a statement label of a
    //   branch target statement that appears in the same scoping unit.
    //   Note that the variable may be defined with a statement label value
    //   only by an ASSIGN statement in the same scoping unit as the assigned
    //   GOTO statement.

    mlir::Location loc = toLocation();
    Fortran::lower::pft::Evaluation &eval = getEval();
    const Fortran::lower::pft::SymbolLabelMap &symbolLabelMap =
        eval.getOwningProcedure()->assignSymbolLabelMap;
    const Fortran::semantics::Symbol &symbol =
        *std::get<Fortran::parser::Name>(stmt.t).symbol;
    auto selectExpr =
        builder->create<fir::LoadOp>(loc, getSymbolAddress(symbol));
    auto iter = symbolLabelMap.find(symbol);
    if (iter == symbolLabelMap.end()) {
      // Fail for a nonconforming program unit that does not have any ASSIGN
      // statements.  The front end should check for this.
      mlir::emitError(loc, "(semantics issue) no assigned goto targets");
      exit(1);
    }
    auto labelSet = iter->second;
    llvm::SmallVector<int64_t> indexList;
    llvm::SmallVector<mlir::Block *> blockList;
    auto addLabel = [&](Fortran::parser::Label label) {
      indexList.push_back(label);
      blockList.push_back(blockOfLabel(eval, label));
    };
    // Add labels from an explicit list.  The list may have duplicates.
    for (Fortran::parser::Label label :
         std::get<std::list<Fortran::parser::Label>>(stmt.t)) {
      if (labelSet.count(label) &&
          !llvm::is_contained(indexList, label)) { // ignore duplicates
        addLabel(label);
      }
    }
    // Absent an explicit list, add all possible label targets.
    if (indexList.empty())
      for (auto &label : labelSet)
        addLabel(label);
    // Add a nop/fallthrough branch to the switch for a nonconforming program
    // unit that violates the program requirement above.
    blockList.push_back(eval.nonNopSuccessor().block); // default
    builder->create<fir::SelectOp>(loc, selectExpr, indexList, blockList);
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
      if (const auto *localInitList =
              std::get_if<Fortran::parser::LocalitySpec::LocalInit>(&x.u))
        for (const Fortran::parser::Name &x : localInitList->v)
          info.localInitSymList.push_back(x.symbol);
      if (const auto *sharedList =
              std::get_if<Fortran::parser::LocalitySpec::Shared>(&x.u))
        for (const Fortran::parser::Name &x : sharedList->v)
          info.sharedSymList.push_back(x.symbol);
      if (std::get_if<Fortran::parser::LocalitySpec::Local>(&x.u))
        TODO(toLocation(), "do concurrent locality specs not implemented");
    }
    return incrementLoopNestInfo;
  }

  /// Generate FIR for a DO construct.  There are six variants:
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
      genFIRConditionalBranch(*whileCondition, bodyBlock, exitBlock);
    } else if (const auto *bounds =
                   std::get_if<Fortran::parser::LoopControl::Bounds>(
                       &loopControl->u)) {
      // Non-concurrent increment loop.
      IncrementLoopInfo &info = incrementLoopNestInfo.emplace_back(
          *bounds->name.thing.symbol, bounds->lower, bounds->upper,
          bounds->step);
      if (unstructuredContext) {
        maybeStartBlock(preheaderBlock);
        info.hasRealControl = info.loopVariableSym.GetType()->IsNumeric(
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
          // innermost dimension.  The (first) body block of a non-innermost
          // dimension is the preheader block of the immediately enclosed
          // dimension.  The latch block of a non-innermost dimension is the
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

    // Increment loop begin code.  (Infinite/while code was already generated.)
    if (!infiniteLoop && !whileCondition)
      genFIRIncrementLoopBegin(incrementLoopNestInfo);

    // Loop body code - NonLabelDoStmt and EndDoStmt code is generated here.
    // Their genFIR calls are nops except for block management in some cases.
    for (Fortran::lower::pft::Evaluation &e : eval.getNestedEvaluations())
      genFIR(e, unstructuredContext);

    // Loop end code.
    if (infiniteLoop || whileCondition)
      genFIRBranch(headerBlock);
    else
      genFIRIncrementLoopEnd(incrementLoopNestInfo);
  }

  /// Generate FIR to begin a structured or unstructured increment loop nest.
  void genFIRIncrementLoopBegin(IncrementLoopNestInfo &incrementLoopNestInfo) {
    assert(!incrementLoopNestInfo.empty() && "empty loop nest");
    mlir::Location loc = toLocation();
    auto genControlValue = [&](const Fortran::lower::SomeExpr *expr,
                               const IncrementLoopInfo &info) {
      mlir::Type controlType = info.isStructured() ? builder->getIndexType()
                                                   : info.getLoopVariableType();
      Fortran::lower::StatementContext stmtCtx;
      if (expr)
        return builder->createConvert(loc, controlType,
                                      createFIRExpr(loc, expr, stmtCtx));

      if (info.hasRealControl)
        return builder->createRealConstant(loc, controlType, 1u);
      return builder->createIntegerConstant(loc, controlType, 1); // step
    };
    auto handleLocalitySpec = [&](IncrementLoopInfo &info) {
      // Generate Local Init Assignments
      for (const Fortran::semantics::Symbol *sym : info.localInitSymList) {
        const auto *hostDetails =
            sym->detailsIf<Fortran::semantics::HostAssocDetails>();
        assert(hostDetails && "missing local_init variable host variable");
        const Fortran::semantics::Symbol &hostSym = hostDetails->symbol();
        (void)hostSym;
        TODO(loc, "do concurrent locality specs not implemented");
      }
      // Handle shared locality spec
      for (const Fortran::semantics::Symbol *sym : info.sharedSymList) {
        const auto *hostDetails =
            sym->detailsIf<Fortran::semantics::HostAssocDetails>();
        assert(hostDetails && "missing shared variable host variable");
        const Fortran::semantics::Symbol &hostSym = hostDetails->symbol();
        copySymbolBinding(hostSym, *sym);
      }
    };
    for (IncrementLoopInfo &info : incrementLoopNestInfo) {
      info.loopVariable =
          genLoopVariableAddress(loc, info.loopVariableSym, info.isUnordered);
      mlir::Value lowerValue = genControlValue(info.lowerExpr, info);
      mlir::Value upperValue = genControlValue(info.upperExpr, info);
      info.stepValue = genControlValue(info.stepExpr, info);

      // Structured loop - generate fir.do_loop.
      if (info.isStructured()) {
        mlir::Value doVarInit = nullptr;
        if (info.doVarIsALoopArg())
          doVarInit = builder->createConvert(loc, info.getLoopVariableType(),
                                             lowerValue);

        info.doLoop = builder->create<fir::DoLoopOp>(
            loc, lowerValue, upperValue, info.stepValue, info.isUnordered,
            /*finalCountValue=*/!info.isUnordered,
            doVarInit ? mlir::ValueRange{doVarInit} : mlir::ValueRange{});
        builder->setInsertionPointToStart(info.doLoop.getBody());
        mlir::Value value;
        if (!doVarInit) {
          // Update the loop variable value, as it may have non-index
          // references.
          value = builder->createConvert(loc, info.getLoopVariableType(),
                                         info.doLoop.getInductionVar());
        } else {
          // The loop variable value is the region's argument rather
          // than the DoLoop's index value.
          value = info.doLoop.getRegionIterArgs()[0];
        }
        builder->create<fir::StoreOp>(loc, value, info.loopVariable);
        if (info.maskExpr) {
          Fortran::lower::StatementContext stmtCtx;
          mlir::Value maskCond = createFIRExpr(loc, info.maskExpr, stmtCtx);
          stmtCtx.finalize();
          mlir::Value maskCondCast =
              builder->createConvert(loc, builder->getI1Type(), maskCond);
          auto ifOp = builder->create<fir::IfOp>(loc, maskCondCast,
                                                 /*withElseRegion=*/false);
          builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
        }
        handleLocalitySpec(info);
        continue;
      }

      // Unstructured loop preheader - initialize tripVariable and loopVariable.
      mlir::Value tripCount;
      if (info.hasRealControl) {
        auto diff1 =
            builder->create<mlir::arith::SubFOp>(loc, upperValue, lowerValue);
        auto diff2 =
            builder->create<mlir::arith::AddFOp>(loc, diff1, info.stepValue);
        tripCount =
            builder->create<mlir::arith::DivFOp>(loc, diff2, info.stepValue);
        tripCount =
            builder->createConvert(loc, builder->getIndexType(), tripCount);

      } else {
        auto diff1 =
            builder->create<mlir::arith::SubIOp>(loc, upperValue, lowerValue);
        auto diff2 =
            builder->create<mlir::arith::AddIOp>(loc, diff1, info.stepValue);
        tripCount =
            builder->create<mlir::arith::DivSIOp>(loc, diff2, info.stepValue);
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
        genFIRConditionalBranch(cond, info.maskBlock, info.exitBlock);
        startBlock(info.maskBlock);
        mlir::Block *latchBlock = getEval().getLastNestedEvaluation().block;
        assert(latchBlock && "missing masked concurrent loop latch block");
        Fortran::lower::StatementContext stmtCtx;
        mlir::Value maskCond = createFIRExpr(loc, info.maskExpr, stmtCtx);
        stmtCtx.finalize();
        genFIRConditionalBranch(maskCond, info.bodyBlock, latchBlock);
      } else {
        genFIRConditionalBranch(cond, info.bodyBlock, info.exitBlock);
        if (&info != &incrementLoopNestInfo.back()) // not innermost
          startBlock(info.bodyBlock); // preheader block of enclosed dimension
      }
      if (!info.localInitSymList.empty()) {
        mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
        builder->setInsertionPointToStart(info.bodyBlock);
        handleLocalitySpec(info);
        builder->restoreInsertionPoint(insertPt);
      }
    }
  }

  /// Generate FIR to end a structured or unstructured increment loop nest.
  void genFIRIncrementLoopEnd(IncrementLoopNestInfo &incrementLoopNestInfo) {
    assert(!incrementLoopNestInfo.empty() && "empty loop nest");
    mlir::Location loc = toLocation();
    for (auto it = incrementLoopNestInfo.rbegin(),
              rend = incrementLoopNestInfo.rend();
         it != rend; ++it) {
      IncrementLoopInfo &info = *it;
      if (info.isStructured()) {
        // End fir.do_loop.
        if (!info.isUnordered) {
          builder->setInsertionPointToEnd(info.doLoop.getBody());
          llvm::SmallVector<mlir::Value, 2> results;
          results.push_back(builder->create<mlir::arith::AddIOp>(
              loc, info.doLoop.getInductionVar(), info.doLoop.getStep()));
          if (info.doVarIsALoopArg()) {
            // If we use an extra iteration variable of the same data
            // type as the original do-variable, we have to increment
            // it by the step value. Note that the step has 'index'
            // type, so we need to cast it, first.
            mlir::Value stepCast = builder->createConvert(
                loc, info.getLoopVariableType(), info.doLoop.getStep());
            mlir::Value doVarValue =
                builder->create<fir::LoadOp>(loc, info.loopVariable);
            results.push_back(builder->create<mlir::arith::AddIOp>(
                loc, doVarValue, stepCast));
          }
          builder->create<fir::ResultOp>(loc, results);
        }
        builder->setInsertionPointAfter(info.doLoop);
        if (info.isUnordered)
          continue;
        // The loop control variable may be used after loop execution.
        mlir::Value lcv = nullptr;
        if (info.doVarIsALoopArg()) {
          // Final do-variable value is the second result of the DoLoop.
          assert(info.doLoop.getResults().size() == 2 &&
                 "invalid do-variable handling");
          lcv = info.doLoop.getResult(1);
        } else {
          lcv = builder->createConvert(loc, info.getLoopVariableType(),
                                       info.doLoop.getResult(0));
        }
        builder->create<fir::StoreOp>(loc, lcv, info.loopVariable);
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
      if (info.hasRealControl)
        value =
            builder->create<mlir::arith::AddFOp>(loc, value, info.stepValue);
      else
        value =
            builder->create<mlir::arith::AddIOp>(loc, value, info.stepValue);
      builder->create<fir::StoreOp>(loc, value, info.loopVariable);

      genFIRBranch(info.headerBlock);
      if (&info != &incrementLoopNestInfo.front()) // not outermost
        startBlock(info.exitBlock); // latch block of enclosing dimension
    }
  }

  /// Generate structured or unstructured FIR for an IF construct.
  /// The initial statement may be either an IfStmt or an IfThenStmt.
  void genFIR(const Fortran::parser::IfConstruct &) {
    mlir::Location loc = toLocation();
    Fortran::lower::pft::Evaluation &eval = getEval();
    if (eval.lowerAsStructured()) {
      // Structured fir.if nest.
      fir::IfOp topIfOp, currentIfOp;
      for (Fortran::lower::pft::Evaluation &e : eval.getNestedEvaluations()) {
        auto genIfOp = [&](mlir::Value cond) {
          auto ifOp = builder->create<fir::IfOp>(loc, cond, /*withElse=*/true);
          builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
          return ifOp;
        };
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
        } else {
          genFIR(e, /*unstructuredContext=*/false);
        }
      }
      return;
    }

    // Unstructured branch sequence.
    for (Fortran::lower::pft::Evaluation &e : eval.getNestedEvaluations()) {
      auto genIfBranch = [&](mlir::Value cond) {
        if (e.lexicalSuccessor == e.controlSuccessor) // empty block -> exit
          genFIRConditionalBranch(cond, e.parentConstruct->constructExit,
                                  e.controlSuccessor);
        else // non-empty block
          genFIRConditionalBranch(cond, e.lexicalSuccessor, e.controlSuccessor);
      };
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
      }
    }
  }

  void genFIR(const Fortran::parser::CaseConstruct &) {
    for (Fortran::lower::pft::Evaluation &e : getEval().getNestedEvaluations())
      genFIR(e);
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
    mlir::Value tmp = builder->createTemporary(
        loc, toTy, toStringRef(sym->name()),
        llvm::ArrayRef<mlir::NamedAttribute>{
            Fortran::lower::getAdaptToByRefAttr(*builder)});
    mlir::Value cast = builder->createConvert(loc, toTy, inducVar);
    builder->create<fir::StoreOp>(loc, cast, tmp);
    localSymbols.addSymbol(*sym, tmp, /*force=*/true);
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
    std::visit([&](const auto &x) { genFIR(x); }, stmt.u);
  }

  void genFIR(const Fortran::parser::EndForallStmt &) {
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
    prepareExplicitSpace(stmt);
    genFIR(std::get<
               Fortran::common::Indirection<Fortran::parser::ConcurrentHeader>>(
               stmt.t)
               .value());
    genFIR(std::get<Fortran::parser::UnlabeledStatement<
               Fortran::parser::ForallAssignmentStmt>>(stmt.t)
               .statement);
    cleanupExplicitSpace();
  }

  /// Generate FIR for a FORALL construct.
  void genFIR(const Fortran::parser::ForallConstruct &forall) {
    prepareExplicitSpace(forall);
    genNestedStatement(
        std::get<
            Fortran::parser::Statement<Fortran::parser::ForallConstructStmt>>(
            forall.t));
    for (const Fortran::parser::ForallBodyConstruct &s :
         std::get<std::list<Fortran::parser::ForallBodyConstruct>>(forall.t)) {
      std::visit(
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
  }

  /// Lower the concurrent header specification.
  void genFIR(const Fortran::parser::ForallConstructStmt &stmt) {
    genFIR(std::get<
               Fortran::common::Indirection<Fortran::parser::ConcurrentHeader>>(
               stmt.t)
               .value());
  }

  void genFIR(const Fortran::parser::CompilerDirective &) {
    mlir::emitWarning(toLocation(), "ignoring all compiler directives");
  }

  void genFIR(const Fortran::parser::OpenACCConstruct &acc) {
    mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
    genOpenACCConstruct(*this, bridge.getSemanticsContext(), getEval(), acc);
    for (Fortran::lower::pft::Evaluation &e : getEval().getNestedEvaluations())
      genFIR(e);
    builder->restoreInsertionPoint(insertPt);
  }

  void genFIR(const Fortran::parser::OpenACCDeclarativeConstruct &accDecl) {
    mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
    genOpenACCDeclarativeConstruct(*this, getEval(), accDecl);
    for (Fortran::lower::pft::Evaluation &e : getEval().getNestedEvaluations())
      genFIR(e);
    builder->restoreInsertionPoint(insertPt);
  }

  void genFIR(const Fortran::parser::OpenMPConstruct &omp) {
    mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
    localSymbols.pushScope();
    genOpenMPConstruct(*this, getEval(), omp);

    const Fortran::parser::OpenMPLoopConstruct *ompLoop =
        std::get_if<Fortran::parser::OpenMPLoopConstruct>(&omp.u);

    // If loop is part of an OpenMP Construct then the OpenMP dialect
    // workshare loop operation has already been created. Only the
    // body needs to be created here and the do_loop can be skipped.
    // Skip the number of collapsed loops, which is 1 when there is a
    // no collapse requested.

    Fortran::lower::pft::Evaluation *curEval = &getEval();
    const Fortran::parser::OmpClauseList *loopOpClauseList = nullptr;
    if (ompLoop) {
      loopOpClauseList = &std::get<Fortran::parser::OmpClauseList>(
          std::get<Fortran::parser::OmpBeginLoopDirective>(ompLoop->t).t);
      int64_t collapseValue =
          Fortran::lower::getCollapseValue(*loopOpClauseList);

      curEval = &curEval->getFirstNestedEvaluation();
      for (int64_t i = 1; i < collapseValue; i++) {
        curEval = &*std::next(curEval->getNestedEvaluations().begin());
      }
    }

    for (Fortran::lower::pft::Evaluation &e : curEval->getNestedEvaluations())
      genFIR(e);

    if (ompLoop)
      genOpenMPReduction(*this, *loopOpClauseList);

    localSymbols.popScope();
    builder->restoreInsertionPoint(insertPt);
  }

  void genFIR(const Fortran::parser::OpenMPDeclarativeConstruct &ompDecl) {
    mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
    genOpenMPDeclarativeConstruct(*this, getEval(), ompDecl);
    for (Fortran::lower::pft::Evaluation &e : getEval().getNestedEvaluations())
      genFIR(e);
    builder->restoreInsertionPoint(insertPt);
  }

  /// Generate FIR for a SELECT CASE statement.
  /// The type may be CHARACTER, INTEGER, or LOGICAL.
  void genFIR(const Fortran::parser::SelectCaseStmt &stmt) {
    Fortran::lower::pft::Evaluation &eval = getEval();
    mlir::MLIRContext *context = builder->getContext();
    mlir::Location loc = toLocation();
    Fortran::lower::StatementContext stmtCtx;
    const Fortran::lower::SomeExpr *expr = Fortran::semantics::GetExpr(
        std::get<Fortran::parser::Scalar<Fortran::parser::Expr>>(stmt.t));
    bool isCharSelector = isCharacterCategory(expr->GetType()->category());
    bool isLogicalSelector = isLogicalCategory(expr->GetType()->category());
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
    mlir::Block *defaultBlock = eval.parentConstruct->constructExit->block;
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
      defaultBlock = eval.parentConstruct->constructExit->block;
    attrList.push_back(mlir::UnitAttr::get(context));
    blockList.push_back(defaultBlock);

    // Generate a fir::SelectCaseOp.
    // Explicit branch code is better for the LOGICAL type.  The CHARACTER type
    // does not yet have downstream support, and also uses explicit branch code.
    // The -no-structured-fir option can be used to force generation of INTEGER
    // type branch code.
    if (!isLogicalSelector && !isCharSelector && eval.lowerAsStructured()) {
      // Numeric selector is a ssa register, all temps that may have
      // been generated while evaluating it can be cleaned-up before the
      // fir.select_case.
      stmtCtx.finalize();
      builder->create<fir::SelectCaseOp>(loc, selector, attrList, valueList,
                                         blockList);
      return;
    }

    // Generate a sequence of case value comparisons and branches.
    auto caseValue = valueList.begin();
    auto caseBlock = blockList.begin();
    bool skipFinalization = false;
    for (const auto &attr : llvm::enumerate(attrList)) {
      if (attr.value().isa<mlir::UnitAttr>()) {
        if (attrList.size() == 1)
          stmtCtx.finalize();
        genFIRBranch(*caseBlock++);
        break;
      }
      auto genCond = [&](mlir::Value rhs,
                         mlir::arith::CmpIPredicate pred) -> mlir::Value {
        if (!isCharSelector)
          return builder->create<mlir::arith::CmpIOp>(loc, pred, selector, rhs);
        fir::factory::CharacterExprHelper charHelper{*builder, loc};
        std::pair<mlir::Value, mlir::Value> lhsVal =
            charHelper.createUnboxChar(selector);
        mlir::Value &lhsAddr = lhsVal.first;
        mlir::Value &lhsLen = lhsVal.second;
        std::pair<mlir::Value, mlir::Value> rhsVal =
            charHelper.createUnboxChar(rhs);
        mlir::Value &rhsAddr = rhsVal.first;
        mlir::Value &rhsLen = rhsVal.second;
        mlir::Value result = fir::runtime::genCharCompare(
            *builder, loc, pred, lhsAddr, lhsLen, rhsAddr, rhsLen);
        if (stmtCtx.workListIsEmpty() || skipFinalization)
          return result;
        if (attr.index() == attrList.size() - 2) {
          stmtCtx.finalize();
          return result;
        }
        fir::IfOp ifOp = builder->create<fir::IfOp>(loc, result,
                                                    /*withElseRegion=*/false);
        builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
        stmtCtx.finalizeAndKeep();
        builder->setInsertionPointAfter(ifOp);
        return result;
      };
      mlir::Block *newBlock = insertBlock(*caseBlock);
      if (attr.value().isa<fir::ClosedIntervalAttr>()) {
        mlir::Block *newBlock2 = insertBlock(*caseBlock);
        skipFinalization = true;
        mlir::Value cond =
            genCond(*caseValue++, mlir::arith::CmpIPredicate::sge);
        genFIRConditionalBranch(cond, newBlock, newBlock2);
        builder->setInsertionPointToEnd(newBlock);
        skipFinalization = false;
        mlir::Value cond2 =
            genCond(*caseValue++, mlir::arith::CmpIPredicate::sle);
        genFIRConditionalBranch(cond2, *caseBlock++, newBlock2);
        builder->setInsertionPointToEnd(newBlock2);
        continue;
      }
      mlir::arith::CmpIPredicate pred;
      if (attr.value().isa<fir::PointIntervalAttr>()) {
        pred = mlir::arith::CmpIPredicate::eq;
      } else if (attr.value().isa<fir::LowerBoundAttr>()) {
        pred = mlir::arith::CmpIPredicate::sge;
      } else {
        assert(attr.value().isa<fir::UpperBoundAttr>() &&
               "unexpected predicate");
        pred = mlir::arith::CmpIPredicate::sle;
      }
      mlir::Value cond = genCond(*caseValue++, pred);
      genFIRConditionalBranch(cond, *caseBlock++, newBlock);
      builder->setInsertionPointToEnd(newBlock);
    }
    assert(caseValue == valueList.end() && caseBlock == blockList.end() &&
           "select case list mismatch");
    assert(stmtCtx.workListIsEmpty() && "statement context must be empty");
  }

  fir::ExtendedValue
  genAssociateSelector(const Fortran::lower::SomeExpr &selector,
                       Fortran::lower::StatementContext &stmtCtx) {
    return Fortran::lower::isArraySectionWithoutVectorSubscript(selector)
               ? Fortran::lower::createSomeArrayBox(*this, selector,
                                                    localSymbols, stmtCtx)
               : genExprAddr(selector, stmtCtx);
  }

  void genFIR(const Fortran::parser::AssociateConstruct &) {
    Fortran::lower::StatementContext stmtCtx;
    Fortran::lower::pft::Evaluation &eval = getEval();
    for (Fortran::lower::pft::Evaluation &e : eval.getNestedEvaluations()) {
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
          localSymbols.addSymbol(sym, genAssociateSelector(selector, stmtCtx));
        }
      } else if (e.getIf<Fortran::parser::EndAssociateStmt>()) {
        if (eval.lowerAsUnstructured())
          maybeStartBlock(e.block);
        stmtCtx.finalize();
        localSymbols.popScope();
      } else {
        genFIR(e);
      }
    }
  }

  void genFIR(const Fortran::parser::BlockConstruct &blockConstruct) {
    setCurrentPositionAt(blockConstruct);
    TODO(toLocation(), "BlockConstruct implementation");
  }
  void genFIR(const Fortran::parser::BlockStmt &) {
    TODO(toLocation(), "BlockStmt implementation");
  }
  void genFIR(const Fortran::parser::EndBlockStmt &) {
    TODO(toLocation(), "EndBlockStmt implementation");
  }

  void genFIR(const Fortran::parser::ChangeTeamConstruct &construct) {
    TODO(toLocation(), "ChangeTeamConstruct implementation");
  }
  void genFIR(const Fortran::parser::ChangeTeamStmt &stmt) {
    TODO(toLocation(), "ChangeTeamStmt implementation");
  }
  void genFIR(const Fortran::parser::EndChangeTeamStmt &stmt) {
    TODO(toLocation(), "EndChangeTeamStmt implementation");
  }

  void genFIR(const Fortran::parser::CriticalConstruct &criticalConstruct) {
    setCurrentPositionAt(criticalConstruct);
    TODO(toLocation(), "CriticalConstruct implementation");
  }
  void genFIR(const Fortran::parser::CriticalStmt &) {
    TODO(toLocation(), "CriticalStmt implementation");
  }
  void genFIR(const Fortran::parser::EndCriticalStmt &) {
    TODO(toLocation(), "EndCriticalStmt implementation");
  }

  void genFIR(const Fortran::parser::SelectRankConstruct &selectRankConstruct) {
    setCurrentPositionAt(selectRankConstruct);
    TODO(toLocation(), "SelectRankConstruct implementation");
  }
  void genFIR(const Fortran::parser::SelectRankStmt &) {
    TODO(toLocation(), "SelectRankStmt implementation");
  }
  void genFIR(const Fortran::parser::SelectRankCaseStmt &) {
    TODO(toLocation(), "SelectRankCaseStmt implementation");
  }

  void genFIR(const Fortran::parser::SelectTypeConstruct &selectTypeConstruct) {
    setCurrentPositionAt(selectTypeConstruct);
    TODO(toLocation(), "SelectTypeConstruct implementation");
  }
  void genFIR(const Fortran::parser::SelectTypeStmt &) {
    TODO(toLocation(), "SelectTypeStmt implementation");
  }
  void genFIR(const Fortran::parser::TypeGuardStmt &) {
    TODO(toLocation(), "TypeGuardStmt implementation");
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

    mlir::Block *endBlock = nullptr;
    mlir::Block *eorBlock = nullptr;
    mlir::Block *errBlock = nullptr;
    for (const auto &spec : specList) {
      std::visit(Fortran::common::visitors{
                     [&](const Fortran::parser::EndLabel &label) {
                       endBlock = blockOfLabel(eval, label.v);
                     },
                     [&](const Fortran::parser::EorLabel &label) {
                       eorBlock = blockOfLabel(eval, label.v);
                     },
                     [&](const Fortran::parser::ErrLabel &label) {
                       errBlock = blockOfLabel(eval, label.v);
                     },
                     [](const auto &) {}},
                 spec.u);
    }
    if (!endBlock && !eorBlock && !errBlock)
      return;

    mlir::Location loc = toLocation();
    mlir::Type indexType = builder->getIndexType();
    mlir::Value selector = builder->createConvert(loc, indexType, iostat);
    llvm::SmallVector<int64_t> indexList;
    llvm::SmallVector<mlir::Block *> blockList;
    if (eorBlock) {
      indexList.push_back(Fortran::runtime::io::IostatEor);
      blockList.push_back(eorBlock);
    }
    if (endBlock) {
      indexList.push_back(Fortran::runtime::io::IostatEnd);
      blockList.push_back(endBlock);
    }
    if (errBlock) {
      indexList.push_back(0);
      blockList.push_back(eval.nonNopSuccessor().block);
      // ERR label statement is the default successor.
      blockList.push_back(errBlock);
    } else {
      // Fallthrough successor statement is the default successor.
      blockList.push_back(eval.nonNopSuccessor().block);
    }
    builder->create<fir::SelectOp>(loc, selector, indexList, blockList);
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
      fir::MutableBoxValue box = genExprMutableBox(loc, *expr);
      fir::factory::disassociateMutableBox(*builder, loc, box);
    }
  }

  //===--------------------------------------------------------------------===//

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
      llvm::Optional<llvm::SmallVector<mlir::Value>> lbounds = llvm::None,
      llvm::Optional<llvm::SmallVector<mlir::Value>> ubounds = llvm::None) {

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
    return Fortran::lower::createMutableBox(loc, *this, expr, localSymbols);
  }

  /// Shared for both assignments and pointer assignments.
  void genAssignment(const Fortran::evaluate::Assignment &assign) {
    Fortran::lower::StatementContext stmtCtx;
    mlir::Location loc = toLocation();
    if (explicitIterationSpace()) {
      Fortran::lower::createArrayLoads(*this, explicitIterSpace, localSymbols);
      explicitIterSpace.genLoopNest();
    }
    std::visit(
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
              // Assignment to polymorphic allocatables may require changing the
              // variable dynamic type (See Fortran 2018 10.2.1.3 p3).
              if (lhsType->IsPolymorphic() &&
                  Fortran::lower::isWholeAllocatable(assign.lhs))
                TODO(loc, "assignment to polymorphic allocatable");

              // Note: No ad-hoc handling for pointers is required here. The
              // target will be assigned as per 2018 10.2.1.3 p2. genExprAddr
              // on a pointer returns the target address and not the address of
              // the pointer variable.

              if (assign.lhs.Rank() > 0 || explicitIterationSpace()) {
                // Array assignment
                // See Fortran 2018 10.2.1.3 p5, p6, and p7
                genArrayAssignment(assign, stmtCtx);
                return;
              }

              // Scalar assignment
              const bool isNumericScalar =
                  isNumericScalarCategory(lhsType->category());
              fir::ExtendedValue rhs = isNumericScalar
                                           ? genExprValue(assign.rhs, stmtCtx)
                                           : genExprAddr(assign.rhs, stmtCtx);
              const bool lhsIsWholeAllocatable =
                  Fortran::lower::isWholeAllocatable(assign.lhs);
              llvm::Optional<fir::factory::MutableBoxReallocation> lhsRealloc;
              llvm::Optional<fir::MutableBoxValue> lhsMutableBox;
              auto lhs = [&]() -> fir::ExtendedValue {
                if (lhsIsWholeAllocatable) {
                  lhsMutableBox = genExprMutableBox(loc, assign.lhs);
                  llvm::SmallVector<mlir::Value> lengthParams;
                  if (const fir::CharBoxValue *charBox = rhs.getCharBox())
                    lengthParams.push_back(charBox->getLen());
                  else if (fir::isDerivedWithLenParameters(rhs))
                    TODO(loc, "assignment to derived type allocatable with "
                              "LEN parameters");
                  lhsRealloc = fir::factory::genReallocIfNeeded(
                      *builder, loc, *lhsMutableBox,
                      /*shape=*/llvm::None, lengthParams);
                  return lhsRealloc->newValue;
                }
                return genExprAddr(assign.lhs, stmtCtx);
              }();

              if (isNumericScalar) {
                // Fortran 2018 10.2.1.3 p8 and p9
                // Conversions should have been inserted by semantic analysis,
                // but they can be incorrect between the rhs and lhs. Correct
                // that here.
                mlir::Value addr = fir::getBase(lhs);
                mlir::Value val = fir::getBase(rhs);
                // A function with multiple entry points returning different
                // types tags all result variables with one of the largest
                // types to allow them to share the same storage.  Assignment
                // to a result variable of one of the other types requires
                // conversion to the actual type.
                mlir::Type toTy = genType(assign.lhs);
                mlir::Value cast =
                    builder->convertWithSemantics(loc, toTy, val);
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
                // Fortran 2018 10.2.1.3 p13 and p14
                // Recursively gen an assignment on each element pair.
                fir::factory::genRecordAssignment(*builder, loc, lhs, rhs);
              } else {
                llvm_unreachable("unknown category");
              }
              if (lhsIsWholeAllocatable)
                fir::factory::finalizeRealloc(
                    *builder, loc, lhsMutableBox.value(),
                    /*lbounds=*/llvm::None, /*takeLboundsIfRealloc=*/false,
                    lhsRealloc.value());
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

            // [3] Pointer assignment with possibly empty bounds-spec. R1035: a
            // bounds-spec is a lower bound value.
            [&](const Fortran::evaluate::Assignment::BoundsSpec &lbExprs) {
              if (Fortran::evaluate::IsProcedure(assign.rhs))
                TODO(loc, "procedure pointer assignment");
              std::optional<Fortran::evaluate::DynamicType> lhsType =
                  assign.lhs.GetType();
              std::optional<Fortran::evaluate::DynamicType> rhsType =
                  assign.rhs.GetType();
              // Polymorphic lhs/rhs may need more care. See F2018 10.2.2.3.
              // If the pointer object is not polymorphic (7.3.2.3) and the
              // pointer target is polymorphic with dynamic type that differs
              // from its declared type, the assignment target is the ancestor
              // component of the pointer target that has the type of the
              // pointer object. Otherwise, the assignment target is the pointer
              // target.
              if ((lhsType && !lhsType->IsPolymorphic()) &&
                  (rhsType && rhsType->IsPolymorphic()))
                TODO(loc, "non-polymorphic pointer assignment with polymorphic "
                          "entity on rhs");

              llvm::SmallVector<mlir::Value> lbounds;
              for (const Fortran::evaluate::ExtentExpr &lbExpr : lbExprs)
                lbounds.push_back(
                    fir::getBase(genExprValue(toEvExpr(lbExpr), stmtCtx)));
              if (explicitIterationSpace()) {
                // Pointer assignment in FORALL context. Copy the rhs box value
                // into the lhs box variable.
                genArrayAssignment(assign, stmtCtx, lbounds);
                return;
              }
              fir::MutableBoxValue lhs = genExprMutableBox(loc, assign.lhs);
              Fortran::lower::associateMutableBox(*this, loc, lhs, assign.rhs,
                                                  lbounds, stmtCtx);
            },

            // [4] Pointer assignment with bounds-remapping. R1036: a
            // bounds-remapping is a pair, lower bound and upper bound.
            [&](const Fortran::evaluate::Assignment::BoundsRemapping
                    &boundExprs) {
              std::optional<Fortran::evaluate::DynamicType> lhsType =
                  assign.lhs.GetType();
              std::optional<Fortran::evaluate::DynamicType> rhsType =
                  assign.rhs.GetType();
              // Polymorphic lhs/rhs may need more care. See F2018 10.2.2.3.
              if ((lhsType && lhsType->IsPolymorphic()) ||
                  (rhsType && rhsType->IsPolymorphic()))
                TODO(loc, "pointer assignment involving polymorphic entity");

              llvm::SmallVector<mlir::Value> lbounds;
              llvm::SmallVector<mlir::Value> ubounds;
              for (const std::pair<Fortran::evaluate::ExtentExpr,
                                   Fortran::evaluate::ExtentExpr> &pair :
                   boundExprs) {
                const Fortran::evaluate::ExtentExpr &lbExpr = pair.first;
                const Fortran::evaluate::ExtentExpr &ubExpr = pair.second;
                lbounds.push_back(
                    fir::getBase(genExprValue(toEvExpr(lbExpr), stmtCtx)));
                ubounds.push_back(
                    fir::getBase(genExprValue(toEvExpr(ubExpr), stmtCtx)));
              }
              if (explicitIterationSpace()) {
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
              // Do not generate a temp in case rhs is an array section.
              fir::ExtendedValue rhs =
                  Fortran::lower::isArraySectionWithoutVectorSubscript(
                      assign.rhs)
                      ? Fortran::lower::createSomeArrayBox(
                            *this, assign.rhs, localSymbols, stmtCtx)
                      : genExprAddr(assign.rhs, stmtCtx);
              fir::factory::associateMutableBoxWithRemap(*builder, loc, lhs,
                                                         rhs, lbounds, ubounds);
              if (explicitIterationSpace()) {
                mlir::ValueRange inners = explicitIterSpace.getInnerArgs();
                if (!inners.empty())
                  builder->create<fir::ResultOp>(loc, inners);
              }
            },
        },
        assign.u);
    if (explicitIterationSpace())
      Fortran::lower::createArrayMergeStores(*this, explicitIterSpace);
  }

  void genFIR(const Fortran::parser::WhereConstruct &c) {
    implicitIterSpace.growStack();
    genNestedStatement(
        std::get<
            Fortran::parser::Statement<Fortran::parser::WhereConstructStmt>>(
            c.t));
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
  }
  void genFIR(const Fortran::parser::WhereBodyConstruct &body) {
    std::visit(
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
  void genFIR(const Fortran::parser::WhereConstructStmt &stmt) {
    implicitIterSpace.append(Fortran::semantics::GetExpr(
        std::get<Fortran::parser::LogicalExpr>(stmt.t)));
  }
  void genFIR(const Fortran::parser::WhereConstruct::MaskedElsewhere &ew) {
    genNestedStatement(
        std::get<
            Fortran::parser::Statement<Fortran::parser::MaskedElsewhereStmt>>(
            ew.t));
    for (const auto &body :
         std::get<std::list<Fortran::parser::WhereBodyConstruct>>(ew.t))
      genFIR(body);
  }
  void genFIR(const Fortran::parser::MaskedElsewhereStmt &stmt) {
    implicitIterSpace.append(Fortran::semantics::GetExpr(
        std::get<Fortran::parser::LogicalExpr>(stmt.t)));
  }
  void genFIR(const Fortran::parser::WhereConstruct::Elsewhere &ew) {
    genNestedStatement(
        std::get<Fortran::parser::Statement<Fortran::parser::ElsewhereStmt>>(
            ew.t));
    for (const auto &body :
         std::get<std::list<Fortran::parser::WhereBodyConstruct>>(ew.t))
      genFIR(body);
  }
  void genFIR(const Fortran::parser::ElsewhereStmt &stmt) {
    implicitIterSpace.append(nullptr);
  }
  void genFIR(const Fortran::parser::EndWhereStmt &) {
    implicitIterSpace.shrinkStack();
  }

  void genFIR(const Fortran::parser::WhereStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    const auto &assign = std::get<Fortran::parser::AssignmentStmt>(stmt.t);
    implicitIterSpace.growStack();
    implicitIterSpace.append(Fortran::semantics::GetExpr(
        std::get<Fortran::parser::LogicalExpr>(stmt.t)));
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
    if (funit->isMainProgram()) {
      genExitRoutine();
      return;
    }
    mlir::Location loc = toLocation();
    if (stmt.v) {
      // Alternate return statement - If this is a subroutine where some
      // alternate entries have alternate returns, but the active entry point
      // does not, ignore the alternate return value.  Otherwise, assign it
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
      funit->finalBlock = builder->createBlock(&builder->getRegion());
      builder->restoreInsertionPoint(insPt);
    }
    builder->create<mlir::cf::BranchOp>(loc, funit->finalBlock);
  }

  void genFIR(const Fortran::parser::CycleStmt &) {
    genFIRBranch(getEval().controlSuccessor->block);
  }
  void genFIR(const Fortran::parser::ExitStmt &) {
    genFIRBranch(getEval().controlSuccessor->block);
  }
  void genFIR(const Fortran::parser::GotoStmt &) {
    genFIRBranch(getEval().controlSuccessor->block);
  }

  // Nop statements - No code, or code is generated at the construct level.
  void genFIR(const Fortran::parser::AssociateStmt &) {}       // nop
  void genFIR(const Fortran::parser::CaseStmt &) {}            // nop
  void genFIR(const Fortran::parser::ContinueStmt &) {}        // nop
  void genFIR(const Fortran::parser::ElseIfStmt &) {}          // nop
  void genFIR(const Fortran::parser::ElseStmt &) {}            // nop
  void genFIR(const Fortran::parser::EndAssociateStmt &) {}    // nop
  void genFIR(const Fortran::parser::EndDoStmt &) {}           // nop
  void genFIR(const Fortran::parser::EndFunctionStmt &) {}     // nop
  void genFIR(const Fortran::parser::EndIfStmt &) {}           // nop
  void genFIR(const Fortran::parser::EndMpSubprogramStmt &) {} // nop
  void genFIR(const Fortran::parser::EndSelectStmt &) {}       // nop
  void genFIR(const Fortran::parser::EndSubroutineStmt &) {}   // nop
  void genFIR(const Fortran::parser::EntryStmt &) {}           // nop
  void genFIR(const Fortran::parser::IfStmt &) {}              // nop
  void genFIR(const Fortran::parser::IfThenStmt &) {}          // nop
  void genFIR(const Fortran::parser::NonLabelDoStmt &) {}      // nop
  void genFIR(const Fortran::parser::OmpEndLoopDirective &) {} // nop

  void genFIR(const Fortran::parser::NamelistStmt &) {
    TODO(toLocation(), "NamelistStmt lowering");
  }

  /// Generate FIR for the Evaluation `eval`.
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              bool unstructuredContext = true) {
    if (unstructuredContext) {
      // When transitioning from unstructured to structured code,
      // the structured code could be a target that starts a new block.
      maybeStartBlock(eval.isConstruct() && eval.lowerAsStructured()
                          ? eval.getFirstNestedEvaluation().block
                          : eval.block);
    }

    setCurrentEval(eval);
    setCurrentPosition(eval.position);
    eval.visit([&](const auto &stmt) { genFIR(stmt); });

    if (unstructuredContext && blockIsUnterminated()) {
      // Exit from an unstructured IF or SELECT construct block.
      Fortran::lower::pft::Evaluation *successor{};
      if (eval.isActionStmt())
        successor = eval.controlSuccessor;
      else if (eval.isConstruct() &&
               eval.getLastNestedEvaluation()
                   .lexicalSuccessor->isIntermediateConstructStmt())
        successor = eval.constructExit;
      else if (eval.isConstructStmt() &&
               eval.lexicalSuccessor == eval.controlSuccessor)
        // empty construct block
        successor = eval.parentConstruct->constructExit;
      if (successor && successor->block)
        genFIRBranch(successor->block);
    }
  }

  void mapCPtrArgByValue(const Fortran::semantics::Symbol &sym,
                         mlir::Value val) {
    mlir::Type symTy = Fortran::lower::translateSymbolToFIRType(*this, sym);
    mlir::Location loc = toLocation();
    mlir::Value res = builder->create<fir::AllocaOp>(loc, symTy);
    mlir::Value resAddr =
        fir::factory::genCPtrOrCFunptrAddr(*builder, loc, res, symTy);
    mlir::Value argAddrVal =
        builder->createConvert(loc, fir::unwrapRefType(resAddr.getType()), val);
    builder->create<fir::StoreOp>(loc, argAddrVal, resAddr);
    addSymbol(sym, res);
  }

  void mapTrivialByValue(const Fortran::semantics::Symbol &sym,
                         mlir::Value val) {
    mlir::Location loc = toLocation();
    mlir::Value res = builder->create<fir::AllocaOp>(loc, val.getType());
    builder->create<fir::StoreOp>(loc, val, res);
    addSymbol(sym, res);
  }

  /// Map mlir function block arguments to the corresponding Fortran dummy
  /// variables. When the result is passed as a hidden argument, the Fortran
  /// result is also mapped. The symbol map is used to hold this mapping.
  void mapDummiesAndResults(Fortran::lower::pft::FunctionLikeUnit &funit,
                            const Fortran::lower::CalleeInterface &callee) {
    assert(builder && "require a builder object at this point");
    using PassBy = Fortran::lower::CalleeInterface::PassEntityBy;
    auto mapPassedEntity = [&](const auto arg) {
      if (arg.passBy == PassBy::AddressAndLength) {
        if (callee.characterize().IsBindC())
          return;
        // TODO: now that fir call has some attributes regarding character
        // return, PassBy::AddressAndLength should be retired.
        mlir::Location loc = toLocation();
        fir::factory::CharacterExprHelper charHelp{*builder, loc};
        mlir::Value box =
            charHelp.createEmboxChar(arg.firArgument, arg.firLength);
        addSymbol(arg.entity->get(), box);
      } else {
        if (arg.entity.has_value()) {
          if (arg.passBy == PassBy::Value) {
            mlir::Type argTy = arg.firArgument.getType();
            if (argTy.isa<fir::RecordType>())
              TODO(toLocation(), "derived type argument passed by value");
            if (Fortran::semantics::IsBuiltinCPtr(arg.entity->get()) &&
                Fortran::lower::isCPtrArgByValueType(argTy)) {
              mapCPtrArgByValue(arg.entity->get(), arg.firArgument);
              return;
            }
            if (fir::isa_trivial(argTy)) {
              mapTrivialByValue(arg.entity->get(), arg.firArgument);
              return;
            }
          }
          addSymbol(arg.entity->get(), arg.firArgument);
        } else {
          assert(funit.parentHasHostAssoc());
          funit.parentHostAssoc().internalProcedureBindings(*this,
                                                            localSymbols);
        }
      }
    };
    for (const Fortran::lower::CalleeInterface::PassedEntity &arg :
         callee.getPassedArguments())
      mapPassedEntity(arg);
    if (std::optional<Fortran::lower::CalleeInterface::PassedEntity>
            passedResult = callee.getPassedResult()) {
      mapPassedEntity(*passedResult);
      // FIXME: need to make sure things are OK here. addSymbol may not be OK
      if (funit.primaryResult &&
          passedResult->entity->get() != *funit.primaryResult)
        addSymbol(*funit.primaryResult,
                  getSymbolAddress(passedResult->entity->get()));
    }
  }

  /// Instantiate variable \p var and add it to the symbol map.
  /// See ConvertVariable.cpp.
  void instantiateVar(const Fortran::lower::pft::Variable &var,
                      Fortran::lower::AggregateStoreMap &storeMap) {
    Fortran::lower::instantiateVariable(*this, var, localSymbols, storeMap);
    if (var.hasSymbol() &&
        var.getSymbol().test(
            Fortran::semantics::Symbol::Flag::OmpThreadprivate))
      Fortran::lower::genThreadprivateOp(*this, var);
  }

  /// Prepare to translate a new function
  void startNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    assert(!builder && "expected nullptr");
    Fortran::lower::CalleeInterface callee(funit, *this);
    mlir::func::FuncOp func = callee.addEntryBlockAndMapArguments();
    builder = new fir::FirOpBuilder(func, bridge.getKindMap());
    assert(builder && "FirOpBuilder did not instantiate");
    builder->setFastMathFlags(bridge.getLoweringOptions().getMathOptions());
    builder->setInsertionPointToStart(&func.front());
    func.setVisibility(mlir::SymbolTable::Visibility::Public);

    mapDummiesAndResults(funit, callee);

    // Note: not storing Variable references because getOrderedSymbolTable
    // below returns a temporary.
    llvm::SmallVector<Fortran::lower::pft::Variable> deferredFuncResultList;

    // Backup actual argument for entry character results
    // with different lengths. It needs to be added to the non
    // primary results symbol before mapSymbolAttributes is called.
    Fortran::lower::SymbolBox resultArg;
    if (std::optional<Fortran::lower::CalleeInterface::PassedEntity>
            passedResult = callee.getPassedResult())
      resultArg = lookupSymbol(passedResult->entity->get());

    Fortran::lower::AggregateStoreMap storeMap;
    // The front-end is currently not adding module variables referenced
    // in a module procedure as host associated. As a result we need to
    // instantiate all module variables here if this is a module procedure.
    // It is likely that the front-end behavior should change here.
    // This also applies to internal procedures inside module procedures.
    if (auto *module = Fortran::lower::pft::getAncestor<
            Fortran::lower::pft::ModuleLikeUnit>(funit))
      for (const Fortran::lower::pft::Variable &var :
           module->getOrderedSymbolTable())
        instantiateVar(var, storeMap);

    mlir::Value primaryFuncResultStorage;
    for (const Fortran::lower::pft::Variable &var :
         funit.getOrderedSymbolTable()) {
      // Always instantiate aggregate storage blocks.
      if (var.isAggregateStore()) {
        instantiateVar(var, storeMap);
        continue;
      }
      const Fortran::semantics::Symbol &sym = var.getSymbol();
      if (funit.parentHasHostAssoc()) {
        // Never instantitate host associated variables, as they are already
        // instantiated from an argument tuple. Instead, just bind the symbol to
        // the reference to the host variable, which must be in the map.
        const Fortran::semantics::Symbol &ultimate = sym.GetUltimate();
        if (funit.parentHostAssoc().isAssociated(ultimate)) {
          Fortran::lower::SymbolBox hostBox =
              localSymbols.lookupSymbol(ultimate);
          assert(hostBox && "host association is not in map");
          localSymbols.addSymbol(sym, hostBox.toExtendedValue());
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
        addSymbol(altResult.getSymbol(), resultArg.getAddr());
        Fortran::lower::mapSymbolAttributes(*this, altResult, localSymbols,
                                            stmtCtx);
      } else {
        Fortran::lower::mapSymbolAttributes(*this, altResult, localSymbols,
                                            stmtCtx, primaryFuncResultStorage);
      }
    }

    // If this is a host procedure with host associations, then create the tuple
    // of pointers for passing to the internal procedures.
    if (!funit.getHostAssoc().empty())
      funit.getHostAssoc().hostProcedureBindings(*this, localSymbols);

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
      genFIRBranch(alternateEntryEval->lexicalSuccessor->block);
  }

  /// Create global blocks for the current function.  This eliminates the
  /// distinction between forward and backward targets when generating
  /// branches.  A block is "global" if it can be the target of a GOTO or
  /// other source code branch.  A block that can only be targeted by a
  /// compiler generated branch is "local".  For example, a DO loop preheader
  /// block containing loop initialization code is global.  A loop header
  /// block, which is the target of the loop back edge, is local.  Blocks
  /// belong to a region.  Any block within a nested region must be replaced
  /// with a block belonging to that region.  Branches may not cross region
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
      genFIRBranch(newBlock);
    // Some blocks may be re/started more than once, and might not be empty.
    // If the new block already has (only) a terminator, set the insertion
    // point to the start of the block.  Otherwise set it to the end.
    builder->setInsertionPointToStart(newBlock);
    if (blockIsUnterminated())
      builder->setInsertionPointToEnd(newBlock);
  }

  /// Conditionally switch code insertion to a new block.
  void maybeStartBlock(mlir::Block *newBlock) {
    if (newBlock)
      startBlock(newBlock);
  }

  /// Emit return and cleanup after the function has been translated.
  void endNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    setCurrentPosition(Fortran::lower::pft::stmtSourceLoc(funit.endStmt));
    if (funit.isMainProgram())
      genExitRoutine();
    else
      genFIRProcedureExit(funit, funit.getSubprogramSymbol());
    funit.finalBlock = nullptr;
    LLVM_DEBUG(llvm::dbgs() << "*** Lowering result:\n\n"
                            << *builder->getFunction() << '\n');
    // FIXME: Simplification should happen in a normal pass, not here.
    mlir::IRRewriter rewriter(*builder);
    (void)mlir::simplifyRegions(rewriter,
                                {builder->getRegion()}); // remove dead code
    delete builder;
    builder = nullptr;
    hostAssocTuple = mlir::Value{};
    localSymbols.clear();
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
    mlir::func::FuncOp func = fir::FirOpBuilder::createFunction(
        mlir::UnknownLoc::get(context), getModuleOp(),
        fir::NameUniquer::doGenerated("Sham"),
        mlir::FunctionType::get(context, llvm::None, llvm::None));
    func.addEntryBlock();
    builder = new fir::FirOpBuilder(func, bridge.getKindMap());
    assert(builder && "FirOpBuilder did not instantiate");
    builder->setFastMathFlags(bridge.getLoweringOptions().getMathOptions());
    createGlobals();
    if (mlir::Region *region = func.getCallableRegion())
      region->dropAllReferences();
    func.erase();
    delete builder;
    builder = nullptr;
    localSymbols.clear();
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

  /// Lower a procedure (nest).
  void lowerFunc(Fortran::lower::pft::FunctionLikeUnit &funit) {
    if (!funit.isMainProgram()) {
      const Fortran::semantics::Symbol &procSymbol =
          funit.getSubprogramSymbol();
      if (procSymbol.owner().IsSubmodule())
        TODO(toLocation(), "support for submodules");
      if (Fortran::semantics::IsSeparateModuleProcedureInterface(&procSymbol))
        TODO(toLocation(), "separate module procedure");
    }
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
    for (Fortran::lower::pft::FunctionLikeUnit &f : funit.nestedFunctions)
      lowerFunc(f); // internal procedure
  }

  /// Lower module variable definitions to fir::globalOp and OpenMP/OpenACC
  /// declarative construct.
  void lowerModuleDeclScope(Fortran::lower::pft::ModuleLikeUnit &mod) {
    setCurrentPosition(mod.getStartingSourceLoc());
    createGlobalOutsideOfFunctionLowering([&]() {
      for (const Fortran::lower::pft::Variable &var :
           mod.getOrderedSymbolTable()) {
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
    for (Fortran::lower::pft::FunctionLikeUnit &f : mod.nestedFunctions)
      lowerFunc(f);
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
    std::visit(
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
    std::visit([&](const auto &s) { analyzeExplicitSpace(s); }, stmt.u);
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
    std::visit(Fortran::common::visitors{
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
      std::visit(Fortran::common::visitors{
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

  //===--------------------------------------------------------------------===//

  Fortran::lower::LoweringBridge &bridge;
  Fortran::evaluate::FoldingContext foldingContext;
  fir::FirOpBuilder *builder = nullptr;
  Fortran::lower::pft::Evaluation *evalPtr = nullptr;
  Fortran::lower::SymMap localSymbols;
  Fortran::parser::CharBlock currentPosition;
  RuntimeTypeInfoConverter runtimeTypeInfoConverter;

  /// WHERE statement/construct mask expression stack.
  Fortran::lower::ImplicitIterSpace implicitIterSpace;

  /// FORALL context
  Fortran::lower::ExplicitIterSpace explicitIterSpace;

  /// Tuple of host assoicated variables.
  mlir::Value hostAssocTuple;
};

} // namespace

Fortran::evaluate::FoldingContext
Fortran::lower::LoweringBridge::createFoldingContext() const {
  return {getDefaultKinds(), getIntrinsicTable(), getTargetCharacteristics()};
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
    const std::vector<Fortran::lower::EnvironmentDefault> &envDefaults)
    : semanticsContext{semanticsContext}, defaultKinds{defaultKinds},
      intrinsics{intrinsics}, targetCharacteristics{targetCharacteristics},
      cooked{&cooked}, context{context}, kindMap{kindMap},
      loweringOptions{loweringOptions}, envDefaults{envDefaults} {
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
    if (!diag.getLocation().isa<mlir::UnknownLoc>())
      os << diag.getLocation() << ": ";
    os << diag << '\n';
    os.flush();
    return mlir::success();
  });

  // Create the module and attach the attributes.
  module = std::make_unique<mlir::ModuleOp>(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
  assert(module.get() && "module was not created");
  fir::setTargetTriple(*module.get(), triple);
  fir::setKindMapping(*module.get(), kindMap);
}
