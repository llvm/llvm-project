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
#include "../../runtime/iostat.h"
#include "ConvertVariable.h"
#include "MaskExpr.h"
#include "StatementContext.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/CharacterRuntime.h"
#include "flang/Lower/Coarray.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/IO.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/OpenACC.h"
#include "flang/Lower/OpenMP.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Parser.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "flang-lower-bridge"

static llvm::cl::opt<bool> dumpBeforeFir(
    "fdebug-dump-pre-fir", llvm::cl::init(false),
    llvm::cl::desc("dump the Pre-FIR tree prior to FIR generation"));

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

  // Data members common to both structured and unstructured loops.
  const Fortran::semantics::Symbol &loopVariableSym;
  const Fortran::semantics::SomeExpr *lowerExpr;
  const Fortran::semantics::SomeExpr *upperExpr;
  const Fortran::semantics::SomeExpr *stepExpr;
  const Fortran::semantics::SomeExpr *maskExpr = nullptr;
  bool isUnordered; // do concurrent, forall
  llvm::SmallVector<const Fortran::semantics::Symbol *> localInitSymList;
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

using IncrementLoopNestInfo = llvm::SmallVector<IncrementLoopInfo>;
} // namespace

/// Clone subexpression and wrap it as a generic `Fortran::evaluate::Expr`.
template <typename A>
static Fortran::evaluate::Expr<Fortran::evaluate::SomeType>
toEvExpr(const A &x) {
  return Fortran::evaluate::AsGenericExpr(Fortran::common::Clone(x));
}

//===----------------------------------------------------------------------===//
// FirConverter
//===----------------------------------------------------------------------===//

namespace {
/// Walk over the pre-FIR tree (PFT) and lower it to the FIR dialect of MLIR.
///
/// After building the PFT, the FirConverter processes that representation
/// and lowers it to the FIR executable representation.
class FirConverter : public Fortran::lower::AbstractConverter {
public:
  explicit FirConverter(Fortran::lower::LoweringBridge &bridge)
      : bridge{bridge}, foldingContext{bridge.createFoldingContext()} {}
  virtual ~FirConverter() = default;

  /// Convert the PFT to FIR
  void run(Fortran::lower::pft::Program &pft) {
    // Declare mlir::FuncOp for all the FunctionLikeUnit defined in the PFT
    // before lowering any function bodies so that the definition signatures
    // prevail on call spot signatures.
    declareFunctions(pft);

    // Define variables of the modules defined in this program. This is done
    // first to ensure they are defined before lowering any function that may
    // use them.
    lowerModuleVariables(pft);
    // do translation
    for (auto &u : pft.getUnits()) {
      std::visit(
          Fortran::common::visitors{
              [&](Fortran::lower::pft::FunctionLikeUnit &f) { lowerFunc(f); },
              [&](Fortran::lower::pft::ModuleLikeUnit &m) { lowerMod(m); },
              [&](Fortran::lower::pft::BlockDataUnit &b) { lowerBlockData(b); },
              [&](Fortran::lower::pft::CompilerDirectiveUnit &d) {
                setCurrentPosition(
                    d.get<Fortran::parser::CompilerDirective>().source);
                mlir::emitWarning(toLocation(),
                                  "ignoring all compiler directives");
              },
          },
          u);
    }
  }

  /// Declare mlir::FuncOp for all the FunctionLikeUnit defined in the PFT
  /// without any other side-effects.
  void declareFunctions(Fortran::lower::pft::Program &pft) {
    for (auto &u : pft.getUnits()) {
      std::visit(Fortran::common::visitors{
                     [&](Fortran::lower::pft::FunctionLikeUnit &f) {
                       declareFunction(f);
                     },
                     [&](Fortran::lower::pft::ModuleLikeUnit &m) {
                       for (auto &f : m.nestedFunctions)
                         declareFunction(f);
                     },
                     [&](Fortran::lower::pft::BlockDataUnit &) {
                       // No functions defined in block data.
                     },
                     [&](Fortran::lower::pft::CompilerDirectiveUnit &) {
                       // No functions defined.
                     },
                 },
                 u);
    }
  }

  /// Declare a function.
  void declareFunction(
      Fortran::lower::pft::FunctionLikeUnit &funit,
      llvm::SetVector<const Fortran::semantics::Symbol *> hosted = {}) {
    if (!hosted.empty())
      TODO(toLocation(), "internal procedure has host associated variables");
    for (int entryIndex = 0, last = funit.entryPointList.size();
         entryIndex < last; ++entryIndex) {
      funit.setActiveEntry(entryIndex);
      // Calling CalleeInterface ctor will build the mlir::FuncOp with no other
      // side effects.
      // TODO: when doing some compiler profiling on real apps, it may be worth
      // to check it's better to save the CalleeInterface instead of recomputing
      // it later when lowering the body. CalleeInterface ctor should be linear
      // with the number of arguments, so it is not awful to do it that way for
      // now, but the linear coefficient might be non negligible. Until
      // measured, stick to the solution that impacts the code less.
      Fortran::lower::CalleeInterface{funit, *this};
    }
    funit.setActiveEntry(0);

    llvm::SetVector<const Fortran::semantics::Symbol *> escapeHost;
    for (auto &f : funit.nestedFunctions)
      collectHostAssociatedVariables(f, escapeHost);
    for (auto &f : funit.nestedFunctions)
      declareFunction(f, escapeHost); // internal procedure
  }

  /// Collects the canonical list of all host associated symbols. These bindings
  /// must be aggregated into a tuple which can then be added to each of the
  /// internal procedure declarations and passed at each call site.
  void collectHostAssociatedVariables(
      Fortran::lower::pft::FunctionLikeUnit &funit,
      llvm::SetVector<const Fortran::semantics::Symbol *> &escapees) {
    for (const auto &var : funit.varList[0]) {
      const auto &sym = var.getSymbol();
      if (const auto *details =
              sym.detailsIf<Fortran::semantics::HostAssocDetails>())
        escapees.insert(&details->symbol());
    }
  }

  /// Loop through modules defined in this file to generate the fir::globalOp
  /// for module variables.
  void lowerModuleVariables(Fortran::lower::pft::Program &pft) {
    for (auto &u : pft.getUnits()) {
      std::visit(Fortran::common::visitors{
                     [&](Fortran::lower::pft::ModuleLikeUnit &m) {
                       lowerModuleVariables(m);
                     },
                     [](auto &) {
                       // Not a module, so no processing needed here.
                     },
                 },
                 u);
    }
  }

  //===--------------------------------------------------------------------===//
  // AbstractConverter overrides
  //===--------------------------------------------------------------------===//

  mlir::Value getSymbolAddress(Fortran::lower::SymbolRef sym) override final {
    return lookupSymbol(sym).getAddr();
  }

  bool bindSymbol(Fortran::lower::SymbolRef sym,
                  mlir::Value val) override final {
    if (lookupSymbol(sym))
      return false;
    addSymbol(sym, val);
    return true;
  }

  bool lookupLabelSet(Fortran::lower::SymbolRef sym,
                      Fortran::lower::pft::LabelSet &labelSet) override final {
    auto &owningProc = *getEval().getOwningProcedure();
    auto iter = owningProc.assignSymbolLabelMap.find(sym);
    if (iter == owningProc.assignSymbolLabelMap.end())
      return false;
    labelSet = iter->second;
    return true;
  }

  Fortran::lower::pft::Evaluation *
  lookupLabel(Fortran::lower::pft::Label label) override final {
    auto &owningProc = *getEval().getOwningProcedure();
    auto iter = owningProc.labelEvaluationMap.find(label);
    if (iter == owningProc.labelEvaluationMap.end())
      return nullptr;
    return iter->second;
  }

  fir::ExtendedValue genExprAddr(const Fortran::lower::SomeExpr &expr,
                                 Fortran::lower::StatementContext &context,
                                 mlir::Location *loc = nullptr) override final {
    return createSomeExtendedAddress(loc ? *loc : toLocation(), *this, expr,
                                     localSymbols, context);
  }
  fir::ExtendedValue
  genExprValue(const Fortran::lower::SomeExpr &expr,
               Fortran::lower::StatementContext &context,
               mlir::Location *loc = nullptr) override final {
    return createSomeExtendedExpression(loc ? *loc : toLocation(), *this, expr,
                                        localSymbols, context);
  }
  fir::MutableBoxValue
  genExprMutableBox(mlir::Location loc,
                    const Fortran::lower::SomeExpr &expr) override final {
    return createSomeMutableBox(loc, *this, expr, localSymbols);
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

  mlir::Location getCurrentLocation() override final { return toLocation(); }

  /// Generate a dummy location.
  mlir::Location genLocation() override final {
    // Note: builder may not be instantiated yet
    return mlir::UnknownLoc::get(&getMLIRContext());
  }

  /// Generate a `Location` from the `CharBlock`.
  mlir::Location
  genLocation(const Fortran::parser::CharBlock &block) override final {
    if (const auto *cooked = bridge.getCookedSource()) {
      auto loc = cooked->GetSourcePositionRange(block);
      if (loc.has_value()) {
        // loc is a pair (begin, end); use the beginning position
        auto &filePos = loc->first;
        return mlir::FileLineColLoc::get(filePos.file.path(), filePos.line,
                                         filePos.column, &getMLIRContext());
      }
    }
    return genLocation();
  }

  Fortran::lower::FirOpBuilder &getFirOpBuilder() override final {
    return *builder;
  }

  mlir::ModuleOp &getModuleOp() override final { return bridge.getModule(); }

  mlir::MLIRContext &getMLIRContext() override final {
    return bridge.getMLIRContext();
  }
  std::string
  mangleName(const Fortran::semantics::Symbol &symbol) override final {
    return Fortran::lower::mangle::mangleName(symbol);
  }

  fir::KindMapping &getKindMap() override final { return bridge.getKindMap(); }

private:
  FirConverter() = delete;
  FirConverter(const FirConverter &) = delete;
  FirConverter &operator=(const FirConverter &) = delete;

  //===--------------------------------------------------------------------===//
  // Helper member functions
  //===--------------------------------------------------------------------===//

  mlir::Value createFIRExpr(mlir::Location loc,
                            const Fortran::semantics::SomeExpr *expr,
                            Fortran::lower::StatementContext &stmtCtx) {
    return fir::getBase(genExprValue(*expr, stmtCtx, &loc));
  }

  /// Find the symbol in the local map or return null.
  Fortran::lower::SymbolBox
  lookupSymbol(const Fortran::semantics::Symbol &sym) {
    if (auto v = localSymbols.lookupSymbol(sym))
      return v;
    return {};
  }

  /// Add the symbol to the local map. If the symbol is already in the map, it
  /// is not updated. Instead the value `false` is returned.
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

  mlir::Value createTemp(mlir::Location loc,
                         const Fortran::semantics::Symbol &sym,
                         llvm::ArrayRef<mlir::Value> shape = {}) {
    // FIXME: should return fir::ExtendedValue
    if (auto v = lookupSymbol(sym))
      return v.getAddr();
    auto newVal = builder->createTemporary(loc, genType(sym),
                                           toStringRef(sym.name()), shape);
    addSymbol(sym, newVal);
    return newVal;
  }

  bool isNumericScalarCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Integer ||
           cat == Fortran::common::TypeCategory::Real ||
           cat == Fortran::common::TypeCategory::Complex ||
           cat == Fortran::common::TypeCategory::Logical;
  }
  bool isLogicalCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Logical;
  }
  bool isCharacterCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Character;
  }
  bool isDerivedCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Derived;
  }

  /// Insert a new block before \p block.  Leave the insertion point unchanged.
  mlir::Block *insertBlock(mlir::Block *block) {
    auto insertPt = builder->saveInsertionPoint();
    auto newBlock = builder->createBlock(block);
    builder->restoreInsertionPoint(insertPt);
    return newBlock;
  }

  mlir::Block *blockOfLabel(Fortran::lower::pft::Evaluation &eval,
                            Fortran::parser::Label label) {
    const auto &labelEvaluationMap =
        eval.getOwningProcedure()->labelEvaluationMap;
    const auto iter = labelEvaluationMap.find(label);
    assert(iter != labelEvaluationMap.end() && "label missing from map");
    auto *block = iter->second->block;
    assert(block && "missing labeled evaluation block");
    return block;
  }

  void genFIRBranch(mlir::Block *targetBlock) {
    assert(targetBlock && "missing unconditional target block");
    builder->create<mlir::BranchOp>(toLocation(), targetBlock);
  }

  void genFIRConditionalBranch(mlir::Value cond, mlir::Block *trueTarget,
                               mlir::Block *falseTarget) {
    assert(trueTarget && "missing conditional branch true block");
    assert(falseTarget && "missing conditional branch false block");
    auto loc = toLocation();
    auto bcc = builder->createConvert(loc, builder->getI1Type(), cond);
    builder->create<mlir::CondBranchOp>(loc, bcc, trueTarget, llvm::None,
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
    auto cond =
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
      builder->create<mlir::ReturnOp>(toLocation());
  }
  void genFIR(const Fortran::parser::EndProgramStmt &) { genExitRoutine(); }

  /// END of procedure-like constructs
  ///
  /// Generate the cleanup block before the procedure exits
  void genReturnSymbol(const Fortran::semantics::Symbol &functionSymbol) {
    const auto &resultSym =
        functionSymbol.get<Fortran::semantics::SubprogramDetails>().result();
    auto resultSymBox = lookupSymbol(resultSym);
    auto loc = toLocation();
    if (!resultSymBox) {
      mlir::emitError(loc, "failed lowering function return");
      return;
    }
    auto resultVal = resultSymBox.match(
        [&](const fir::CharBoxValue &x) -> mlir::Value {
          return Fortran::lower::CharacterExprHelper{*builder, loc}
              .createEmboxChar(x.getBuffer(), x.getLen());
        },
        [&](const auto &) -> mlir::Value {
          auto resultRef = resultSymBox.getAddr();
          auto resultType = genType(resultSym);
          mlir::Type resultRefType = builder->getRefType(resultType);
          // A function with multiple entry points returning different types
          // tags all result variables with one of the largest types to allow
          // them to share the same storage.  Convert this to the actual type.
          if (resultRef.getType() != resultRefType)
            resultRef = builder->createConvert(loc, resultRefType, resultRef);
          // Derived types are return by reference (they are passed by the
          // caller)
          if (resultType.isa<fir::RecordType>())
            return resultRef;
          return builder->create<fir::LoadOp>(loc, resultRef);
        });
    builder->create<mlir::ReturnOp>(loc, resultVal);
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
    if (auto *finalBlock = funit.finalBlock) {
      // The current block must end with a terminator.
      if (blockIsUnterminated())
        builder->create<mlir::BranchOp>(toLocation(), finalBlock);
      // Set insertion point to final block.
      builder->setInsertionPoint(finalBlock, finalBlock->end());
    }
    if (Fortran::semantics::IsFunction(symbol)) {
      genReturnSymbol(symbol);
    } else if (Fortran::semantics::HasAlternateReturns(symbol)) {
      mlir::Value retval = builder->create<fir::LoadOp>(
          toLocation(), getAltReturnResult(symbol));
      builder->create<mlir::ReturnOp>(toLocation(), retval);
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
    auto loc = toLocation();
    Fortran::lower::StatementContext stmtCtx;
    auto condExpr = createFIRExpr(
        loc,
        Fortran::semantics::GetExpr(
            std::get<Fortran::parser::ScalarLogicalExpr>(stmt->t)),
        stmtCtx);
    stmtCtx.finalize();
    auto cond = builder->createConvert(loc, builder->getI1Type(), condExpr);
    if (negate)
      cond = builder->create<mlir::XOrOp>(
          loc, cond, builder->createIntegerConstant(loc, cond.getType(), 1));
    return cond;
  }

  mlir::FuncOp getFunc(llvm::StringRef name, mlir::FunctionType ty) {
    if (auto func = builder->getNamedFunction(name)) {
      assert(func.getType() == ty);
      return func;
    }
    return builder->createFunction(toLocation(), name, ty);
  }

  /// Lowering of CALL statement
  void genFIR(const Fortran::parser::CallStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    auto &eval = getEval();
    setCurrentPosition(stmt.v.source);
    assert(stmt.typedCall && "Call was not analyzed");
    // Call statement lowering shares code with function call lowering.
    Fortran::semantics::SomeExpr expr{*stmt.typedCall};
    auto res = createFIRExpr(toLocation(), &expr, stmtCtx);
    if (!res)
      return; // "Normal" subroutine call.
    // Call with alternate return specifiers.
    // The call returns an index that selects an alternate return branch target.
    llvm::SmallVector<int64_t> indexList;
    llvm::SmallVector<mlir::Block *> blockList;
    int64_t index = 0;
    for (const auto &arg :
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
    auto &eval = getEval();
    auto selectExpr =
        createFIRExpr(toLocation(),
                      Fortran::semantics::GetExpr(
                          std::get<Fortran::parser::ScalarIntExpr>(stmt.t)),
                      stmtCtx);
    stmtCtx.finalize();
    llvm::SmallVector<int64_t> indexList;
    llvm::SmallVector<mlir::Block *> blockList;
    int64_t index = 0;
    for (auto &label : std::get<std::list<Fortran::parser::Label>>(stmt.t)) {
      indexList.push_back(++index);
      blockList.push_back(blockOfLabel(eval, label));
    }
    blockList.push_back(eval.nonNopSuccessor().block); // default
    builder->create<fir::SelectOp>(toLocation(), selectExpr, indexList,
                                   blockList);
  }

  void genFIR(const Fortran::parser::ArithmeticIfStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    auto &eval = getEval();
    auto expr = createFIRExpr(
        toLocation(),
        Fortran::semantics::GetExpr(std::get<Fortran::parser::Expr>(stmt.t)),
        stmtCtx);
    stmtCtx.finalize();
    auto exprType = expr.getType();
    auto loc = toLocation();
    if (exprType.isSignlessInteger()) {
      // Arithmetic expression has Integer type.  Generate a SelectCaseOp
      // with ranges {(-inf:-1], 0=default, [1:inf)}.
      MLIRContext *context = builder->getContext();
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
    auto sum = builder->create<mlir::AddFOp>(loc, expr, expr);
    auto zero = builder->create<mlir::ConstantOp>(
        loc, exprType, builder->getFloatAttr(exprType, 0.0));
    auto cond1 =
        builder->create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OLT, sum, zero);
    auto *elseIfBlock =
        builder->getBlock()->splitBlock(builder->getInsertionPoint());
    genFIRConditionalBranch(cond1, blockOfLabel(eval, std::get<1>(stmt.t)),
                            elseIfBlock);
    startBlock(elseIfBlock);
    auto cond2 =
        builder->create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OGT, sum, zero);
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

    auto loc = toLocation();
    auto &eval = getEval();
    const auto &symbolLabelMap =
        eval.getOwningProcedure()->assignSymbolLabelMap;
    const auto &symbol = *std::get<Fortran::parser::Name>(stmt.t).symbol;
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
    for (auto &label : std::get<std::list<Fortran::parser::Label>>(stmt.t)) {
      if (labelSet.count(label) &&
          std::find(indexList.begin(), indexList.end(), label) ==
              indexList.end()) { // ignore duplicates
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
    for (const auto &control :
         std::get<std::list<Fortran::parser::ConcurrentControl>>(header.t))
      incrementLoopNestInfo.emplace_back(
          *std::get<0>(control.t).symbol, std::get<1>(control.t),
          std::get<2>(control.t), std::get<3>(control.t), true);
    auto &info = incrementLoopNestInfo.back();
    info.maskExpr = Fortran::semantics::GetExpr(
        std::get<std::optional<Fortran::parser::ScalarLogicalExpr>>(header.t));
    for (const auto &x : localityList) {
      if (const auto *localInitList =
              std::get_if<Fortran::parser::LocalitySpec::LocalInit>(&x.u))
        for (const auto &x : localInitList->v)
          info.localInitSymList.push_back(x.symbol);
      if (std::get_if<Fortran::parser::LocalitySpec::Local>(&x.u))
        TODO(toLocation(), "do concurrent locality specs not implemented");
    }
    return incrementLoopNestInfo;
  }

  /// Generate FIR for a DO construct.  There are six variants:
  ///  - unstructured infinite and while loops
  ///  - structured and unstructured increment loops
  ///  - structured and unstructured concurrent loops
  void genFIR(const Fortran::parser::DoConstruct &) {
    // Collect loop nest information.
    // Generate begin loop code directly for infinite and while loops.
    auto &eval = getEval();
    bool unstructuredContext = eval.lowerAsUnstructured();
    auto &doStmtEval = eval.getFirstNestedEvaluation();
    auto *doStmt = doStmtEval.getIf<Fortran::parser::NonLabelDoStmt>();
    const auto &loopControl =
        std::get<std::optional<Fortran::parser::LoopControl>>(doStmt->t);
    auto *preheaderBlock = doStmtEval.block;
    auto *beginBlock = preheaderBlock ? preheaderBlock : builder->getBlock();
    auto createNextBeginBlock = [&]() {
      // Step beginBlock through unstructured preheader, header, and mask
      // blocks, created in outermost to innermost order.
      return beginBlock = beginBlock->splitBlock(beginBlock->end());
    };
    auto *headerBlock = unstructuredContext ? createNextBeginBlock() : nullptr;
    auto *bodyBlock = doStmtEval.lexicalSuccessor->block;
    auto *exitBlock = doStmtEval.parentConstruct->constructExit->block;
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
      startBlock(headerBlock);
      genFIRConditionalBranch(*whileCondition, bodyBlock, exitBlock);
    } else if (const auto *bounds =
                   std::get_if<Fortran::parser::LoopControl::Bounds>(
                       &loopControl->u)) {
      // Non-concurrent increment loop.
      auto &info = incrementLoopNestInfo.emplace_back(
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
        for (auto &info : incrementLoopNestInfo) {
          // The original loop body provides the body and latch blocks of the
          // innermost dimension.  The (first) body block of a non-innermost
          // dimension is the preheader block of the immediately enclosed
          // dimension.  The latch block of a non-innermost dimension is the
          // exit block of the immediately enclosed dimension.
          auto createNextExitBlock = [&]() {
            // Create unstructured loop exit blocks, outermost to innermost.
            return exitBlock = insertBlock(exitBlock);
          };
          auto isInnermost = &info == &incrementLoopNestInfo.back();
          auto isOutermost = &info == &incrementLoopNestInfo.front();
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
    for (auto &e : eval.getNestedEvaluations())
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
    auto loc = toLocation();
    auto controlType = incrementLoopNestInfo[0].isStructured()
                           ? builder->getIndexType()
                           : genType(incrementLoopNestInfo[0].loopVariableSym);
    auto hasRealControl = incrementLoopNestInfo[0].hasRealControl;
    auto genControlValue = [&](const Fortran::semantics::SomeExpr *expr) {
      Fortran::lower::StatementContext stmtCtx;
      if (expr)
        return builder->createConvert(loc, controlType,
                                      createFIRExpr(loc, expr, stmtCtx));
      if (hasRealControl)
        return builder->createRealConstant(loc, controlType, 1u);
      return builder->createIntegerConstant(loc, controlType, 1); // step
    };
    auto genLocalInitAssignments = [&](IncrementLoopInfo &info) {
      for (const auto *sym : info.localInitSymList) {
        const auto *hostDetails =
            sym->detailsIf<Fortran::semantics::HostAssocDetails>();
        assert(hostDetails && "missing local_init variable host variable");
        [[maybe_unused]] const Fortran::semantics::Symbol &hostSym =
            hostDetails->symbol();
        TODO(loc, "do concurrent locality specs not implemented");
        // assign sym = hostSym
      }
    };
    for (auto &info : incrementLoopNestInfo) {
      info.loopVariable = createTemp(loc, info.loopVariableSym);
      auto lowerValue = genControlValue(info.lowerExpr);
      auto upperValue = genControlValue(info.upperExpr);
      info.stepValue = genControlValue(info.stepExpr);

      // Structured loop - generate fir.do_loop.
      if (info.isStructured()) {
        info.doLoop = builder->create<fir::DoLoopOp>(
            loc, lowerValue, upperValue, info.stepValue, info.isUnordered,
            /*finalCountValue*/ !info.isUnordered);
        builder->setInsertionPointToStart(info.doLoop.getBody());
        // Update the loop variable value, as it may have non-index references.
        auto value = builder->createConvert(loc, genType(info.loopVariableSym),
                                            info.doLoop.getInductionVar());
        builder->create<fir::StoreOp>(loc, value, info.loopVariable);
        if (info.maskExpr) {
          Fortran::lower::StatementContext stmtCtx;
          auto maskCond = createFIRExpr(loc, info.maskExpr, stmtCtx);
          stmtCtx.finalize();
          auto ifOp = builder->create<fir::IfOp>(loc, maskCond,
                                                 /*withElseRegion=*/false);
          builder->setInsertionPointToStart(&ifOp.thenRegion().front());
        }
        genLocalInitAssignments(info);
        continue;
      }

      // Unstructured loop preheader - initialize tripVariable and loopVariable.
      mlir::Value tripCount;
      if (info.hasRealControl) {
        auto diff1 = builder->create<mlir::SubFOp>(loc, upperValue, lowerValue);
        auto diff2 = builder->create<mlir::AddFOp>(loc, diff1, info.stepValue);
        tripCount = builder->create<mlir::DivFOp>(loc, diff2, info.stepValue);
        controlType = builder->getIndexType();
        tripCount = builder->createConvert(loc, controlType, tripCount);
      } else {
        auto diff1 = builder->create<mlir::SubIOp>(loc, upperValue, lowerValue);
        auto diff2 = builder->create<mlir::AddIOp>(loc, diff1, info.stepValue);
        tripCount =
            builder->create<mlir::SignedDivIOp>(loc, diff2, info.stepValue);
      }
      if (fir::isAlwaysExecuteLoopBody()) { // minimum tripCount is 1
        auto one = builder->createIntegerConstant(loc, controlType, 1);
        auto cond = builder->create<mlir::CmpIOp>(loc, CmpIPredicate::slt,
                                                  tripCount, one);
        tripCount = builder->create<mlir::SelectOp>(loc, cond, one, tripCount);
      }
      info.tripVariable = builder->createTemporary(loc, controlType);
      builder->create<fir::StoreOp>(loc, tripCount, info.tripVariable);
      builder->create<fir::StoreOp>(loc, lowerValue, info.loopVariable);

      // Unstructured loop header - generate loop condition and mask.
      // Note - Currently there is no way to tag a loop as a concurrent loop.
      startBlock(info.headerBlock);
      tripCount = builder->create<fir::LoadOp>(loc, info.tripVariable);
      auto zero = builder->createIntegerConstant(loc, controlType, 0);
      auto cond = builder->create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt,
                                                tripCount, zero);
      if (info.maskExpr) {
        genFIRConditionalBranch(cond, info.maskBlock, info.exitBlock);
        startBlock(info.maskBlock);
        auto latchBlock = getEval().getLastNestedEvaluation().block;
        assert(latchBlock && "missing masked concurrent loop latch block");
        Fortran::lower::StatementContext stmtCtx;
        auto maskCond = createFIRExpr(loc, info.maskExpr, stmtCtx);
        stmtCtx.finalize();
        genFIRConditionalBranch(maskCond, info.bodyBlock, latchBlock);
      } else {
        genFIRConditionalBranch(cond, info.bodyBlock, info.exitBlock);
        if (&info != &incrementLoopNestInfo.back()) // not innermost
          startBlock(info.bodyBlock); // preheader block of enclosed dimension
      }
      if (!info.localInitSymList.empty()) {
        auto insertPt = builder->saveInsertionPoint();
        builder->setInsertionPointToStart(info.bodyBlock);
        genLocalInitAssignments(info);
        builder->restoreInsertionPoint(insertPt);
      }
    }
  }

  /// Generate FIR to end a structured or unstructured increment loop nest.
  void genFIRIncrementLoopEnd(IncrementLoopNestInfo &incrementLoopNestInfo) {
    assert(!incrementLoopNestInfo.empty() && "empty loop nest");
    auto loc = toLocation();
    for (auto it = incrementLoopNestInfo.rbegin(),
              rend = incrementLoopNestInfo.rend();
         it != rend; ++it) {
      auto &info = *it;
      if (info.isStructured()) {
        // End fir.do_loop.
        if (!info.isUnordered) {
          builder->setInsertionPointToEnd(info.doLoop.getBody());
          mlir::Value result = builder->create<mlir::AddIOp>(
              loc, info.doLoop.getInductionVar(), info.doLoop.step());
          builder->create<fir::ResultOp>(loc, result);
        }
        builder->setInsertionPointAfter(info.doLoop);
        if (info.isUnordered)
          continue;
        // The loop control variable may be used after loop execution.
        auto lcv = builder->createConvert(loc, genType(info.loopVariableSym),
                                          info.doLoop.getResult(0));
        builder->create<fir::StoreOp>(loc, lcv, info.loopVariable);
        continue;
      }

      // Unstructured loop - decrement tripVariable and step loopVariable.
      mlir::Value tripCount =
          builder->create<fir::LoadOp>(loc, info.tripVariable);
      auto tripVarType = info.hasRealControl ? builder->getIndexType()
                                             : genType(info.loopVariableSym);
      auto one = builder->createIntegerConstant(loc, tripVarType, 1);
      tripCount = builder->create<mlir::SubIOp>(loc, tripCount, one);
      builder->create<fir::StoreOp>(loc, tripCount, info.tripVariable);
      mlir::Value value = builder->create<fir::LoadOp>(loc, info.loopVariable);
      if (info.hasRealControl)
        value = builder->create<mlir::AddFOp>(loc, value, info.stepValue);
      else
        value = builder->create<mlir::AddIOp>(loc, value, info.stepValue);
      builder->create<fir::StoreOp>(loc, value, info.loopVariable);

      genFIRBranch(info.headerBlock);
      if (&info != &incrementLoopNestInfo.front()) // not outermost
        startBlock(info.exitBlock); // latch block of enclosing dimension
    }
  }

  /// Generate structured or unstructured FIR for an IF construct.
  /// The initial statement may be either an IfStmt or an IfThenStmt.
  void genFIR(const Fortran::parser::IfConstruct &) {
    auto loc = toLocation();
    auto &eval = getEval();
    if (eval.lowerAsStructured()) {
      // Structured fir.if nest.
      fir::IfOp topIfOp, currentIfOp;
      for (auto &e : eval.getNestedEvaluations()) {
        auto genIfOp = [&](mlir::Value cond) {
          auto ifOp = builder->create<fir::IfOp>(loc, cond, /*withElse=*/true);
          builder->setInsertionPointToStart(&ifOp.thenRegion().front());
          return ifOp;
        };
        if (auto *s = e.getIf<Fortran::parser::IfThenStmt>()) {
          topIfOp = currentIfOp = genIfOp(genIfCondition(s, e.negateCondition));
        } else if (auto *s = e.getIf<Fortran::parser::IfStmt>()) {
          topIfOp = currentIfOp = genIfOp(genIfCondition(s, e.negateCondition));
        } else if (auto *s = e.getIf<Fortran::parser::ElseIfStmt>()) {
          builder->setInsertionPointToStart(&currentIfOp.elseRegion().front());
          currentIfOp = genIfOp(genIfCondition(s));
        } else if (e.isA<Fortran::parser::ElseStmt>()) {
          builder->setInsertionPointToStart(&currentIfOp.elseRegion().front());
        } else if (e.isA<Fortran::parser::EndIfStmt>()) {
          builder->setInsertionPointAfter(topIfOp);
        } else {
          genFIR(e, /*unstructuredContext=*/false);
        }
      }
      return;
    }

    // Unstructured branch sequence.
    for (auto &e : eval.getNestedEvaluations()) {
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
    for (auto &e : getEval().getNestedEvaluations())
      genFIR(e);
  }

  /// Generate FIR for a FORALL statement.
  void genFIR(const Fortran::parser::ForallStmt &forallStmt) {
    auto incrementLoopNestInfo = getConcurrentControl(
        std::get<
            Fortran::common::Indirection<Fortran::parser::ConcurrentHeader>>(
            forallStmt.t)
            .value());
    auto &forallAssignment = std::get<Fortran::parser::UnlabeledStatement<
        Fortran::parser::ForallAssignmentStmt>>(forallStmt.t);
    genFIR(incrementLoopNestInfo, forallAssignment.statement);
  }

  /// Generate FIR for a FORALL construct.
  void genFIR(const Fortran::parser::ForallConstruct &forallConstruct) {
    auto &forallConstructStmt = std::get<
        Fortran::parser::Statement<Fortran::parser::ForallConstructStmt>>(
        forallConstruct.t);
    setCurrentPosition(forallConstructStmt.source);
    auto incrementLoopNestInfo = getConcurrentControl(
        std::get<
            Fortran::common::Indirection<Fortran::parser::ConcurrentHeader>>(
            forallConstructStmt.statement.t)
            .value());
    for (auto &s : std::get<std::list<Fortran::parser::ForallBodyConstruct>>(
             forallConstruct.t)) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::Statement<
                  Fortran::parser::ForallAssignmentStmt> &b) {
                setCurrentPosition(b.source);
                genFIR(incrementLoopNestInfo, b.statement);
              },
              [&](const Fortran::parser::Statement<Fortran::parser::WhereStmt>
                      &b) {
                setCurrentPosition(b.source);
                genFIR(b.statement);
              },
              [&](const Fortran::parser::WhereConstruct &b) { genFIR(b); },
              [&](const Fortran::common::Indirection<
                  Fortran::parser::ForallConstruct> &b) { genFIR(b.value()); },
              [&](const Fortran::parser::Statement<Fortran::parser::ForallStmt>
                      &b) {
                setCurrentPosition(b.source);
                genFIR(b.statement);
              },
          },
          s.u);
    }
  }

  /// Generate FIR for a FORALL assignment statement.
  void genFIR(IncrementLoopNestInfo &incrementLoopNestInfo,
              const Fortran::parser::ForallAssignmentStmt &s) {
    genFIRIncrementLoopBegin(incrementLoopNestInfo);
    mlir::emitWarning(toLocation(), "Forall assignments are not temporized, "
                                    "so may be invalid\n");
    std::visit([&](auto &b) { genFIR(b); }, s.u);
    genFIRIncrementLoopEnd(incrementLoopNestInfo);
  }

  void genFIR(const Fortran::parser::CompilerDirective &) {
    mlir::emitWarning(toLocation(), "ignoring all compiler directives");
  }

  void genFIR(const Fortran::parser::OpenACCConstruct &acc) {
    auto insertPt = builder->saveInsertionPoint();
    genOpenACCConstruct(*this, getEval(), acc);
    for (auto &e : getEval().getNestedEvaluations())
      genFIR(e);
    builder->restoreInsertionPoint(insertPt);
  }

  void genFIR(const Fortran::parser::OpenMPConstruct &omp) {
    auto insertPt = builder->saveInsertionPoint();
    localSymbols.pushScope();
    genOpenMPConstruct(*this, getEval(), omp);
    // If loop is part of an OpenMP Construct then the OpenMP dialect
    // workshare loop operation has already been created. Only the
    // body needs to be created here and the do_loop can be skipped.
    Fortran::lower::pft::Evaluation *curEval =
        std::get_if<Fortran::parser::OpenMPLoopConstruct>(&omp.u)
            ? &getEval().getFirstNestedEvaluation()
            : &getEval();
    for (auto &e : curEval->getNestedEvaluations())
      genFIR(e);
    localSymbols.popScope();
    builder->restoreInsertionPoint(insertPt);
  }

  /// Generate FIR for a SELECT CASE statement.
  /// The type may be CHARACTER, INTEGER, or LOGICAL.
  void genFIR(const Fortran::parser::SelectCaseStmt &stmt) {
    auto &eval = getEval();
    auto *context = builder->getContext();
    auto loc = toLocation();
    Fortran::lower::StatementContext stmtCtx;
    const auto *expr = Fortran::semantics::GetExpr(
        std::get<Fortran::parser::Scalar<Fortran::parser::Expr>>(stmt.t));
    bool isCharSelector = isCharacterCategory(expr->GetType()->category());
    bool isLogicalSelector = isLogicalCategory(expr->GetType()->category());
    auto charValue = [&](const Fortran::lower::SomeExpr *expr) {
      fir::ExtendedValue exv = genExprAddr(*expr, stmtCtx, &loc);
      return exv.match(
          [&](const fir::CharBoxValue &cbv) {
            return Fortran::lower::CharacterExprHelper{*builder, loc}
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
    auto selectType = selector.getType();
    llvm::SmallVector<mlir::Attribute> attrList;
    llvm::SmallVector<mlir::Value> valueList;
    llvm::SmallVector<mlir::Block *> blockList;
    auto *defaultBlock = eval.parentConstruct->constructExit->block;
    using CaseValue = Fortran::parser::Scalar<Fortran::parser::ConstantExpr>;
    auto addValue = [&](const CaseValue &caseValue) {
      const auto *expr = Fortran::semantics::GetExpr(caseValue.thing);
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
      for (auto &caseValueRange : *caseValueRangeList) {
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
    stmtCtx.finalize();

    // Generate a fir::SelectCaseOp.
    // Explicit branch code is better for the LOGICAL type.  The CHARACTER type
    // does not yet have downstream support, and also uses explicit branch code.
    // The -no-structured-fir option can be used to force generation of INTEGER
    // type branch code.
    if (!isLogicalSelector && !isCharSelector && eval.lowerAsStructured()) {
      builder->create<fir::SelectCaseOp>(loc, selector, attrList, valueList,
                                         blockList);
      return;
    }

    // Generate a sequence of case value comparisons and branches.
    auto caseValue = valueList.begin();
    auto caseBlock = blockList.begin();
    for (auto attr : attrList) {
      if (attr.isa<mlir::UnitAttr>()) {
        genFIRBranch(*caseBlock++);
        break;
      }
      auto genCond = [&](mlir::Value rhs,
                         mlir::CmpIPredicate pred) -> mlir::Value {
        if (!isCharSelector)
          return builder->create<mlir::CmpIOp>(loc, pred, selector, rhs);
        Fortran::lower::CharacterExprHelper charHelper{*builder, loc};
        auto [lhsAddr, lhsLen] = charHelper.createUnboxChar(selector);
        auto [rhsAddr, rhsLen] = charHelper.createUnboxChar(rhs);
        return Fortran::lower::genRawCharCompare(*builder, loc, pred, lhsAddr,
                                                 lhsLen, rhsAddr, rhsLen);
      };
      auto *newBlock = insertBlock(*caseBlock);
      if (attr.isa<fir::ClosedIntervalAttr>()) {
        auto *newBlock2 = insertBlock(*caseBlock);
        auto cond = genCond(*caseValue++, mlir::CmpIPredicate::sge);
        genFIRConditionalBranch(cond, newBlock, newBlock2);
        builder->setInsertionPointToEnd(newBlock);
        auto cond2 = genCond(*caseValue++, mlir::CmpIPredicate::sle);
        genFIRConditionalBranch(cond2, *caseBlock++, newBlock2);
        builder->setInsertionPointToEnd(newBlock2);
        continue;
      }
      mlir::CmpIPredicate pred;
      if (attr.isa<fir::PointIntervalAttr>()) {
        pred = mlir::CmpIPredicate::eq;
      } else if (attr.isa<fir::LowerBoundAttr>()) {
        pred = mlir::CmpIPredicate::sge;
      } else {
        assert(attr.isa<fir::UpperBoundAttr>() && "unexpected predicate");
        pred = mlir::CmpIPredicate::sle;
      }
      auto cond = genCond(*caseValue++, pred);
      genFIRConditionalBranch(cond, *caseBlock++, newBlock);
      builder->setInsertionPointToEnd(newBlock);
    }
    assert(caseValue == valueList.end() && caseBlock == blockList.end() &&
           "select case list mismatch");
  }

  fir::ExtendedValue
  genAssociateSelector(const Fortran::semantics::SomeExpr &selector,
                       Fortran::lower::StatementContext &stmtCtx) {
    return isArraySectionWithoutVectorSubscript(selector)
               ? Fortran::lower::createSomeArrayBox(*this, selector,
                                                    localSymbols, stmtCtx)
               : genExprAddr(selector, stmtCtx);
  }

  void genFIR(const Fortran::parser::AssociateConstruct &) {
    Fortran::lower::StatementContext stmtCtx;
    for (auto &e : getEval().getNestedEvaluations()) {
      if (auto *stmt = e.getIf<Fortran::parser::AssociateStmt>()) {
        localSymbols.pushScope();
        for (auto &assoc :
             std::get<std::list<Fortran::parser::Association>>(stmt->t)) {
          auto &sym = *std::get<Fortran::parser::Name>(assoc.t).symbol;
          const auto &selector =
              *sym.get<Fortran::semantics::AssocEntityDetails>().expr();
          localSymbols.addSymbol(sym, genAssociateSelector(selector, stmtCtx));
        }
      } else if (e.getIf<Fortran::parser::EndAssociateStmt>()) {
        stmtCtx.finalize();
        localSymbols.popScope();
      } else {
        genFIR(e);
      }
    }
  }

  void genFIR(const Fortran::parser::BlockConstruct &) {
    TODO(toLocation(), "BlockConstruct lowering");
  }
  void genFIR(const Fortran::parser::BlockStmt &) {
    TODO(toLocation(), "BlockStmt lowering");
  }
  void genFIR(const Fortran::parser::EndBlockStmt &) {
    TODO(toLocation(), "EndBlockStmt lowering");
  }

  void genFIR(const Fortran::parser::ChangeTeamConstruct &construct) {
    genChangeTeamConstruct(*this, getEval(), construct);
  }
  void genFIR(const Fortran::parser::ChangeTeamStmt &stmt) {
    genChangeTeamStmt(*this, getEval(), stmt);
  }
  void genFIR(const Fortran::parser::EndChangeTeamStmt &stmt) {
    genEndChangeTeamStmt(*this, getEval(), stmt);
  }

  void genFIR(const Fortran::parser::CriticalConstruct &) {
    TODO(toLocation(), "CriticalConstruct lowering");
  }
  void genFIR(const Fortran::parser::CriticalStmt &) {
    TODO(toLocation(), "CriticalStmt lowering");
  }
  void genFIR(const Fortran::parser::EndCriticalStmt &) {
    TODO(toLocation(), "EndCriticalStmt lowering");
  }

  void genFIR(const Fortran::parser::SelectRankConstruct &) {
    TODO(toLocation(), "SelectRankConstruct lowering");
  }
  void genFIR(const Fortran::parser::SelectRankStmt &) {
    TODO(toLocation(), "SelectRankStmt lowering");
  }
  void genFIR(const Fortran::parser::SelectRankCaseStmt &) {
    TODO(toLocation(), "SelectRankCaseStmt lowering");
  }

  void genFIR(const Fortran::parser::SelectTypeConstruct &) {
    TODO(toLocation(), "SelectTypeConstruct lowering");
  }
  void genFIR(const Fortran::parser::SelectTypeStmt &) {
    TODO(toLocation(), "SelectTypeStmt lowering");
  }
  void genFIR(const Fortran::parser::TypeGuardStmt &) {
    TODO(toLocation(), "TypeGuardStmt lowering");
  }

  //===--------------------------------------------------------------------===//
  // IO statements (see io.h)
  //===--------------------------------------------------------------------===//

  void genFIR(const Fortran::parser::BackspaceStmt &stmt) {
    auto iostat = genBackspaceStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::CloseStmt &stmt) {
    auto iostat = genCloseStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::EndfileStmt &stmt) {
    auto iostat = genEndfileStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::FlushStmt &stmt) {
    auto iostat = genFlushStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::InquireStmt &stmt) {
    auto iostat = genInquireStatement(*this, stmt);
    if (const auto *specs =
            std::get_if<std::list<Fortran::parser::InquireSpec>>(&stmt.u))
      genIoConditionBranches(getEval(), *specs, iostat);
  }
  void genFIR(const Fortran::parser::OpenStmt &stmt) {
    auto iostat = genOpenStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::PrintStmt &stmt) {
    genPrintStatement(*this, stmt);
  }
  void genFIR(const Fortran::parser::ReadStmt &stmt) {
    auto iostat = genReadStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.controls, iostat);
  }
  void genFIR(const Fortran::parser::RewindStmt &stmt) {
    auto iostat = genRewindStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::WaitStmt &stmt) {
    auto iostat = genWaitStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::WriteStmt &stmt) {
    auto iostat = genWriteStatement(*this, stmt);
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

    auto loc = toLocation();
    auto indexType = builder->getIndexType();
    auto selector = builder->createConvert(loc, indexType, iostat);
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
    auto loc = toLocation();
    for (auto &po : stmt.v) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::Name &name) {
                assert(name.symbol);
                auto pointer = lookupSymbol(*name.symbol);
                pointer.match(
                    [&](const fir::MutableBoxValue &box) {
                      Fortran::lower::disassociateMutableBox(*builder, loc,
                                                             box);
                    },
                    [&](const auto &) {
                      fir::emitFatalError(
                          loc,
                          "entity in nullify was not lowered as a pointer");
                    });
              },
              [&](const Fortran::parser::StructureComponent &) {
                TODO(loc, "StructureComponent NullifyStmt lowering");
              },
          },
          po.u);
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
    return createSomeInitializerExpression(toLocation(), *this, expr,
                                           localSymbols, stmtCtx);
  }

  /// Generate an array assignment.
  /// This is an assignment expression with rank > 0. The assignment may or may
  /// not be in a WHERE context.
  void genArrayAssignment(const Fortran::evaluate::Assignment &assign,
                          Fortran::lower::StatementContext &stmtCtx) {
    if (masks.empty()) {
      // No masks, so create a simple array assignment.
      createSomeArrayAssignment(*this, assign.lhs, assign.rhs, localSymbols,
                                stmtCtx);
      return;
    }

    // Generate a masked array assignment.
    createMaskedArrayAssignment(*this, assign.lhs, assign.rhs, masks,
                                localSymbols, masks.stmtCtx);
  }

  static bool isArraySectionWithoutVectorSubscript(
      const Fortran::semantics::SomeExpr &expr) {
    return expr.Rank() > 0 && Fortran::evaluate::IsVariable(expr) &&
           !Fortran::evaluate::UnwrapWholeSymbolDataRef(expr) &&
           !Fortran::evaluate::HasVectorSubscript(expr);
  }

  // Recursively assign members of a record type.
  void genRecordAssignment(const fir::ExtendedValue &lhs,
                           const fir::ExtendedValue &rhs,
                           Fortran::lower::StatementContext &stmtCtx) {
    auto loc = genLocation();
    auto lhsTy = fir::dyn_cast_ptrEleTy(fir::getBase(lhs).getType())
                     .dyn_cast<fir::RecordType>();
    assert(lhsTy && "must be a record type");
    auto fieldTy = fir::FieldType::get(lhsTy.getContext());
    for (auto [fldName, fldType] : lhsTy.getTypeList()) {
      if (fir::isa_char(fldType)) {
        auto fldCharTy = fldType.cast<fir::CharacterType>();
        if (!fldCharTy.hasConstantLen())
          TODO(loc, "LEN type parameter not constant");
        mlir::Value field = builder->create<fir::FieldIndexOp>(
            loc, fieldTy, fldName, lhsTy, fir::getTypeParams(lhs));
        auto fldRefTy = builder->getRefType(fldType);
        auto lenVal = builder->createIntegerConstant(loc, builder->getI64Type(),
                                                     fldCharTy.getLen());
        mlir::Value from = builder->create<fir::CoordinateOp>(
            loc, fldRefTy, fir::getBase(rhs), field);
        fir::ExtendedValue fromPtr{fir::CharBoxValue{from, lenVal}};
        mlir::Value to = builder->create<fir::CoordinateOp>(
            loc, fldRefTy, fir::getBase(lhs), field);
        fir::ExtendedValue toPtr{fir::CharBoxValue{to, lenVal}};
        Fortran::lower::CharacterExprHelper{*builder, loc}.createAssign(
            toPtr, fromPtr);
        continue;
      }
      if (!fir::isa_trivial(fldType))
        TODO(toLocation(), "derived type assignment of non-trivial member");
      mlir::Value field = builder->create<fir::FieldIndexOp>(
          loc, fieldTy, fldName, lhsTy, fir::getTypeParams(lhs));
      auto fldRefTy = builder->getRefType(fldType);
      auto elePtr = builder->create<fir::CoordinateOp>(
          loc, fldRefTy, fir::getBase(rhs), field);
      auto loadVal = builder->create<fir::LoadOp>(loc, elePtr);
      auto toPtr = builder->create<fir::CoordinateOp>(loc, fldRefTy,
                                                      fir::getBase(lhs), field);
      builder->create<fir::StoreOp>(loc, loadVal, toPtr);
    }
  }

  /// Shared for both assignments and pointer assignments.
  void genAssignment(const Fortran::evaluate::Assignment &assign) {
    Fortran::lower::StatementContext stmtCtx;
    auto loc = toLocation();
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::Assignment::Intrinsic &) {
              const auto *sym =
                  Fortran::evaluate::UnwrapWholeSymbolDataRef(assign.lhs);
              // Assignment of allocatable are more complex, the lhs may need to
              // be deallocated/reallocated. See Fortran 2018 10.2.1.3 p3
              const bool isHeap =
                  sym && Fortran::semantics::IsAllocatable(*sym);
              if (isHeap) {
                TODO(loc, "assignment to allocatable not implemented");
              }
              // Nothing to do for pointers, the target will be assigned.
              // as per 2018 10.2.1.3 p2. genExprAddr on a pointer returns
              // the target address.
              if (assign.lhs.Rank() > 0) {
                // Array assignment
                // See Fortran 2018 10.2.1.3 p5, p6, and p7
                genArrayAssignment(assign, stmtCtx);
                return;
              }

              // Scalar assignment
              auto lhsType = assign.lhs.GetType();
              assert(lhsType && "lhs cannot be typeless");
              if (isNumericScalarCategory(lhsType->category())) {
                // Fortran 2018 10.2.1.3 p8 and p9
                // Conversions should have been inserted by semantic analysis,
                // but they can be incorrect between the rhs and lhs. Correct
                // that here.
                auto addr = fir::getBase(genExprAddr(assign.lhs, stmtCtx));
                auto val = createFIRExpr(loc, &assign.rhs, stmtCtx);
                // A function with multiple entry points returning different
                // types tags all result variables with one of the largest
                // types to allow them to share the same storage.  Assignment
                // to a result variable of one of the other types requires
                // conversion to the actual type.
                auto toTy = genType(assign.lhs);
                auto cast = builder->convertWithSemantics(loc, toTy, val);
                if (fir::dyn_cast_ptrEleTy(addr.getType()) != toTy) {
                  assert(sym->IsFuncResult() && "type mismatch");
                  addr = builder->createConvert(
                      toLocation(), builder->getRefType(toTy), addr);
                }
                builder->create<fir::StoreOp>(loc, cast, addr);
                return;
              }
              if (isCharacterCategory(lhsType->category())) {
                // Fortran 2018 10.2.1.3 p10 and p11
                auto lhs = genExprAddr(assign.lhs, stmtCtx);
                // Current character assignment only works with in memory
                // characters since !fir.array<> cannot be addressed with
                // fir.coordinate_of without being inside a !fir.ref<> or other
                // memory types. So use genExprAddr for rhs.
                auto rhs = genExprAddr(assign.rhs, stmtCtx);
                Fortran::lower::CharacterExprHelper{*builder, loc}.createAssign(
                    lhs, rhs);
                return;
              }
              if (isDerivedCategory(lhsType->category())) {
                // Fortran 2018 10.2.1.3 p13 and p14
                // Recursively gen an assignment on each element pair.
                auto lhs = genExprAddr(assign.lhs, stmtCtx);
                auto rhs = genExprAddr(assign.rhs, stmtCtx);
                genRecordAssignment(lhs, rhs, stmtCtx);
                return;
              }
              llvm_unreachable("unknown category");
            },
            [&](const Fortran::evaluate::ProcedureRef &procRef) {
              // User defined assignment: call the procedure.
              Fortran::semantics::SomeExpr expr{procRef};
              createFIRExpr(toLocation(), &expr, stmtCtx);
            },
            [&](const Fortran::evaluate::Assignment::BoundsSpec &lbExprs) {
              // Pointer assignment with possibly empty bounds-spec
              auto lhs = genExprMutableBox(loc, assign.lhs);
              if (Fortran::common::Unwrap<Fortran::evaluate::NullPointer>(
                      assign.rhs.u)) {
                Fortran::lower::disassociateMutableBox(*builder, loc, lhs);
                return;
              }
              auto lhsType = assign.lhs.GetType();
              auto rhsType = assign.rhs.GetType();
              // Polymorphic lhs/rhs may need more care. See F2018 10.2.2.3.
              if ((lhsType && lhsType->IsPolymorphic()) ||
                  (rhsType && rhsType->IsPolymorphic()))
                TODO(loc, "pointer assignment involving polymorphic entity");
              llvm::SmallVector<mlir::Value> lbounds;
              for (const auto &lbExpr : lbExprs)
                lbounds.push_back(
                    fir::getBase(genExprValue(toEvExpr(lbExpr), stmtCtx)));

              // Do not generate a temp in case rhs is an array section.
              auto rhs = isArraySectionWithoutVectorSubscript(assign.rhs)
                             ? Fortran::lower::createSomeArrayBox(
                                   *this, assign.rhs, localSymbols, stmtCtx)
                             : genExprAddr(assign.rhs, stmtCtx);
              Fortran::lower::associateMutableBoxWithShift(*builder, loc, lhs,
                                                           rhs, lbounds);
            },
            [&](const Fortran::evaluate::Assignment::BoundsRemapping
                    &boundExprs) {
              // Pointer assignment with bounds-remapping
              auto lhs = genExprMutableBox(loc, assign.lhs);
              if (Fortran::common::Unwrap<Fortran::evaluate::NullPointer>(
                      assign.rhs.u)) {
                Fortran::lower::disassociateMutableBox(*builder, loc, lhs);
                return;
              }
              auto lhsType = assign.lhs.GetType();
              auto rhsType = assign.rhs.GetType();
              // Polymorphic lhs/rhs may need more care. See F2018 10.2.2.3.
              if ((lhsType && lhsType->IsPolymorphic()) ||
                  (rhsType && rhsType->IsPolymorphic()))
                TODO(loc, "pointer assignment involving polymorphic entity");
              llvm::SmallVector<mlir::Value> lbounds;
              llvm::SmallVector<mlir::Value> ubounds;
              for (const auto &[lbExpr, ubExpr] : boundExprs) {
                lbounds.push_back(
                    fir::getBase(genExprValue(toEvExpr(lbExpr), stmtCtx)));
                ubounds.push_back(
                    fir::getBase(genExprValue(toEvExpr(ubExpr), stmtCtx)));
              }
              // Do not generate a temp in case rhs is an array section.
              auto rhs = isArraySectionWithoutVectorSubscript(assign.rhs)
                             ? Fortran::lower::createSomeArrayBox(
                                   *this, assign.rhs, localSymbols, stmtCtx)
                             : genExprAddr(assign.rhs, stmtCtx);
              Fortran::lower::associateMutableBoxWithRemap(
                  *builder, loc, lhs, rhs, lbounds, ubounds);
            },
        },
        assign.u);
  }

  void genFIR(const Fortran::parser::WhereConstruct &c) {
    masks.growStack();
    genFIR(std::get<
               Fortran::parser::Statement<Fortran::parser::WhereConstructStmt>>(
               c.t)
               .statement);
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
    genFIR(
        std::get<Fortran::parser::Statement<Fortran::parser::EndWhereStmt>>(c.t)
            .statement);
  }
  void genFIR(const Fortran::parser::WhereBodyConstruct &body) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Statement<
                Fortran::parser::AssignmentStmt> &stmt) {
              genFIR(stmt.statement);
            },
            [&](const Fortran::parser::Statement<Fortran::parser::WhereStmt>
                    &stmt) { genFIR(stmt.statement); },
            [&](const Fortran::common::Indirection<
                Fortran::parser::WhereConstruct> &c) { genFIR(c.value()); },
        },
        body.u);
  }
  void genFIR(const Fortran::parser::WhereConstructStmt &stmt) {
    masks.append(Fortran::semantics::GetExpr(
        std::get<Fortran::parser::LogicalExpr>(stmt.t)));
  }
  void genFIR(const Fortran::parser::WhereConstruct::MaskedElsewhere &ew) {
    genFIR(
        std::get<
            Fortran::parser::Statement<Fortran::parser::MaskedElsewhereStmt>>(
            ew.t)
            .statement);
    for (const auto &body :
         std::get<std::list<Fortran::parser::WhereBodyConstruct>>(ew.t))
      genFIR(body);
  }
  void genFIR(const Fortran::parser::MaskedElsewhereStmt &stmt) {
    masks.append(Fortran::semantics::GetExpr(
        std::get<Fortran::parser::LogicalExpr>(stmt.t)));
  }
  void genFIR(const Fortran::parser::WhereConstruct::Elsewhere &ew) {
    genFIR(std::get<Fortran::parser::Statement<Fortran::parser::ElsewhereStmt>>(
               ew.t)
               .statement);
    for (const auto &body :
         std::get<std::list<Fortran::parser::WhereBodyConstruct>>(ew.t))
      genFIR(body);
  }
  void genFIR(const Fortran::parser::ElsewhereStmt &stmt) {
    masks.append(nullptr);
  }
  void genFIR(const Fortran::parser::EndWhereStmt &) { masks.shrinkStack(); }

  void genFIR(const Fortran::parser::WhereStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    const auto &assign = std::get<Fortran::parser::AssignmentStmt>(stmt.t);
    masks.growStack();
    masks.append(Fortran::semantics::GetExpr(
        std::get<Fortran::parser::LogicalExpr>(stmt.t)));
    genAssignment(*assign.typedAssignment->v);
    masks.shrinkStack();
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
    const auto &symbol = *std::get<Fortran::parser::Name>(stmt.t).symbol;
    auto loc = toLocation();
    const auto labelValue = builder->createIntegerConstant(
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

  void genFIR(const Fortran::parser::NamelistStmt &) {
    TODO(toLocation(), "NamelistStmt lowering");
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
    auto *funit = getEval().getOwningProcedure();
    assert(funit && "not inside main program, function or subroutine");
    if (funit->isMainProgram()) {
      genExitRoutine();
      return;
    }
    auto loc = toLocation();
    if (stmt.v) {
      // Alternate return statement - If this is a subroutine where some
      // alternate entries have alternate returns, but the active entry point
      // does not, ignore the alternate return value.  Otherwise, assign it
      // to the compiler-generated result variable.
      const auto &symbol = funit->getSubprogramSymbol();
      if (Fortran::semantics::HasAlternateReturns(symbol)) {
        Fortran::lower::StatementContext stmtCtx;
        const auto *expr = Fortran::semantics::GetExpr(*stmt.v);
        assert(expr && "missing alternate return expression");
        auto altReturnIndex = builder->createConvert(
            loc, builder->getIndexType(), createFIRExpr(loc, expr, stmtCtx));
        builder->create<fir::StoreOp>(loc, altReturnIndex,
                                      getAltReturnResult(symbol));
      }
    }
    // Branch to the last block of the SUBROUTINE, which has the actual return.
    if (!funit->finalBlock) {
      const auto insPt = builder->saveInsertionPoint();
      funit->finalBlock = builder->createBlock(&builder->getRegion());
      builder->restoreInsertionPoint(insPt);
    }
    builder->create<mlir::BranchOp>(loc, funit->finalBlock);
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
  void genFIR(const Fortran::parser::AssociateStmt &) {}        // nop
  void genFIR(const Fortran::parser::CaseStmt &) {}             // nop
  void genFIR(const Fortran::parser::ContinueStmt &) {}         // nop
  void genFIR(const Fortran::parser::ElseIfStmt &) {}           // nop
  void genFIR(const Fortran::parser::ElseStmt &) {}             // nop
  void genFIR(const Fortran::parser::EndAssociateStmt &) {}     // nop
  void genFIR(const Fortran::parser::EndDoStmt &) {}            // nop
  void genFIR(const Fortran::parser::EndForallStmt &) {}        // nop
  void genFIR(const Fortran::parser::EndFunctionStmt &) {}      // nop
  void genFIR(const Fortran::parser::EndIfStmt &) {}            // nop
  void genFIR(const Fortran::parser::EndMpSubprogramStmt &) {}  // nop
  void genFIR(const Fortran::parser::EndSelectStmt &) {}        // nop
  void genFIR(const Fortran::parser::EndSubroutineStmt &) {}    // nop
  void genFIR(const Fortran::parser::EntryStmt &) {}            // nop
  void genFIR(const Fortran::parser::ForallAssignmentStmt &) {} // nop
  void genFIR(const Fortran::parser::ForallConstructStmt &) {}  // nop
  void genFIR(const Fortran::parser::IfStmt &) {}               // nop
  void genFIR(const Fortran::parser::IfThenStmt &) {}           // nop
  void genFIR(const Fortran::parser::NonLabelDoStmt &) {}       // nop
  void genFIR(const Fortran::parser::OmpEndLoopDirective &) {}  // nop

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
      if (successor && successor->block)
        genFIRBranch(successor->block);
    }
  }

  /// Map mlir function block arguments to the corresponding Fortran dummy
  /// variables. When the result is passed as a hidden argument, the Fortran
  /// result is also mapped. The symbol map is used to hold this mapping.
  void mapDummiesAndResults(const Fortran::lower::pft::FunctionLikeUnit &funit,
                            const Fortran::lower::CalleeInterface &callee) {
    assert(builder && "need a builder at this point");
    using PassBy = Fortran::lower::CalleeInterface::PassEntityBy;
    auto mapPassedEntity = [&](const auto arg) -> void {
      if (arg.passBy == PassBy::AddressAndLength) {
        // TODO: now that fir call has some attributes regarding character
        // return, this should PassBy::AddressAndLength should be retired.
        auto loc = toLocation();
        Fortran::lower::CharacterExprHelper charHelp{*builder, loc};
        auto box = charHelp.createEmboxChar(arg.firArgument, arg.firLength);
        addSymbol(arg.entity.get(), box);
      } else {
        addSymbol(arg.entity.get(), arg.firArgument);
      }
    };
    for (const auto &arg : callee.getPassedArguments())
      mapPassedEntity(arg);

    // Allocate local skeleton instances of dummies from other entry points.
    // Most of these locals will not survive into final generated code, but
    // some will.  It is illegal to reference them at run time if they do.
    for (const auto *arg : funit.nonUniversalDummyArguments) {
      if (lookupSymbol(*arg))
        continue;
      auto type = genType(*arg);
      // TODO: Account for VALUE arguments (and possibly other variants).
      type = builder->getRefType(type);
      addSymbol(*arg, builder->create<fir::UndefOp>(toLocation(), type));
    }
    if (auto passedResult = callee.getPassedResult()) {
      mapPassedEntity(*passedResult);
      // FIXME: need to make sure things are OK here. addSymbol is may not be OK
      if (funit.primaryResult &&
          passedResult->entity.get() != *funit.primaryResult)
        addSymbol(*funit.primaryResult, getSymbolAddress(passedResult->entity));
    }
  }

  /// Instantiate variable \p var and add it to the symbol map.
  /// See ConvertVariable.cpp.
  void instantiateVar(const Fortran::lower::pft::Variable &var,
                      Fortran::lower::AggregateStoreMap &storeMap) {
    Fortran::lower::instantiateVariable(*this, var, localSymbols, storeMap);
  }

  /// Prepare to translate a new function
  void startNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    assert(!builder && "expected nullptr");
    Fortran::lower::CalleeInterface callee(funit, *this);
    mlir::FuncOp func = callee.addEntryBlockAndMapArguments();
    builder = new Fortran::lower::FirOpBuilder(func, bridge.getKindMap());
    assert(builder && "FirOpBuilder did not instantiate");
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
    if (auto passedResult = callee.getPassedResult())
      resultArg = lookupSymbol(passedResult->entity.get());

    Fortran::lower::AggregateStoreMap storeMap;
    // The front-end is currently not adding module variables referenced
    // in a module procedure as host associated. As a result we need to
    // instantiate all module variables here if this is a module procedure.
    // It is likely that the front-end behaviour should change here.
    if (auto *module =
            funit.parent.getIf<Fortran::lower::pft::ModuleLikeUnit>())
      for (const auto &var : module->getOrderedSymbolTable())
        instantiateVar(var, storeMap);

    mlir::Value primaryFuncResultStorage;
    for (const auto &var : funit.getOrderedSymbolTable()) {
      if (var.isAggregateStore()) {
        instantiateVar(var, storeMap);
        continue;
      }
      const Fortran::semantics::Symbol &sym = var.getSymbol();
      if (!sym.IsFuncResult() || !funit.primaryResult) {
        instantiateVar(var, storeMap);
      } else if (&sym == funit.primaryResult) {
        instantiateVar(var, storeMap);
        primaryFuncResultStorage = getSymbolAddress(sym);
      } else {
        deferredFuncResultList.push_back(var);
      }
    }

    /// TODO: should use same mechanism as equivalence?
    /// One blocking point is character entry returns that need special handling
    /// since they are not locally allocated but come as argument. CHARACTER(*)
    /// is not something that fit wells with equivalence lowering.
    for (const auto &altResult : deferredFuncResultList) {
      if (auto passedResult = callee.getPassedResult())
        addSymbol(altResult.getSymbol(), resultArg.getAddr());
      Fortran::lower::StatementContext stmtCtx;
      Fortran::lower::mapSymbolAttributes(*this, altResult, localSymbols,
                                          stmtCtx, primaryFuncResultStorage);
    }

    // Create most function blocks in advance.
    auto *alternateEntryEval = funit.getEntryEval();
    if (alternateEntryEval) {
      // Move to executable successor.
      alternateEntryEval = alternateEntryEval->lexicalSuccessor;
      auto evalIsNewBlock = alternateEntryEval->isNewBlock;
      alternateEntryEval->isNewBlock = true;
      createEmptyBlocks(funit.evaluationList);
      alternateEntryEval->isNewBlock = evalIsNewBlock;
    } else {
      createEmptyBlocks(funit.evaluationList);
    }

    // Reinstate entry block as the current insertion point.
    builder->setInsertionPointToEnd(&func.front());

    if (callee.hasAlternateReturns()) {
      // Create a local temp to hold the alternate return index.
      // Give it an integer index type and the subroutine name (for dumps).
      // Attach it to the subroutine symbol in the localSymbols map.
      // Initialize it to zero, the "fallthrough" alternate return value.
      const auto &symbol = funit.getSubprogramSymbol();
      auto loc = toLocation();
      auto idxTy = builder->getIndexType();
      const auto altResult =
          builder->createTemporary(loc, idxTy, toStringRef(symbol.name()));
      addSymbol(symbol, altResult);
      const auto zero = builder->createIntegerConstant(loc, idxTy, 0);
      builder->create<fir::StoreOp>(loc, zero, altResult);
    }

    if (alternateEntryEval) {
      genFIRBranch(alternateEntryEval->block);
      builder->setInsertionPointToStart(
          builder->createBlock(&builder->getRegion()));
    }
  }

  /// Create empty blocks for the current function.
  void createEmptyBlocks(
      std::list<Fortran::lower::pft::Evaluation> &evaluationList) {
    auto *region = &builder->getRegion();
    for (auto &eval : evaluationList) {
      if (eval.isNewBlock)
        eval.block = builder->createBlock(region);
      if (eval.isConstruct() || eval.isDirective()) {
        if (eval.lowerAsUnstructured()) {
          createEmptyBlocks(eval.getNestedEvaluations());
        } else if (eval.hasNestedEvaluations()) {
          // A structured construct that is a target starts a new block.
          auto &constructStmt = eval.getFirstNestedEvaluation();
          if (constructStmt.isNewBlock)
            constructStmt.block = builder->createBlock(region);
        }
      }
    }
  }

  /// Return the predicate: "current block does not have a terminator branch".
  bool blockIsUnterminated() {
    auto *currentBlock = builder->getBlock();
    return currentBlock->empty() ||
           !currentBlock->back().hasTrait<mlir::OpTrait::IsTerminator>();
  }

  /// Unconditionally switch code insertion to a new block.
  void startBlock(mlir::Block *newBlock) {
    assert(newBlock && "missing block");
    if (blockIsUnterminated())
      genFIRBranch(newBlock); // default termination is a fallthrough branch
    builder->setInsertionPointToEnd(newBlock); // newBlock might not be empty
  }

  /// Conditionally switch code insertion to a new block.
  void maybeStartBlock(mlir::Block *newBlock) {
    if (newBlock)
      startBlock(newBlock);
  }

  /// Emit return and cleanup after the function has been translated.
  void endNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    setCurrentPosition(
        Fortran::lower::pft::FunctionLikeUnit::stmtSourceLoc(funit.endStmt));
    if (funit.isMainProgram())
      genExitRoutine();
    else
      genFIRProcedureExit(funit, funit.getSubprogramSymbol());
    funit.finalBlock = nullptr;
    // FIXME: Simplification should happen in a normal pass, not here.
    (void)mlir::simplifyRegions({builder->getRegion()}); // remove dead code
    delete builder;
    builder = nullptr;
    localSymbols.clear();
  }

  /// Instantiate the data from a BLOCK DATA unit.
  void lowerBlockData(Fortran::lower::pft::BlockDataUnit &bdunit) {
    // FIXME: get rid of the bogus function context and instantiate the
    // globals directly into the module.
    auto *context = &getMLIRContext();
    auto func = Fortran::lower::FirOpBuilder::createFunction(
        mlir::UnknownLoc::get(context), getModuleOp(),
        fir::NameUniquer::doGenerated("Sham"),
        mlir::FunctionType::get(context, llvm::None, llvm::None));

    builder = new Fortran::lower::FirOpBuilder(func, bridge.getKindMap());
    Fortran::lower::AggregateStoreMap fakeMap;
    for (const auto &[_, sym] : bdunit.symTab) {
      Fortran::lower::pft::Variable var(*sym, true);
      instantiateVar(var, fakeMap);
    }

    if (auto *region = func.getCallableRegion())
      region->dropAllReferences();
    func.erase();
    delete builder;
    builder = nullptr;
    localSymbols.clear();
  }

  /// Lower a procedure (nest).
  void lowerFunc(Fortran::lower::pft::FunctionLikeUnit &funit) {
    for (int entryIndex = 0, last = funit.entryPointList.size();
         entryIndex < last; ++entryIndex) {
      funit.setActiveEntry(entryIndex);
      startNewFunction(funit); // this entry point of this procedure
      for (auto &eval : funit.evaluationList)
        genFIR(eval);
      endNewFunction(funit);
    }
    funit.setActiveEntry(0);
    for (auto &f : funit.nestedFunctions)
      lowerFunc(f); // internal procedure
  }

  /// Lower module variable definitions to fir::globalOp
  void lowerModuleVariables(Fortran::lower::pft::ModuleLikeUnit &mod) {
    // FIXME: get rid of the bogus function context and instantiate the
    // globals directly into the module.
    auto *context = &getMLIRContext();
    auto func = Fortran::lower::FirOpBuilder::createFunction(
        mlir::UnknownLoc::get(context), getModuleOp(),
        fir::NameUniquer::doGenerated("ModuleSham"),
        mlir::FunctionType::get(context, llvm::None, llvm::None));
    builder = new Fortran::lower::FirOpBuilder(func, bridge.getKindMap());
    for (const auto &var : mod.getOrderedSymbolTable())
      Fortran::lower::defineModuleVariable(*this, var);
    if (auto *region = func.getCallableRegion())
      region->dropAllReferences();
    func.erase();
    delete builder;
    builder = nullptr;
  }

  /// Lower functions contained in a module.
  void lowerMod(Fortran::lower::pft::ModuleLikeUnit &mod) {
    for (auto &f : mod.nestedFunctions)
      lowerFunc(f);
  }

  void setCurrentPosition(const Fortran::parser::CharBlock &position) {
    if (position != Fortran::parser::CharBlock{})
      currentPosition = position;
  }

  //
  // Utility methods
  //

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

  Fortran::lower::LoweringBridge &bridge;
  Fortran::evaluate::FoldingContext foldingContext;
  Fortran::lower::FirOpBuilder *builder = nullptr;
  Fortran::lower::pft::Evaluation *evalPtr = nullptr;
  Fortran::lower::SymMap localSymbols;
  Fortran::parser::CharBlock currentPosition;

  /// WHERE statement/construct mask expression stack.
  Fortran::lower::MaskExpr masks;
};

} // namespace

Fortran::evaluate::FoldingContext
Fortran::lower::LoweringBridge::createFoldingContext() const {
  return {getDefaultKinds(), getIntrinsicTable()};
}

void Fortran::lower::LoweringBridge::lower(
    const Fortran::parser::Program &prg,
    const Fortran::semantics::SemanticsContext &semanticsContext) {
  auto pft = Fortran::lower::createPFT(prg, semanticsContext);
  if (dumpBeforeFir)
    Fortran::lower::dumpPFT(llvm::errs(), *pft);
  FirConverter converter{*this};
  converter.run(*pft);
}

void Fortran::lower::LoweringBridge::parseSourceFile(llvm::SourceMgr &srcMgr) {
  auto owningRef = mlir::parseSourceFile(srcMgr, &context);
  module.reset(new mlir::ModuleOp(owningRef.get().getOperation()));
  owningRef.release();
}

Fortran::lower::LoweringBridge::LoweringBridge(
    mlir::MLIRContext &context,
    const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds,
    const Fortran::evaluate::IntrinsicProcTable &intrinsics,
    const Fortran::parser::AllCookedSources &cooked, llvm::StringRef triple,
    fir::KindMapping &kindMap)
    : defaultKinds{defaultKinds}, intrinsics{intrinsics}, cooked{&cooked},
      context{context}, kindMap{kindMap} {
  // Register the diagnostic handler.
  context.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
    auto &os = llvm::errs();
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
    if (!diag.getLocation().isa<UnknownLoc>())
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
