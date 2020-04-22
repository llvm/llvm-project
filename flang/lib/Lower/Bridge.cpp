//===-- Bridge.cc -- bridge to lower to MLIR ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Bridge.h"
#include "SymbolMap.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/IO.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Parser.h"
#include "mlir/Target/LLVMIR.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MD5.h"

#undef TODO
#define TODO()                                                                 \
  {                                                                            \
    if (disableToDoAssertions)                                                 \
      mlir::emitError(toLocation(), __FILE__)                                  \
          << ':' << __LINE__ << " not implemented";                            \
    else                                                                       \
      llvm_unreachable("not yet implemented");                                 \
  }

static llvm::cl::opt<bool>
    dumpBeforeFir("fdebug-dump-pre-fir", llvm::cl::init(false),
                  llvm::cl::desc("dump the IR tree prior to FIR"));

static llvm::cl::opt<bool>
    disableToDoAssertions("disable-burnside-todo",
                          llvm::cl::desc("disable burnside bridge asserts"),
                          llvm::cl::init(false), llvm::cl::Hidden);

static llvm::cl::opt<std::size_t>
    nameLengthHashSize("length-to-hash-string-literal",
                       llvm::cl::desc("string literals that exceed this length"
                                      " will use a hash value as their symbol "
                                      "name"),
                       llvm::cl::init(32));

static llvm::cl::opt<bool>
    useOldInitializerCode("enable-old-initializer-lowering",
                          llvm::cl::desc("TODO: remove the old code!"),
                          llvm::cl::init(false), llvm::cl::Hidden);

namespace {
/// Information for generating a structured or unstructured increment loop.
struct IncrementLoopInfo {
  explicit IncrementLoopInfo(
      Fortran::semantics::Symbol *sym,
      const Fortran::parser::ScalarExpr &lowerExpr,
      const Fortran::parser::ScalarExpr &upperExpr,
      const std::optional<Fortran::parser::ScalarExpr> &stepExpr,
      mlir::Type type)
      : loopVariableSym{sym}, lowerExpr{lowerExpr}, upperExpr{upperExpr},
        stepExpr{stepExpr}, loopVariableType{type} {}

  bool isStructured() const { return !headerBlock; }

  // Data members for both structured and unstructured loops.
  Fortran::semantics::Symbol *loopVariableSym;
  const Fortran::parser::ScalarExpr &lowerExpr;
  const Fortran::parser::ScalarExpr &upperExpr;
  const std::optional<Fortran::parser::ScalarExpr> &stepExpr;
  mlir::Type loopVariableType;

  mlir::Value loopVariable{};
  mlir::Value stepValue{}; // possible uses in multiple blocks

  // Data members for structured loops.
  fir::LoopOp doLoop{};
  mlir::OpBuilder::InsertPoint insertionPoint{};

  // Data members for unstructured loops.
  mlir::Value tripVariable{};
  mlir::Block *headerBlock{nullptr};    // loop entry and test block
  mlir::Block *bodyBlock{nullptr};      // first loop body block
  mlir::Block *successorBlock{nullptr}; // loop exit target block
};
} // namespace

static bool symIsChar(const Fortran::semantics::Symbol &sym) {
  return sym.GetType()->category() ==
         Fortran::semantics::DeclTypeSpec::Character;
}

static bool symIsArray(const Fortran::semantics::Symbol &sym) {
  const auto *det = sym.detailsIf<Fortran::semantics::ObjectEntityDetails>();
  return det ? det->IsArray() : false;
}

static bool isExplicitShape(const Fortran::semantics::Symbol &sym) {
  const auto *det = sym.detailsIf<Fortran::semantics::ObjectEntityDetails>();
  if (det && det->IsArray())
    return det->shape().IsExplicitShape();
  return false;
}

/// Temporary helper to detect shapes that do not require evaluating
/// bound expressions at runtime or to get the shape from a descriptor.
static bool isConstantShape(const Fortran::semantics::ArraySpec &shape) {
  auto isConstant = [](const auto &bound) {
    const auto &expr = bound.GetExplicit();
    return expr.has_value() && Fortran::evaluate::IsConstantExpr(*expr);
  };
  for (const auto &susbcript : shape) {
    const auto &lb = susbcript.lbound();
    const auto &ub = susbcript.ubound();
    if (isConstant(lb) && (isConstant(ub) || ub.isAssumed()))
      continue;
    return false;
  }
  return true;
}

namespace {
struct SymbolIndexAnalyzer {
  using FromBox = std::monostate;

  explicit SymbolIndexAnalyzer(const Fortran::semantics::Symbol &sym)
      : sym{sym} {}
  SymbolIndexAnalyzer() = delete;
  SymbolIndexAnalyzer(const SymbolIndexAnalyzer &) = delete;

  /// Run the analysis on the symbol. Used to determine the type of index to
  /// save in the symbol map.
  void analyze() {
    isChar = symIsChar(sym);
    if (isChar) {
      const auto &lenParam = sym.GetType()->characterTypeSpec().length();
      if (auto expr = lenParam.GetExplicit()) {
        auto len = Fortran::evaluate::AsGenericExpr(std::move(*expr));
        auto asInt = Fortran::evaluate::ToInt64(len);
        if (asInt) {
          charLen = *asInt;
        } else {
          charLen = len;
          staticSize = false;
        }
      } else {
        charLen = FromBox{};
        staticSize = false;
      }
    }
    isArray = symIsArray(sym);
    for (const auto &subs : getSymShape()) {
      auto low = subs.lbound().GetExplicit();
      auto high = subs.ubound().GetExplicit();
      if (staticSize && low && high) {
        auto lb = Fortran::evaluate::ToInt64(*low);
        auto ub = Fortran::evaluate::ToInt64(*high);
        if (lb && ub) {
          staticLBound.push_back(*lb);
          staticShape.push_back(*ub - *lb + 1);
          continue;
        }
      }
      staticSize = false;
      dynamicBound.push_back(&subs);
    }
  }

  /// Get the shape of an analyzed symbol.
  const Fortran::semantics::ArraySpec &getSymShape() {
    return sym.get<Fortran::semantics::ObjectEntityDetails>().shape();
  }

  /// Get the CHARACTER's LEN value, if there is one.
  llvm::Optional<int64_t> getCharLenConst() {
    if (isChar)
      if (auto *res = std::get_if<int64_t>(&charLen))
        return {*res};
    return {};
  }

  /// Get the CHARACTER's LEN expression, if there is one.
  llvm::Optional<Fortran::semantics::SomeExpr> getCharLenExpr() {
    if (isChar)
      if (auto *res = std::get_if<Fortran::semantics::SomeExpr>(&charLen))
        return {*res};
    return {};
  }

  /// Is it a CHARACTER with a constant LEN?
  bool charConstSize() const {
    return isChar && std::holds_alternative<int64_t>(charLen);
  }

  /// Symbol is neither a CHARACTER nor an array.
  bool isTrivial() const { return !(isChar || isArray); }

  /// Return true iff all the lower bound values are the constant 1.
  bool lboundIsAllOnes() const {
    return staticSize &&
           llvm::all_of(staticLBound, [](int64_t v) { return v == 1; });
  }

  llvm::SmallVector<int64_t, 8> staticLBound;
  llvm::SmallVector<int64_t, 8> staticShape;
  llvm::SmallVector<const Fortran::semantics::ShapeSpec *, 8> dynamicBound;
  bool staticSize{true};
  bool isChar{false};
  bool isArray{false};

private:
  std::variant<FromBox, int64_t, Fortran::semantics::SomeExpr> charLen{
      FromBox{}};
  const Fortran::semantics::Symbol &sym;
};
} // namespace

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
  explicit FirConverter(Fortran::lower::LoweringBridge &bridge,
                        fir::NameUniquer &uniquer)
      : mlirContext{bridge.getMLIRContext()}, cooked{bridge.getCookedSource()},
        module{bridge.getModule()}, defaults{bridge.getDefaultKinds()},
        kindMap{bridge.getKindMap()}, uniquer{uniquer} {}
  virtual ~FirConverter() = default;

  /// Convert the PFT to FIR
  void run(Fortran::lower::pft::Program &pft) {
    // do translation
    for (auto &u : pft.getUnits()) {
      std::visit(
          Fortran::common::visitors{
              [&](Fortran::lower::pft::FunctionLikeUnit &f) { lowerFunc(f); },
              [&](Fortran::lower::pft::ModuleLikeUnit &m) { lowerMod(m); },
              [&](Fortran::lower::pft::BlockDataUnit &) { TODO(); },
          },
          u);
    }
  }

  mlir::FunctionType genFunctionType(Fortran::lower::SymbolRef sym) {
    return Fortran::lower::translateSymbolToFIRFunctionType(&mlirContext,
                                                            defaults, sym);
  }

  //===--------------------------------------------------------------------===//
  // AbstractConverter overrides
  //===--------------------------------------------------------------------===//

  mlir::Value genExprAddr(const Fortran::lower::SomeExpr &expr,
                          mlir::Location *loc = nullptr) override final {
    return createFIRAddr(loc ? *loc : toLocation(), &expr);
  }
  mlir::Value genExprValue(const Fortran::lower::SomeExpr &expr,
                           mlir::Location *loc = nullptr) override final {
    return createFIRExpr(loc ? *loc : toLocation(), &expr);
  }

  mlir::Type genType(const Fortran::evaluate::DataRef &data) override final {
    return Fortran::lower::translateDataRefToFIRType(&mlirContext, defaults,
                                                     data);
  }
  mlir::Type genType(const Fortran::lower::SomeExpr &expr) override final {
    return Fortran::lower::translateSomeExprToFIRType(&mlirContext, defaults,
                                                      &expr);
  }
  mlir::Type genType(Fortran::lower::SymbolRef sym) override final {
    return Fortran::lower::translateSymbolToFIRType(&mlirContext, defaults,
                                                    sym);
  }
  mlir::Type genType(Fortran::common::TypeCategory tc,
                     int kind) override final {
    return Fortran::lower::getFIRType(&mlirContext, defaults, tc, kind);
  }
  mlir::Type genType(Fortran::common::TypeCategory tc) override final {
    return Fortran::lower::getFIRType(&mlirContext, defaults, tc);
  }

  mlir::Location getCurrentLocation() override final { return toLocation(); }

  /// Generate a dummy location.
  mlir::Location genLocation() override final {
    // Note: builder may not be instantiated yet
    return mlir::UnknownLoc::get(&mlirContext);
  }

  /// Generate a `Location` from the `CharBlock`.
  mlir::Location
  genLocation(const Fortran::parser::CharBlock &block) override final {
    if (cooked) {
      auto loc = cooked->GetSourcePositionRange(block);
      if (loc.has_value()) {
        // loc is a pair (begin, end); use the beginning position
        auto &filePos = loc->first;
        return mlir::FileLineColLoc::get(filePos.file.path(), filePos.line,
                                         filePos.column, &mlirContext);
      }
    }
    return genLocation();
  }

  Fortran::lower::FirOpBuilder &getFirOpBuilder() override final {
    return *builder;
  }

  mlir::ModuleOp &getModuleOp() override final { return module; }

  std::string
  mangleName(const Fortran::semantics::Symbol &symbol) override final {
    return Fortran::lower::mangle::mangleName(uniquer, symbol);
  }

  std::string uniqueCGIdent(llvm::StringRef name) override final {
    // For "long" identifiers use a hash value
    if (name.size() > nameLengthHashSize) {
      llvm::MD5 hash;
      hash.update(name);
      llvm::MD5::MD5Result result;
      hash.final(result);
      llvm::SmallString<32> str;
      llvm::MD5::stringifyResult(result, str);
      std::string hashName = "h.";
      hashName.append(str.c_str());
      return uniquer.doGenerated(hashName);
    }
    // "Short" identifiers use a reversible hex string
    return uniquer.doGenerated(llvm::toHex(name));
  }

private:
  FirConverter() = delete;
  FirConverter(const FirConverter &) = delete;
  FirConverter &operator=(const FirConverter &) = delete;

  //===--------------------------------------------------------------------===//
  // Helper member functions
  //===--------------------------------------------------------------------===//

  mlir::Value createFIRAddr(mlir::Location loc,
                            const Fortran::semantics::SomeExpr *expr) {
    return createSomeAddress(loc, *this, *expr, localSymbols);
  }

  mlir::Value createFIRExpr(mlir::Location loc,
                            const Fortran::semantics::SomeExpr *expr) {
    return createSomeExpression(loc, *this, *expr, localSymbols);
  }

  /// Find the symbol in the local map or return null.
  mlir::Value lookupSymbol(const Fortran::semantics::Symbol &sym) {
    if (auto v = localSymbols.lookupSymbol(sym))
      return v;
    return {};
  }

  /// Add the symbol to the local map. If the symbol is already in the map, it
  /// is not updated. Instead the value `false` is returned.
  bool addSymbol(const Fortran::semantics::SymbolRef sym, mlir::Value val,
                 bool forced = false) {
    if (forced)
      localSymbols.erase(sym);
    else if (lookupSymbol(sym))
      return false;
    localSymbols.addSymbol(sym, val);
    return true;
  }

  bool addCharSymbol(const Fortran::semantics::SymbolRef sym, mlir::Value val,
                     mlir::Value len, bool forced = false) {
    if (forced)
      localSymbols.erase(sym);
    else if (lookupSymbol(sym))
      return false;
    localSymbols.addCharSymbol(sym, val, len);
    return true;
  }

  mlir::Value createTemp(mlir::Location loc,
                         const Fortran::semantics::Symbol &sym,
                         llvm::ArrayRef<mlir::Value> shape = {}) {
    if (auto v = lookupSymbol(sym))
      return v;
    auto newVal = builder->createTemporary(loc, genType(sym),
                                           sym.name().ToString(), shape);
    addSymbol(sym, newVal);
    return newVal;
  }

  bool isNumericScalarCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::lower::IntegerCat ||
           cat == Fortran::lower::RealCat ||
           cat == Fortran::lower::ComplexCat ||
           cat == Fortran::lower::LogicalCat;
  }

  bool isCharacterCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::lower::CharacterCat;
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

  void genBranch(mlir::Block *targetBlock) {
    assert(targetBlock && "missing unconditional target block");
    builder->create<mlir::BranchOp>(toLocation(), targetBlock);
  }

  void genFIRConditionalBranch(mlir::Value &cond, mlir::Block *trueTarget,
                               mlir::Block *falseTarget) {
    builder->create<mlir::CondBranchOp>(toLocation(), cond, trueTarget,
                                        llvm::None, falseTarget, llvm::None);
  }

  void genFIRConditionalBranch(const Fortran::parser::ScalarLogicalExpr &expr,
                               Fortran::lower::pft::Evaluation *trueTarget,
                               Fortran::lower::pft::Evaluation *falseTarget) {
    assert(trueTarget && "missing conditional branch true block");
    assert(falseTarget && "missing conditional branch true block");
    mlir::Value cond = genExprValue(*Fortran::semantics::GetExpr(expr));
    genFIRConditionalBranch(cond, trueTarget->block, falseTarget->block);
  }

  //
  // Termination of symbolically referenced execution units
  //

  /// END of program
  ///
  /// Generate the cleanup block before the program exits
  void genExitRoutine() { builder->create<mlir::ReturnOp>(toLocation()); }
  void genFIR(const Fortran::parser::EndProgramStmt &) { genExitRoutine(); }

  /// END of procedure-like constructs
  ///
  /// Generate the cleanup block before the procedure exits
  void genReturnSymbol(const Fortran::semantics::Symbol &functionSymbol) {
    const auto &details =
        functionSymbol.get<Fortran::semantics::SubprogramDetails>();
    auto resultRef = lookupSymbol(details.result());
    mlir::Value retval = builder->create<fir::LoadOp>(toLocation(), resultRef);
    builder->create<mlir::ReturnOp>(toLocation(), retval);
  }

  void genFIRProcedureExit(Fortran::lower::pft::FunctionLikeUnit &funit,
                           const Fortran::semantics::Symbol &symbol) {
    // Make sure we end the current block with a terminator.
    if (auto *finalBlock = funit.finalBlock) {
      if (blockIsUnterminated())
        builder->create<mlir::BranchOp>(toLocation(), finalBlock);
      // Set insertion point to final block.
      builder->setInsertionPoint(finalBlock, finalBlock->end());
    }

    if (Fortran::semantics::IsFunction(symbol)) {
      // FUNCTION
      genReturnSymbol(symbol);
      return;
    }

    // SUBROUTINE
    if (Fortran::semantics::HasAlternateReturns(symbol)) {
      // lower to a the constant expression (or zero); the return value will
      // drive a SelectOp in the calling context to branch to the alternate
      // return LABEL block
      TODO();
      mlir::Value intExpr{};
      builder->create<mlir::ReturnOp>(toLocation(), intExpr);
      return;
    }

    genExitRoutine();
  }

  //
  // Statements that have control-flow semantics
  //

  template <typename A>
  std::pair<mlir::OpBuilder::InsertPoint, fir::WhereOp>
  genWhereCondition(const A *stmt, bool withElse = true) {
    auto cond = genExprValue(*Fortran::semantics::GetExpr(
        std::get<Fortran::parser::ScalarLogicalExpr>(stmt->t)));
    auto where = builder->create<fir::WhereOp>(toLocation(), cond, withElse);
    auto insPt = builder->saveInsertionPoint();
    builder->setInsertionPointToStart(&where.whereRegion().front());
    return {insPt, where};
  }

  mlir::Value genFIRLoopIndex(const Fortran::parser::ScalarExpr &x,
                              mlir::Type t) {
    mlir::Value v = genExprValue(*Fortran::semantics::GetExpr(x));
    return builder->create<fir::ConvertOp>(toLocation(), t, v);
  }

  mlir::FuncOp getFunc(llvm::StringRef name, mlir::FunctionType ty) {
    if (auto func = builder->getNamedFunction(name)) {
      assert(func.getType() == ty);
      return func;
    }
    return builder->createFunction(name, ty);
  }

  /// Lowering of CALL statement
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CallStmt &stmt) {
    setCurrentPosition(stmt.v.source);
    assert(stmt.typedCall && "Call was not analyzed");
    // The actual lowering is forwarded to expression lowering
    // where the code is shared with function reference.
    Fortran::semantics::SomeExpr expr{*stmt.typedCall};
    auto res = createFIRExpr(toLocation(), &expr);
    if (res)
      TODO(); // Alternate returns
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::IfStmt &stmt) {
    if (eval.lowerAsUnstructured()) {
      genFIRConditionalBranch(
          std::get<Fortran::parser::ScalarLogicalExpr>(stmt.t),
          eval.lexicalSuccessor, eval.controlSuccessor);
      return;
    }

    // Generate fir.where.
    auto pair = genWhereCondition(&stmt, /*withElse=*/false);
    genFIR(*eval.lexicalSuccessor, /*unstructuredContext=*/false);
    eval.lexicalSuccessor->skip = true;
    builder->restoreInsertionPoint(pair.first);
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::WaitStmt &stmt) {
    genWaitStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::WhereStmt &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ComputedGotoStmt &stmt) {
    mlir::Value selectExpr = genExprValue(*Fortran::semantics::GetExpr(
        std::get<Fortran::parser::ScalarIntExpr>(stmt.t)));
    constexpr int vSize = 10;
    llvm::SmallVector<int64_t, vSize> indexList;
    llvm::SmallVector<mlir::Block *, vSize> blockList;
    int64_t index = 0;
    for (auto &label : std::get<std::list<Fortran::parser::Label>>(stmt.t)) {
      indexList.push_back(++index);
      blockList.push_back(blockOfLabel(eval, label));
    }
    blockList.push_back(eval.lexicalSuccessor->block); // default
    builder->create<fir::SelectOp>(toLocation(), selectExpr, indexList,
                                   blockList);
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ForallStmt &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ArithmeticIfStmt &stmt) {
    mlir::Value expr = genExprValue(
        *Fortran::semantics::GetExpr(std::get<Fortran::parser::Expr>(stmt.t)));
    auto exprType = expr.getType();
    if (exprType.isSignlessInteger()) {
      // Arithmetic expression has Integer type.  Generate a SelectCaseOp
      // with ranges {(-inf:-1], 0=default, [1:inf)}.
      MLIRContext *context = builder->getContext();
      llvm::SmallVector<mlir::Attribute, 3> attrList;
      llvm::SmallVector<mlir::Value, 3> valueList;
      llvm::SmallVector<mlir::Block *, 3> blockList;
      attrList.push_back(fir::UpperBoundAttr::get(context));
      valueList.push_back(builder->createIntegerConstant(exprType, -1));
      blockList.push_back(blockOfLabel(eval, std::get<1>(stmt.t)));
      attrList.push_back(fir::LowerBoundAttr::get(context));
      valueList.push_back(builder->createIntegerConstant(exprType, 1));
      blockList.push_back(blockOfLabel(eval, std::get<3>(stmt.t)));
      attrList.push_back(mlir::UnitAttr::get(context)); // 0 is the "default"
      blockList.push_back(blockOfLabel(eval, std::get<2>(stmt.t)));
      builder->create<fir::SelectCaseOp>(toLocation(), expr, attrList,
                                         valueList, blockList);
      return;
    }
    // Arithmetic expression has Real type.  Generate
    //   sum = expr + expr  [ raise an exception if expr is a NaN ]
    //   if (sum < 0.0) goto L1 else if (sum > 0.0) goto L3 else goto L2
    assert(eval.localBlocks.size() == 1 && "missing arithmetic if block");
    mlir::Value sum = builder->create<fir::AddfOp>(toLocation(), expr, expr);
    mlir::Value zero = builder->create<mlir::ConstantOp>(
        toLocation(), exprType, builder->getFloatAttr(exprType, 0.0));
    mlir::Value cond1 = builder->create<mlir::CmpFOp>(
        toLocation(), mlir::CmpFPredicate::OLT, sum, zero);
    genFIRConditionalBranch(cond1, blockOfLabel(eval, std::get<1>(stmt.t)),
                            eval.localBlocks[0]);
    startBlock(eval.localBlocks[0]);
    mlir::Value cond2 = builder->create<mlir::CmpFOp>(
        toLocation(), mlir::CmpFPredicate::OGT, sum, zero);
    genFIRConditionalBranch(cond2, blockOfLabel(eval, std::get<3>(stmt.t)),
                            blockOfLabel(eval, std::get<2>(stmt.t)));
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::AssignedGotoStmt &stmt) {
    // Program requirement 1990 8.2.4 -
    //
    //   At the time of execution of an assigned GOTO statement, the integer
    //   variable must be defined with the value of a statement label of a
    //   branch target statement that appears in the same scoping unit.
    //   Note that the variable may be defined with a statement label value
    //   only by an ASSIGN statement in the same scoping unit as the assigned
    //   GOTO statement.

    const auto &symbolLabelMap =
        eval.getOwningProcedure()->assignSymbolLabelMap;
    const auto &symbol = *std::get<Fortran::parser::Name>(stmt.t).symbol;
    auto variable = lookupSymbol(symbol);
    if (!variable)
      variable = createTemp(toLocation(), symbol);
    auto selectExpr = builder->create<fir::LoadOp>(toLocation(), variable);
    auto iter = symbolLabelMap.find(symbol);
    if (iter == symbolLabelMap.end()) {
      // This "assert" will fail for a nonconforming program unit that does not
      // have any ASSIGN statements.  The front end should check for this.
      // If asserts are inactive, the assigned GOTO statement will be a nop.
      llvm_unreachable("no assigned goto targets");
      return;
    }
    auto labelSet = iter->second;
    constexpr int vSize = 10;
    llvm::SmallVector<int64_t, vSize> indexList;
    llvm::SmallVector<mlir::Block *, vSize> blockList;
    auto addLabel = [&](Fortran::parser::Label label) {
      indexList.push_back(label);
      blockList.push_back(blockOfLabel(eval, label));
    };
    // Add labels from an explicit list.  The list may have duplicates.
    for (auto &label : std::get<std::list<Fortran::parser::Label>>(stmt.t)) {
      if (labelSet.count(label) == 0) {
        // This "assert" will fail for a nonconforming program unit that never
        // ASSIGNs this label to the selector variable.  The front end should
        // check that there is at least one such ASSIGN statement.  If asserts
        // are inactive, the label will be ignored.
        llvm_unreachable("invalid assigned goto target");
        continue;
      }
      if (std::find(indexList.begin(), indexList.end(), label) ==
          indexList.end()) { // ignore duplicates
        addLabel(label);
      }
    }
    // Absent an explicit list, add all possible label targets.
    if (indexList.empty()) {
      for (auto &label : labelSet) {
        addLabel(label);
      }
    }
    // Add a nop/fallthrough branch to the switch for a nonconforming program
    // unit that violates the program requirement above.
    blockList.push_back(eval.lexicalSuccessor->block); // default
    builder->create<fir::SelectOp>(toLocation(), selectExpr, indexList,
                                   blockList);
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::AssociateConstruct &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::BlockConstruct &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ChangeTeamConstruct &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CriticalConstruct &) {
    TODO();
  }

  /// Generate FIR for a DO construct.  There are six variants:
  ///  - unstructured infinite and while loops
  ///  - structured and unstructured increment loops
  ///  - structured and unstructured concurrent loops
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::DoConstruct &) {
    bool unstructuredContext{eval.lowerAsUnstructured()};
    Fortran::lower::pft::Evaluation &doStmtEval = eval.evaluationList->front();
    auto *doStmt = doStmtEval.getIf<Fortran::parser::NonLabelDoStmt>();
    assert(doStmt && "missing DO statement");
    const auto &loopControl =
        std::get<std::optional<Fortran::parser::LoopControl>>(doStmt->t);
    llvm::SmallVector<IncrementLoopInfo, 1> incrementLoopInfo;
    const Fortran::parser::ScalarLogicalExpr *whileCondition = nullptr;
    bool infiniteLoop = !loopControl.has_value();
    if (infiniteLoop) {
      assert(unstructuredContext && "infinite loop must be unstructured");
      startBlock(doStmtEval.localBlocks[0]); // header block
    } else if ((whileCondition =
                    std::get_if<Fortran::parser::ScalarLogicalExpr>(
                        &loopControl->u))) {
      assert(unstructuredContext && "while loop must be unstructured");
      startBlock(doStmtEval.localBlocks[0]); // header block
      genFIRConditionalBranch(*whileCondition, doStmtEval.lexicalSuccessor,
                              doStmtEval.parentConstruct->constructExit);
    } else if (const auto *bounds =
                   std::get_if<Fortran::parser::LoopControl::Bounds>(
                       &loopControl->u)) {
      // "Normal" increment loop.
      incrementLoopInfo.emplace_back(bounds->name.thing.symbol, bounds->lower,
                                     bounds->upper, bounds->step,
                                     genType(*bounds->name.thing.symbol));
      if (unstructuredContext) {
        maybeStartBlock(doStmtEval.block); // preheader block
        incrementLoopInfo[0].headerBlock = doStmtEval.localBlocks[0];
        incrementLoopInfo[0].bodyBlock = doStmtEval.lexicalSuccessor->block;
        incrementLoopInfo[0].successorBlock =
            doStmtEval.parentConstruct->constructExit->block;
      }
    } else {
      const auto *concurrentInfo =
          std::get_if<Fortran::parser::LoopControl::Concurrent>(
              &loopControl->u);
      assert(concurrentInfo && "DO loop variant is invalid");
      TODO();
      // Add entries to incrementLoopInfo.  (Define extra members for a mask.)
    }
    auto n = incrementLoopInfo.size();
    for (decltype(n) i = 0; i < n; ++i)
      genFIRIncrementLoopBegin(incrementLoopInfo[i]);

    // Generate loop body code.
    for (auto &e : *eval.evaluationList)
      genFIR(e, unstructuredContext);

    // Generate end loop code.
    if (infiniteLoop || whileCondition) {
      genBranch(doStmtEval.localBlocks[0]);
    } else {
      for (auto i = incrementLoopInfo.size(); i > 0;)
        genFIRIncrementLoopEnd(incrementLoopInfo[--i]);
    }
  }

  /// Generate FIR to begin a structured or unstructured increment loop.
  void genFIRIncrementLoopBegin(IncrementLoopInfo &info) {
    auto location = toLocation();
    mlir::Type type = info.isStructured()
                          ? mlir::IndexType::get(builder->getContext())
                          : info.loopVariableType;
    auto lowerValue = genFIRLoopIndex(info.lowerExpr, type);
    auto upperValue = genFIRLoopIndex(info.upperExpr, type);
    info.stepValue =
        info.stepExpr.has_value()
            ? genFIRLoopIndex(*info.stepExpr, type)
            : (info.isStructured()
                   ? builder->create<mlir::ConstantIndexOp>(location, 1)
                   : builder->createIntegerConstant(info.loopVariableType, 1));
    assert(info.stepValue && "step value must be set");
    info.loopVariable = createTemp(location, *info.loopVariableSym);

    // Structured loop - generate fir.loop.
    if (info.isStructured()) {
      info.insertionPoint = builder->saveInsertionPoint();
      info.doLoop = builder->create<fir::LoopOp>(location, lowerValue,
                                                 upperValue, info.stepValue);
      builder->setInsertionPointToStart(info.doLoop.getBody());
      // Always store iteration ssa-value to the LCV to avoid missing any
      // aliasing of the LCV.
      auto lcv = builder->create<fir::ConvertOp>(
          location, info.loopVariableType, info.doLoop.getInductionVar());
      builder->create<fir::StoreOp>(location, lcv, info.loopVariable);
      return;
    }

    // Unstructured loop preheader code - initialize tripVariable, loopVariable.
    auto distance =
        builder->create<mlir::SubIOp>(location, upperValue, lowerValue);
    auto adjusted =
        builder->create<mlir::AddIOp>(location, distance, info.stepValue);
    auto tripCount =
        builder->create<mlir::SignedDivIOp>(location, adjusted, info.stepValue);
    info.tripVariable =
        builder->createTemporary(location, info.loopVariableType);
    builder->create<fir::StoreOp>(location, tripCount, info.tripVariable);
    builder->create<fir::StoreOp>(location, lowerValue, info.loopVariable);

    // Unstructured loop header code - generate loop condition.
    startBlock(info.headerBlock);
    mlir::Value tripVariable =
        builder->create<fir::LoadOp>(location, info.tripVariable);
    mlir::Value zero = builder->createIntegerConstant(info.loopVariableType, 0);
    mlir::Value cond = builder->create<mlir::CmpIOp>(
        location, mlir::CmpIPredicate::sgt, tripVariable, zero);
    genFIRConditionalBranch(cond, info.bodyBlock, info.successorBlock);
  }

  /// Generate FIR to end a structured or unstructured increment loop.
  void genFIRIncrementLoopEnd(IncrementLoopInfo &info) {
    mlir::Location location = toLocation();
    if (info.isStructured()) {
      // End fir.loop.
      builder->restoreInsertionPoint(info.insertionPoint);
      return;
    }

    // Unstructured loop - increment loopVariable.
    mlir::Value loopVariable =
        builder->create<fir::LoadOp>(location, info.loopVariable);
    loopVariable =
        builder->create<mlir::AddIOp>(location, loopVariable, info.stepValue);
    builder->create<fir::StoreOp>(location, loopVariable, info.loopVariable);

    // Unstructured loop - decrement tripVariable.
    mlir::Value tripVariable =
        builder->create<fir::LoadOp>(location, info.tripVariable);
    mlir::Value one = builder->create<mlir::ConstantOp>(
        location, builder->getIntegerAttr(info.loopVariableType, 1));
    tripVariable = builder->create<mlir::SubIOp>(location, tripVariable, one);
    builder->create<fir::StoreOp>(location, tripVariable, info.tripVariable);
    genBranch(info.headerBlock);
  }

  /// Generate structured or unstructured FIR for an IF construct.
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::IfConstruct &) {
    if (eval.lowerAsStructured()) {
      // Structured fir.where nest.
      fir::WhereOp underWhere;
      mlir::OpBuilder::InsertPoint insPt;
      for (auto &e : *eval.evaluationList) {
        if (auto *s = e.getIf<Fortran::parser::IfThenStmt>()) {
          // fir.where op
          std::tie(insPt, underWhere) = genWhereCondition(s);
        } else if (auto *s = e.getIf<Fortran::parser::ElseIfStmt>()) {
          // otherwise block, then nested fir.where
          builder->setInsertionPointToStart(&underWhere.otherRegion().front());
          std::tie(std::ignore, underWhere) = genWhereCondition(s);
        } else if (e.isA<Fortran::parser::ElseStmt>()) {
          // otherwise block
          builder->setInsertionPointToStart(&underWhere.otherRegion().front());
        } else if (e.isA<Fortran::parser::EndIfStmt>()) {
          builder->restoreInsertionPoint(insPt);
        } else {
          genFIR(e, /*unstructuredContext=*/false);
        }
      }
      return;
    }

    // Unstructured branch sequence.
    for (auto &e : *eval.evaluationList) {
      const Fortran::parser::ScalarLogicalExpr *cond = nullptr;
      if (auto *s = e.getIf<Fortran::parser::IfThenStmt>()) {
        maybeStartBlock(e.block);
        cond = &std::get<Fortran::parser::ScalarLogicalExpr>(s->t);
      } else if (auto *s = e.getIf<Fortran::parser::ElseIfStmt>()) {
        startBlock(e.block);
        cond = &std::get<Fortran::parser::ScalarLogicalExpr>(s->t);
      }
      if (cond) {
        genFIRConditionalBranch(
            *cond,
            e.lexicalSuccessor == e.controlSuccessor
                ? e.parentConstruct->constructExit // empty block --> exit
                : e.lexicalSuccessor,              // nonempty block
            e.controlSuccessor);
      } else {
        genFIR(e);
      }
    }
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CaseConstruct &) {
    for (auto &e : *eval.evaluationList)
      genFIR(e);
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SelectRankConstruct &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SelectTypeConstruct &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::WhereConstruct &) {
    TODO();
  }

  /// Lower FORALL construct (See 10.2.4)
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ForallConstruct &forall) {
    auto &stmt = std::get<
        Fortran::parser::Statement<Fortran::parser::ForallConstructStmt>>(
        forall.t);
    setCurrentPosition(stmt.source);
    auto &fas = stmt.statement;
    auto &ctrl =
        std::get<
            Fortran::common::Indirection<Fortran::parser::ConcurrentHeader>>(
            fas.t)
            .value();
    (void)ctrl;
    for (auto &s :
         std::get<std::list<Fortran::parser::ForallBodyConstruct>>(forall.t)) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::Statement<
                  Fortran::parser::ForallAssignmentStmt> &b) {
                setCurrentPosition(b.source);
                genFIR(eval, b.statement);
              },
              [&](const Fortran::parser::Statement<Fortran::parser::WhereStmt>
                      &b) {
                setCurrentPosition(b.source);
                genFIR(eval, b.statement);
              },
              [&](const Fortran::parser::WhereConstruct &b) {
                genFIR(eval, b);
              },
              [&](const Fortran::common::Indirection<
                  Fortran::parser::ForallConstruct> &b) {
                genFIR(eval, b.value());
              },
              [&](const Fortran::parser::Statement<Fortran::parser::ForallStmt>
                      &b) {
                setCurrentPosition(b.source);
                genFIR(eval, b.statement);
              },
          },
          s.u);
    }
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ForallAssignmentStmt &s) {
    std::visit([&](auto &b) { genFIR(eval, b); }, s.u);
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CompilerDirective &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::OpenMPConstruct &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::OmpEndLoopDirective &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::AssociateStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndAssociateStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::BlockStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndBlockStmt &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SelectCaseStmt &stmt) {
    using ScalarExpr = Fortran::parser::Scalar<Fortran::parser::Expr>;
    MLIRContext *context = builder->getContext();
    const auto selectExpr = genExprValue(
        *Fortran::semantics::GetExpr(std::get<ScalarExpr>(stmt.t)));
    const auto selectType = selectExpr.getType();
    constexpr int vSize = 10;
    llvm::SmallVector<mlir::Attribute, vSize> attrList;
    llvm::SmallVector<mlir::Value, vSize> valueList;
    llvm::SmallVector<mlir::Block *, vSize> blockList;
    auto *defaultBlock = eval.parentConstruct->constructExit->block;
    using CaseValue = Fortran::parser::Scalar<Fortran::parser::ConstantExpr>;
    auto addValue = [&](const CaseValue &caseValue) {
      const auto *expr = Fortran::semantics::GetExpr(caseValue.thing);
      const auto v = Fortran::evaluate::ToInt64(*expr);
      valueList.push_back(
          v ? builder->createIntegerConstant(selectType, *v)
            : builder->create<fir::ConvertOp>(toLocation(), selectType,
                                              genExprValue(*expr)));
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
    attrList.push_back(mlir::UnitAttr::get(context));
    blockList.push_back(defaultBlock);
    builder->create<fir::SelectCaseOp>(toLocation(), selectExpr, attrList,
                                       valueList, blockList);
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CaseStmt &) {} // nop
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndSelectStmt &) {} // nop

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ChangeTeamStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndChangeTeamStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CriticalStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndCriticalStmt &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::NonLabelDoStmt &) {} // nop
  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::EndDoStmt &) {} // nop

  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::IfThenStmt &) {} // nop
  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::ElseIfStmt &) {} // nop
  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::ElseStmt &) {} // nop
  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::EndIfStmt &) {} // nop

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SelectRankStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SelectRankCaseStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SelectTypeStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::TypeGuardStmt &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::WhereConstructStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::MaskedElsewhereStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ElsewhereStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndWhereStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ForallConstructStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndForallStmt &) {
    TODO();
  }

  //===--------------------------------------------------------------------===//
  // IO statements (see io.h)
  //===--------------------------------------------------------------------===//

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::BackspaceStmt &stmt) {
    genBackspaceStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CloseStmt &stmt) {
    genCloseStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndfileStmt &stmt) {
    genEndfileStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::FlushStmt &stmt) {
    genFlushStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::InquireStmt &stmt) {
    genInquireStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::OpenStmt &stmt) {
    genOpenStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::PrintStmt &stmt) {
    genPrintStatement(*this, stmt,
                      eval.getOwningProcedure()->labelEvaluationMap);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ReadStmt &stmt) {
    genReadStatement(*this, stmt,
                     eval.getOwningProcedure()->labelEvaluationMap);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::RewindStmt &stmt) {
    genRewindStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::WriteStmt &stmt) {
    genWriteStatement(*this, stmt,
                      eval.getOwningProcedure()->labelEvaluationMap);
  }

  //===--------------------------------------------------------------------===//
  // Memory allocation and deallocation
  //===--------------------------------------------------------------------===//

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::AllocateStmt &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::DeallocateStmt &) {
    TODO();
  }

  /// Nullify pointer object list
  ///
  /// For each pointer object, reset the pointer to a disassociated status.
  /// We do this by setting each pointer to null.
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::NullifyStmt &stmt) {
    for (auto &po : stmt.v) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::Name &sym) {
                auto ty = genType(*sym.symbol);
                auto load = builder->create<fir::LoadOp>(
                    toLocation(), lookupSymbol(*sym.symbol));
                auto idxTy = mlir::IndexType::get(&mlirContext);
                auto zero = builder->create<mlir::ConstantOp>(
                    toLocation(), idxTy, builder->getIntegerAttr(idxTy, 0));
                auto cast =
                    builder->create<fir::ConvertOp>(toLocation(), ty, zero);
                builder->create<fir::StoreOp>(toLocation(), cast, load);
              },
              [&](const Fortran::parser::StructureComponent &) { TODO(); },
          },
          po.u);
    }
  }

  //===--------------------------------------------------------------------===//

  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::ContinueStmt &) {
    // do nothing
  }

  // We don't have runtime library support for various features. When they are
  // encountered, we emit an error message and exit immediately.
  void noRuntimeSupport(llvm::StringRef stmt) {
    mlir::emitError(toLocation(), "There is no runtime support for ")
        << stmt << " statement.\n";
    std::exit(1);
  }

  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::EventPostStmt &) {
    // FIXME: There is no runtime call to make for this yet.
    noRuntimeSupport("EVENT POST");
  }
  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::EventWaitStmt &) {
    // FIXME: There is no runtime call to make for this yet.
    noRuntimeSupport("EVENT WAIT");
  }

  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::FormTeamStmt &) {
    // FIXME: There is no runtime call to make for this yet.
    noRuntimeSupport("FORM TEAM");
  }
  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::LockStmt &) {
    // FIXME: There is no runtime call to make for this yet.
    noRuntimeSupport("LOCK");
  }

  /// The LHS and RHS on assignments are not always in agreement in terms of
  /// type. In some cases, the disagreement is between COMPLEX and REAL types.
  /// In that case, the assignment must insert/extract out of a COMPLEX value to
  /// be correct and strongly typed.
  mlir::Value convertOnAssign(mlir::Location loc, mlir::Type toTy,
                              mlir::Value val) {
    assert(toTy && "store location must be typed");
    auto fromTy = val.getType();
    if (fromTy == toTy)
      return val;
    if (fir::isa_real(fromTy) && fir::isa_complex(toTy)) {
      // imaginary part is zero
      auto eleTy = builder->getComplexPartType(toTy);
      auto cast = builder->create<fir::ConvertOp>(loc, eleTy, val);
      llvm::APFloat zero{
          kindMap.getFloatSemantics(toTy.cast<fir::CplxType>().getFKind()), 0};
      auto imag = builder->createRealConstant(loc, eleTy, zero);
      return builder->createComplex(loc, toTy, cast, imag);
    }
    if (fir::isa_complex(fromTy) && fir::isa_real(toTy)) {
      // drop the imaginary part
      auto rp = builder->extractComplexPart(val, /*isImagPart=*/false);
      return builder->create<fir::ConvertOp>(loc, toTy, rp);
    }
    return builder->create<fir::ConvertOp>(loc, toTy, val);
  }

  /// Shared for both assignments and pointer assignments.
  void genFIR(const Fortran::evaluate::Assignment &assignment) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::Assignment::Intrinsic &) {
              const auto *sym =
                  Fortran::evaluate::UnwrapWholeSymbolDataRef(assignment.lhs);
              if (sym && Fortran::semantics::IsAllocatable(*sym)) {
                // Assignment of allocatable are more complex, the lhs
                // may need to be deallocated/reallocated.
                // See Fortran 2018 10.2.1.3 p3
                TODO();
              } else if (sym && Fortran::semantics::IsPointer(*sym)) {
                // Target of the pointer must be assigned.
                // See Fortran 2018 10.2.1.3 p2
                auto lhsType = assignment.lhs.GetType();
                assert(lhsType && "lhs cannot be typeless");
                if (isNumericScalarCategory(lhsType->category())) {
                  auto val = genExprValue(assignment.rhs);
                  auto addr = genExprValue(assignment.lhs);
                  auto toTy = fir::dyn_cast_ptrEleTy(addr.getType());
                  auto cast = convertOnAssign(toLocation(), toTy, val);
                  builder->create<fir::StoreOp>(toLocation(), cast, addr);
                } else if (isCharacterCategory(lhsType->category())) {
                  TODO();
                } else {
                  assert(lhsType->category() == Fortran::lower::DerivedCat);
                  TODO();
                }
              } else if (assignment.lhs.Rank() > 0) {
                // Array assignment
                // See Fortran 2018 10.2.1.3 p5, p6, and p7
                TODO();
              } else {
                // Scalar assignments
                auto lhsType = assignment.lhs.GetType();
                assert(lhsType && "lhs cannot be typeless");
                if (isNumericScalarCategory(lhsType->category())) {
                  // Fortran 2018 10.2.1.3 p8 and p9
                  // Conversions should have been inserted by semantic analysis,
                  // but they can be incorrect between the rhs and lhs. Correct
                  // that here.
                  auto loc = toLocation();
                  auto addr = genExprAddr(assignment.lhs);
                  auto val = genExprValue(assignment.rhs);
                  auto toTy = fir::dyn_cast_ptrEleTy(addr.getType());
                  auto cast = convertOnAssign(loc, toTy, val);
                  builder->create<fir::StoreOp>(loc, cast, addr);
                } else if (isCharacterCategory(lhsType->category())) {
                  // Fortran 2018 10.2.1.3 p10 and p11
                  // Generating value for lhs to get fir.boxchar.
                  auto lhs{genExprValue(assignment.lhs)};
                  auto rhs{genExprValue(assignment.rhs)};
                  builder->createAssign(lhs, rhs);
                } else {
                  assert(lhsType->category() == Fortran::lower::DerivedCat);
                  // Fortran 2018 10.2.1.3 p12 and p13
                  TODO();
                }
              }
            },
            [&](const Fortran::evaluate::ProcedureRef &) {
              // Defined assignment: call ProcRef
              TODO();
            },
            [&](const Fortran::evaluate::Assignment::BoundsSpec &) {
              // Pointer assignment with possibly empty bounds-spec
              TODO();
            },
            [&](const Fortran::evaluate::Assignment::BoundsRemapping &) {
              // Pointer assignment with bounds-remapping
              TODO();
            },
        },
        assignment.u);
  }

  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::PointerAssignmentStmt &stmt) {
    genFIR(*stmt.typedAssignment->v);
  }

  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::AssignmentStmt &stmt) {
    genFIR(*stmt.typedAssignment->v);
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SyncAllStmt &) {
    // FIXME: There is no runtime call to make for this yet.
    noRuntimeSupport("SYNC ALL");
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SyncImagesStmt &) {
    // FIXME: There is no runtime call to make for this yet.
    noRuntimeSupport("SYNC IMAGES");
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SyncMemoryStmt &) {
    // FIXME: There is no runtime call to make for this yet.
    noRuntimeSupport("SYNC MEMORY");
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SyncTeamStmt &) {
    // FIXME: There is no runtime call to make for this yet.
    noRuntimeSupport("SYNC TEAM");
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::UnlockStmt &) {
    // FIXME: There is no runtime call to make for this yet.
    noRuntimeSupport("UNLOCK");
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::AssignStmt &stmt) {
    const auto &symbol = *std::get<Fortran::parser::Name>(stmt.t).symbol;
    auto variable = lookupSymbol(symbol);
    if (!variable)
      variable = createTemp(toLocation(), symbol);
    const auto labelValue = builder->createIntegerConstant(
        genType(symbol), std::get<Fortran::parser::Label>(stmt.t));
    builder->create<fir::StoreOp>(toLocation(), labelValue, variable);
  }

  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::FormatStmt &) {
    // do nothing.

    // FORMAT statements have no semantics. They may be lowered if used by a
    // data transfer statement.
  }

  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::EntryStmt &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::PauseStmt &) {
    // FIXME: There is no runtime call to make for this yet.
    noRuntimeSupport("PAUSE");
  }

  void genFIR(Fortran::lower::pft::Evaluation &,
              const Fortran::parser::DataStmt &) {
    // FIXME: The front-end doesn't provide the right information yet.
    mlir::emitError(toLocation(), "DATA statement is not handled.");
    exit(1);
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::NamelistStmt &) {
    TODO();
  }

  // call FAIL IMAGE in runtime
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::FailImageStmt &stmt) {
    auto callee = genRuntimeFunction(
        Fortran::lower::RuntimeEntryCode::FailImageStatement, *builder);
    llvm::SmallVector<mlir::Value, 1> operands; // FAIL IMAGE has no args
    builder->create<mlir::CallOp>(toLocation(), callee, operands);
  }

  // call STOP, ERROR STOP in runtime
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::StopStmt &stmt) {
    auto callee = genRuntimeFunction(
        Fortran::lower::RuntimeEntryCode::StopStatement, *builder);
    llvm::SmallVector<mlir::Value, 8> operands;
    builder->create<mlir::CallOp>(toLocation(), callee, operands);
  }

  // gen expression, if any; share code with END of procedure
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ReturnStmt &stmt) {
    const auto *funit = eval.getOwningProcedure();
    assert(funit && "not inside main program or a procedure");
    if (funit->isMainProgram()) {
      genExitRoutine();
    } else {
      if (stmt.v) {
        // Alternate return
        TODO();
      }
      // an ordinary RETURN should be lowered as a GOTO to the last block of the
      // SUBROUTINE
      auto *subr = eval.getOwningProcedure();
      assert(subr && "RETURN not in a PROCEDURE");
      if (!subr->finalBlock) {
        auto insPt = builder->saveInsertionPoint();
        subr->finalBlock = builder->createBlock(&builder->getRegion());
        builder->restoreInsertionPoint(insPt);
      }
      builder->create<mlir::BranchOp>(toLocation(), subr->finalBlock);
    }
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CycleStmt &) {
    genBranch(eval.controlSuccessor->block);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ExitStmt &) {
    genBranch(eval.controlSuccessor->block);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::GotoStmt &) {
    genBranch(eval.controlSuccessor->block);
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              bool unstructuredContext = true) {
    if (eval.skip)
      return; // rhs of {Forall,If,Where}Stmt has already been processed

    setCurrentPosition(eval.position);
    if (unstructuredContext) {
      // When transitioning from unstructured to structured code,
      // the structured code might be a target that starts a new block.
      maybeStartBlock(eval.isConstruct() && eval.lowerAsStructured()
                          ? eval.evaluationList->front().block
                          : eval.block);
    }
    eval.visit([&](const auto &stmt) { genFIR(eval, stmt); });
    if (unstructuredContext && eval.isActionStmt() && eval.controlSuccessor &&
        eval.controlSuccessor->block && blockIsUnterminated()) {
      // Exit from an unstructured IF or SELECT construct block.
      genBranch(eval.controlSuccessor->block);
    }
  }

  mlir::FuncOp createNewFunction(mlir::Location loc, llvm::StringRef name,
                                 const Fortran::semantics::Symbol *symbol) {
    mlir::FunctionType ty =
        symbol ? genFunctionType(*symbol)
               : mlir::FunctionType::get(llvm::None, llvm::None, &mlirContext);
    return Fortran::lower::FirOpBuilder::createFunction(loc, module, name, ty);
  }

  /// Evaluate specification expressions of local symbol and add
  /// the resulting `mlir::Value` to localSymbols.
  /// Before evaluating a specification expression, the symbols
  /// appearing in the expression are gathered, and if they are also
  /// local symbols, their specification are evaluated first. In case
  /// a circular dependency occurs, this will crash.
  void instantiateLocalVariable(
      const Fortran::semantics::Symbol &symbol,
      Fortran::lower::SymMap &dummyArgs,
      llvm::DenseSet<Fortran::semantics::SymbolRef> attempted) {
    if (lookupSymbol(symbol))
      return; // already instantiated

    if (IsProcedure(symbol))
      return;

    if (symbol.has<Fortran::semantics::UseDetails>() ||
        symbol.has<Fortran::semantics::HostAssocDetails>())
      TODO(); // Need to keep the localSymbols of other units ?

    if (attempted.find(symbol) != attempted.end())
      TODO(); // Complex dependencies in specification expressions.

    attempted.insert(symbol);
    mlir::Value localValue;
    auto *type = symbol.GetType();
    assert(type && "expected type for local symbol");

    if (type->category() == Fortran::semantics::DeclTypeSpec::Character) {
      const auto &lengthParam = type->characterTypeSpec().length();
      if (auto expr = lengthParam.GetExplicit()) {
        for (const auto &requiredSymbol :
             Fortran::evaluate::CollectSymbols(*expr)) {
          instantiateLocalVariable(requiredSymbol, dummyArgs, attempted);
        }
        auto lenValue =
            genExprValue(Fortran::evaluate::AsGenericExpr(std::move(*expr)));
        if (auto actual = dummyArgs.lookupSymbol(symbol)) {
          auto unboxed = builder->createUnboxChar(actual);
          localValue = builder->createEmboxChar(unboxed.first, lenValue);
        } else {
          // TODO: propagate symbol name to FIR.
          localValue = builder->createCharacterTemp(genType(symbol), lenValue);
        }
      } else if (lengthParam.isDeferred()) {
        TODO();
      } else {
        // Assumed
        localValue = dummyArgs.lookupSymbol(symbol);
        assert(localValue &&
               "expected dummy arguments when length not explicit");
      }
      addSymbol(symbol, localValue);
    } else if (!type->AsIntrinsic()) {
      TODO(); // Derived type / polymorphic
    } else {
      if (auto actualValue = dummyArgs.lookupSymbol(symbol))
        addSymbol(symbol, actualValue);
      else
        createTemp(toLocation(), symbol);
    }
    if (const auto *details =
            symbol.detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
      // For now, only allow compile time constant shapes that do no require
      // to evaluate bounds expression here. Assumed size are also supported.
      if (!isConstantShape(details->shape()))
        TODO();
      // handle bounds specification expressions
      if (!details->coshape().empty())
        TODO(); // handle cobounds specification expressions
      if (details->init())
        TODO(); // init
    } else {
      assert(symbol.has<Fortran::semantics::ProcEntityDetails>());
      TODO(); // Procedure pointers
    }
    attempted.erase(symbol);
  }

  /// Instantiate a global variable. If it hasn't already been processed, add
  /// the global to the ModuleOp as a new uniqued symbol and initialize it with
  /// the correct value. It will be referenced on demand using `fir.addr_of`.
  void instantiateGlobal(const Fortran::lower::pft::Variable &var) {
    const auto &sym = var.getSymbol();
    std::string globalName = mangleName(sym);
    fir::GlobalOp global;
    bool isConst = sym.attrs().test(Fortran::semantics::Attr::PARAMETER);
    if (builder->getNamedGlobal(globalName))
      return;
    if (const auto *details =
            sym.detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
      if (details->init()) {
        if (details->IsArray()) {
          TODO();
          return;
        } else if (!sym.GetType()->AsIntrinsic()) {
          TODO(); // Derived type / polymorphic
          return;
        } else
          global = builder->createGlobal(
              toLocation(), genType(sym), globalName, isConst,
              [&](Fortran::lower::FirOpBuilder &builder) {
                auto initVal = genExprValue(details->init().value());
                builder.create<fir::HasValueOp>(toLocation(), initVal);
              });
      } else {
        global = builder->createGlobal(toLocation(), genType(sym), globalName);
      }
      auto addrOf = builder->create<fir::AddrOfOp>(
          toLocation(), global.resultType(), global.getSymbol());
      addSymbol(sym, addrOf);
    } else {
      TODO(); // Procedure pointer
    }
  }

  /// Create a stack slot for a local variable. Precondition: the insertion
  /// point of the builder must be in the entry block, which is currently being
  /// constructed.
  mlir::Value createNewLocal(mlir::Location loc,
                             const Fortran::semantics::Symbol &sym,
                             llvm::ArrayRef<mlir::Value> shape = {}) {
    auto ty = genType(sym);
    auto nm = sym.name().ToString();
    if (shape.size())
      if (auto arrTy = ty.dyn_cast<fir::SequenceType>()) {
        // elide the constant dimensions before construction
        assert(shape.size() == arrTy.getDimension());
        llvm::SmallVector<mlir::Value, 8> args;
        auto typeShape = arrTy.getShape();
        for (unsigned i = 0, end = arrTy.getDimension(); i < end; ++i)
          if (typeShape[i] == fir::SequenceType::getUnknownExtent())
            args.push_back(shape[i]);
        return builder->allocateLocal(loc, ty, nm, args);
      }
    return builder->allocateLocal(loc, ty, nm, shape);
  }

  /// Instantiate a local variable. Precondition: Each variable will be visited
  /// such that if it depends on other variables, the variables upon which it
  /// depends will already have been visited.
  void instantiateLocal(const Fortran::lower::pft::Variable &var) {
    const auto &sym = var.getSymbol();
    const auto loc = toLocation();
    builder->setLocation(loc);
    auto idxTy = builder->getIndexType();
    const auto isDummy = Fortran::semantics::IsDummy(sym);
    SymbolIndexAnalyzer sia(sym);
    sia.analyze();

    if (sia.isTrivial()) {
      if (isDummy) {
        // This is an argument.
        assert(lookupSymbol(sym) && "must already be in map");
        return;
      }
      // TODO: What about lower host-associated variables? (They probably need
      // to be handled as dummy parameters.)

      // Otherwise, it's a local variable.
      auto local = createNewLocal(loc, sym);
      addSymbol(sym, local);
      return;
    }

    // The non-trivial cases are when we have an argument or local that has a
    // repetition value. Arguments might be passed as simple pointers and need
    // to be cast to a multi-dimensional array with constant bounds (possibly
    // with a missing column), bounds computed in the callee (here), or with
    // bounds from the caller (boxed somewhere else). Locals have the same
    // properties except they are never boxed arguments from the caller and
    // never having a missing column size.
    mlir::Value addr = lookupSymbol(sym);
    mlir::Value len{};
    bool mustBeDummy = false;

    if (sia.isChar) {
      // if element type is a CHARACTER, determine the LEN value
      if (isDummy) {
        auto unboxchar = builder->createUnboxChar(addr);
        auto boxAddr = unboxchar.first;
        if (auto c = sia.getCharLenConst()) {
          // Set/override LEN with a constant
          len = builder->createIntegerConstant(idxTy, *c);
          addr = builder->createEmboxChar(boxAddr, len);
        } else if (auto e = sia.getCharLenExpr()) {
          // Set/override LEN with an expression
          len = genExprValue(*e);
          addr = builder->createEmboxChar(boxAddr, len);
        } else {
          // LEN is from the boxchar
          len = unboxchar.second;
          mustBeDummy = true;
        }
        // XXX: Subsequent lowering expects a CHARACTER variable to be in a
        // boxchar. We assert that here. We might want to reconsider this
        // precondition.
        assert(addr.getType().isa<fir::BoxCharType>());
      } else {
        // local CHARACTER variable
        if (auto c = sia.getCharLenConst()) {
          len = builder->createIntegerConstant(idxTy, *c);
        } else {
          auto e = sia.getCharLenExpr();
          assert(e && "CHARACTER variable must have LEN parameter");
          len = genExprValue(*e);
        }
        assert(!addr);
      }
    }

    if (sia.isArray) {
      // if object is an array process the lower bound and extent values
      llvm::SmallVector<Fortran::lower::SymIndex::Bounds, 8> bounds;
      mustBeDummy = !isExplicitShape(sym);
      if (sia.staticSize) {
        // object shape is constant
        auto castTy = fir::ReferenceType::get(genType(sym));
        if (addr)
          addr = builder->create<fir::ConvertOp>(loc, castTy, addr);
        if (sia.lboundIsAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value, 8> shape;
          for (auto i : sia.staticShape)
            shape.push_back(builder->createIntegerConstant(idxTy, i));
          if (sia.isChar) {
            if (isDummy) {
              localSymbols.addCharSymbolWithShape(sym, addr, len, shape, true);
              return;
            }
            // local CHARACTER array with constant size
            auto local = createNewLocal(loc, sym);
            localSymbols.addCharSymbolWithShape(sym, local, len, shape);
            return;
          }
          if (isDummy) {
            localSymbols.addSymbolWithShape(sym, addr, shape, true);
            return;
          }
          // local array with constant size
          auto local = createNewLocal(loc, sym);
          localSymbols.addSymbolWithShape(sym, local, shape);
          return;
        }
      } else {
        // cast to the known constant parts from the declaration
        auto castTy = fir::ReferenceType::get(genType(sym));
        if (addr) {
          // XXX: special handling for boxchar; see proviso above
          if (auto box =
                  dyn_cast_or_null<fir::EmboxCharOp>(addr.getDefiningOp()))
            addr = builder->create<fir::ConvertOp>(loc, castTy, box.memref());
          else
            addr = builder->create<fir::ConvertOp>(loc, castTy, addr);
        }
      }
      // construct constants and populate `bounds`
      for (const auto &i : llvm::zip(sia.staticLBound, sia.staticShape)) {
        auto fst = builder->createIntegerConstant(idxTy, std::get<0>(i));
        auto snd = builder->createIntegerConstant(idxTy, std::get<1>(i));
        bounds.emplace_back(fst, snd);
      }

      // default array case: populate `bounds` with lower and extent values
      for (const auto &spec : sia.dynamicBound) {
        auto low = spec->lbound().GetExplicit();
        auto high = spec->ubound().GetExplicit();
        if (low && high) {
          // let the folder deal with the common `ub - 1 + 1` case
          auto lb = genExprValue(Fortran::semantics::SomeExpr{*low});
          auto ub = genExprValue(Fortran::semantics::SomeExpr{*high});
          auto ty = ub.getType();
          auto diff = builder->create<mlir::SubIOp>(loc, ty, ub, lb);
          auto one = builder->createIntegerConstant(ty, 1);
          auto sz = builder->create<mlir::AddIOp>(loc, ty, diff, one);
          auto idx = builder->create<fir::ConvertOp>(loc, idxTy, sz);
          bounds.emplace_back(lb, idx);
          continue;
        }
        if (low && spec->ubound().isAssumed()) {
          // An assumed size array. The extent is not computed.
          auto lb = genExprValue(Fortran::semantics::SomeExpr{*low});
          bounds.emplace_back(lb, mlir::Value{});
        }
        break;
      }

      auto unzipInto =
          [&](llvm::SmallVectorImpl<mlir::Value> &shape,
              llvm::ArrayRef<Fortran::lower::SymIndex::Bounds> bounds) {
            std::for_each(bounds.begin(), bounds.end(), [&](const auto &pair) {
              mlir::Value second;
              std::tie(std::ignore, second) = pair;
              shape.push_back(second);
            });
          };
      if (sia.isChar) {
        if (isDummy) {
          localSymbols.addCharSymbolWithBounds(sym, addr, len, bounds, true);
          return;
        }
        // local CHARACTER array with computed bounds
        assert(!mustBeDummy);
        llvm::SmallVector<mlir::Value, 8> shape;
        shape.push_back(len);
        unzipInto(shape, bounds);
        auto local = createNewLocal(loc, sym, shape);
        localSymbols.addCharSymbolWithBounds(sym, local, len, bounds);
        return;
      }
      if (isDummy) {
        localSymbols.addSymbolWithBounds(sym, addr, bounds, true);
        return;
      }
      // local array with computed bounds
      assert(!mustBeDummy);
      llvm::SmallVector<mlir::Value, 8> shape;
      unzipInto(shape, bounds);
      auto local = createNewLocal(loc, sym, shape);
      localSymbols.addSymbolWithBounds(sym, local, bounds);
      return;
    }

    // not an array, so process as scalar argument
    if (sia.isChar) {
      if (isDummy) {
        addCharSymbol(sym, addr, len, true);
        return;
      }
      assert(!mustBeDummy);
      auto charTy = genType(sym);
      auto c = sia.getCharLenConst();
      mlir::Value local = c ? builder->createCharacterTemp(charTy, *c)
                            : builder->createCharacterTemp(charTy, len);
      addCharSymbol(sym, local, len);
      return;
    }
    if (isDummy) {
      addSymbol(sym, addr, true);
      return;
    }
    auto local = createNewLocal(loc, sym);
    addSymbol(sym, local);
  }

  void instantiateVar(const Fortran::lower::pft::Variable &var) {
    if (var.isGlobal()) {
      instantiateGlobal(var);
      return;
    }
    instantiateLocal(var);
  }

  /// Prepare to translate a new function
  void startNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    assert(!builder && "expected nullptr");
    // get mangled name
    std::string name = funit.isMainProgram()
                           ? uniquer.doProgramEntry().str()
                           : mangleName(funit.getSubprogramSymbol());

    // FIXME: do NOT use unknown for the anonymous PROGRAM case. We probably
    // should just stash the location in the funit regardless.
    mlir::Location loc = toLocation(funit.getStartingSourceLoc());
    mlir::FuncOp func =
        Fortran::lower::FirOpBuilder::getNamedFunction(module, name);
    if (!func)
      func = createNewFunction(loc, name, funit.symbol);
    builder = new Fortran::lower::FirOpBuilder(func);
    assert(builder && "FirOpBuilder did not instantiate");
    func.addEntryBlock();
    builder->setInsertionPointToStart(&func.front());

    if (useOldInitializerCode) {
      Fortran::lower::SymMap dummyAssociations;
      // plumb function's arguments
      if (funit.symbol && !funit.isMainProgram()) {
        auto *entryBlock = &func.front();
        const auto &details =
            funit.symbol->get<Fortran::semantics::SubprogramDetails>();
        for (const auto &v :
             llvm::zip(details.dummyArgs(), entryBlock->getArguments())) {
          if (std::get<0>(v)) {
            dummyAssociations.addSymbol(*std::get<0>(v), std::get<1>(v));
          } else {
            TODO(); // handle alternate return
          }
        }

        // Go through the symbol scope and evaluate specification expressions
        llvm::DenseSet<Fortran::semantics::SymbolRef> attempted;
        assert(funit.symbol->scope() && "subprogram symbol must have a scope");
        // TODO: This loop through scope symbols offers no stability guarantee
        // regarding the order. This should not be a problem given how
        // instantiateLocalVariable is implemented, but may harm
        // reproducibility. A solution would be to sort the symbol based on
        // their source location.
        for (const auto &iter : *funit.symbol->scope()) {
          instantiateLocalVariable(iter.second.get(), dummyAssociations,
                                   attempted);
        }

        // if (details.isFunction())
        //  createTemp(toLocation(), details.result());
      }
    } else {
      auto *entryBlock = &func.front();
      if (funit.symbol && !funit.isMainProgram()) {
        const auto &details =
            funit.symbol->get<Fortran::semantics::SubprogramDetails>();
        for (const auto &v :
             llvm::zip(details.dummyArgs(), entryBlock->getArguments())) {
          if (std::get<0>(v)) {
            addSymbol(*std::get<0>(v), std::get<1>(v));
          } else {
            TODO(); // handle alternate return
          }
        }
      }
      for (const auto &var : funit.getOrderedSymbolTable())
        instantiateVar(var);
    }

    // Create most function blocks in advance.
    createEmptyBlocks(funit.evaluationList);

    // Reinstate entry block as the current insertion point.
    builder->setInsertionPointToEnd(&func.front());
  }

  /// Create empty blocks for the current function.
  void createEmptyBlocks(
      std::list<Fortran::lower::pft::Evaluation> &evaluationList) {
    for (auto &eval : evaluationList) {
      if (eval.isNewBlock)
        eval.block = builder->createBlock(&builder->getRegion());
      for (size_t i = 0, n = eval.localBlocks.size(); i < n; ++i)
        eval.localBlocks[i] = builder->createBlock(&builder->getRegion());
      if (eval.isConstruct()) {
        if (eval.lowerAsUnstructured()) {
          createEmptyBlocks(*eval.evaluationList);
        } else {
          // A structured construct that is a target starts a new block.
          Fortran::lower::pft::Evaluation &constructStmt =
              eval.evaluationList->front();
          if (constructStmt.isNewBlock)
            constructStmt.block = builder->createBlock(&builder->getRegion());
        }
      }
    }
  }

  /// Return the predicate: "current block does not have a terminator branch".
  bool blockIsUnterminated() {
    auto *currentBlock = builder->getBlock();
    return currentBlock->empty() || currentBlock->back().isKnownNonTerminator();
  }

  /// Unconditionally switch code insertion to a new block.
  void startBlock(mlir::Block *newBlock) {
    assert(newBlock && "missing block");
    // If the current block does not have a terminator branch,
    // append a fallthrough branch.
    if (blockIsUnterminated())
      genBranch(newBlock);
    builder->setInsertionPointToStart(newBlock);
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

    delete builder;
    builder = nullptr;
    localSymbols.clear();
  }

  /// Lower a procedure-like construct
  void lowerFunc(Fortran::lower::pft::FunctionLikeUnit &funit) {
    startNewFunction(funit);
    // lower this procedure
    for (auto &eval : funit.evaluationList)
      genFIR(eval);
    endNewFunction(funit);
    // recursively lower internal procedures
    for (auto &f : funit.nestedFunctions)
      lowerFunc(f);
  }

  void lowerMod(Fortran::lower::pft::ModuleLikeUnit &mod) {
    // FIXME: do we need to visit the module statements?
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

  mlir::MLIRContext &mlirContext;
  const Fortran::parser::CookedSource *cooked;
  mlir::ModuleOp &module;
  const Fortran::common::IntrinsicTypeDefaultKinds &defaults;
  Fortran::lower::FirOpBuilder *builder = nullptr;
  const fir::KindMapping &kindMap;
  fir::NameUniquer &uniquer;
  Fortran::lower::SymMap localSymbols;
  Fortran::parser::CharBlock currentPosition;
};

} // namespace

void Fortran::lower::LoweringBridge::lower(
    const Fortran::parser::Program &prg, fir::NameUniquer &uniquer,
    const Fortran::semantics::SemanticsContext &semanticsContext) {
  auto pft = Fortran::lower::createPFT(prg, semanticsContext);
  if (dumpBeforeFir)
    Fortran::lower::dumpPFT(llvm::errs(), *pft);
  FirConverter converter{*this, uniquer};
  converter.run(*pft);
}

void Fortran::lower::LoweringBridge::parseSourceFile(llvm::SourceMgr &srcMgr) {
  auto owningRef = mlir::parseSourceFile(srcMgr, context.get());
  module.reset(new mlir::ModuleOp(owningRef.get().getOperation()));
  owningRef.release();
}

Fortran::lower::LoweringBridge::LoweringBridge(
    const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds,
    const Fortran::parser::CookedSource *cooked)
    : defaultKinds{defaultKinds}, cooked{cooked},
      context{std::make_unique<mlir::MLIRContext>()}, kindMap{context.get()} {
  module = std::make_unique<mlir::ModuleOp>(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get())));
}
