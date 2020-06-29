//===-- Bridge.cc -- bridge to lower to MLIR ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Bridge.h"
#include "../../runtime/iostat.h"
#include "SymbolMap.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/Coarray.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/IO.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/OpenMP.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Parser.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MD5.h"

#undef TODO
#define TODO() llvm_unreachable("not yet implemented");

static llvm::cl::opt<bool> dumpBeforeFir(
    "fdebug-dump-pre-fir", llvm::cl::init(false),
    llvm::cl::desc("dump the Pre-FIR tree prior to FIR generation"));

static llvm::cl::opt<std::size_t>
    nameLengthHashSize("length-to-hash-string-literal",
                       llvm::cl::desc("string literals that exceed this length"
                                      " will use a hash value as their symbol "
                                      "name"),
                       llvm::cl::init(32));

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
  return det && det->IsArray();
}

static bool isExplicitShape(const Fortran::semantics::Symbol &sym) {
  const auto *det = sym.detailsIf<Fortran::semantics::ObjectEntityDetails>();
  return det && det->IsArray() && det->shape().IsExplicitShape();
}

// Retrieve a copy of a character literal string from a SomeExpr.
template <int KIND>
llvm::Optional<std::tuple<std::string, std::size_t>> getCharacterLiteralCopy(
    const Fortran::evaluate::Expr<
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Character, KIND>>
        &x) {
  if (const auto *con =
          Fortran::evaluate::UnwrapConstantValue<Fortran::evaluate::Type<
              Fortran::common::TypeCategory::Character, KIND>>(x))
    if (auto val = con->GetScalarValue())
      return std::tuple<std::string, std::size_t>{
          std::string{(const char *)val->c_str(),
                      KIND * (std::size_t)con->LEN()},
          (std::size_t)con->LEN()};
  return llvm::None;
}
llvm::Optional<std::tuple<std::string, std::size_t>> getCharacterLiteralCopy(
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeCharacter> &x) {
  return std::visit([](const auto &e) { return getCharacterLiteralCopy(e); },
                    x.u);
}
llvm::Optional<std::tuple<std::string, std::size_t>> getCharacterLiteralCopy(
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &x) {
  if (const auto *e = Fortran::evaluate::UnwrapExpr<
          Fortran::evaluate::Expr<Fortran::evaluate::SomeCharacter>>(x))
    return getCharacterLiteralCopy(*e);
  return llvm::None;
}
template <typename A>
llvm::Optional<std::tuple<std::string, std::size_t>>
getCharacterLiteralCopy(const std::optional<A> &x) {
  if (x)
    return getCharacterLiteralCopy(*x);
  return llvm::None;
}

namespace {
struct SymbolBoxAnalyzer {
  using FromBox = std::monostate;

  explicit SymbolBoxAnalyzer(const Fortran::semantics::Symbol &sym)
      : sym{sym} {}
  SymbolBoxAnalyzer() = delete;
  SymbolBoxAnalyzer(const SymbolBoxAnalyzer &) = delete;

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
      : bridge{bridge}, uniquer{uniquer}, foldingContext{
                                              bridge.createFoldingContext()} {}
  virtual ~FirConverter() = default;

  /// Convert the PFT to FIR
  void run(Fortran::lower::pft::Program &pft) {
    // do translation
    for (auto &u : pft.getUnits()) {
      std::visit(
          Fortran::common::visitors{
              [&](Fortran::lower::pft::FunctionLikeUnit &f) { lowerFunc(f); },
              [&](Fortran::lower::pft::ModuleLikeUnit &m) { lowerMod(m); },
              [&](Fortran::lower::pft::BlockDataUnit &) {
                mlir::emitError(toLocation(), "BLOCK DATA not handled");
                exit(1);
              },
          },
          u);
    }
  }

  //===--------------------------------------------------------------------===//
  // AbstractConverter overrides
  //===--------------------------------------------------------------------===//

  mlir::Value getSymbolAddress(Fortran::lower::SymbolRef sym) override final {
    return fir::getBase(lookupSymbol(sym));
  }

  mlir::Value genExprAddr(const Fortran::lower::SomeExpr &expr,
                          mlir::Location *loc = nullptr) override final {
    return createFIRAddr(loc ? *loc : toLocation(), &expr);
  }
  mlir::Value genExprValue(const Fortran::lower::SomeExpr &expr,
                           mlir::Location *loc = nullptr) override final {
    return createFIRExpr(loc ? *loc : toLocation(), &expr);
  }
  Fortran::evaluate::FoldingContext &getFoldingContext() override final {
    return foldingContext;
  }

  mlir::Type genType(const Fortran::evaluate::DataRef &data) override final {
    return Fortran::lower::translateDataRefToFIRType(
        &getMLIRContext(), bridge.getDefaultKinds(), data);
  }
  mlir::Type genType(const Fortran::lower::SomeExpr &expr) override final {
    return Fortran::lower::translateSomeExprToFIRType(
        &getMLIRContext(), bridge.getDefaultKinds(), &expr);
  }
  mlir::Type genType(const Fortran::lower::pft::Variable &var) override final {
    return Fortran::lower::translateVariableToFIRType(
        &getMLIRContext(), bridge.getDefaultKinds(), var);
  }
  mlir::Type genType(Fortran::lower::SymbolRef sym) override final {
    return Fortran::lower::translateSymbolToFIRType(
        &getMLIRContext(), bridge.getDefaultKinds(), sym);
  }
  mlir::Type genType(Fortran::common::TypeCategory tc,
                     int kind) override final {
    return Fortran::lower::getFIRType(&getMLIRContext(),
                                      bridge.getDefaultKinds(), tc, kind);
  }
  mlir::Type genType(Fortran::common::TypeCategory tc) override final {
    return Fortran::lower::getFIRType(&getMLIRContext(),
                                      bridge.getDefaultKinds(), tc);
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
    return Fortran::lower::mangle::mangleName(uniquer, symbol);
  }

  std::string uniqueCGIdent(llvm::StringRef prefix,
                            llvm::StringRef name) override final {
    // For "long" identifiers use a hash value
    if (name.size() > nameLengthHashSize) {
      llvm::MD5 hash;
      hash.update(name);
      llvm::MD5::MD5Result result;
      hash.final(result);
      llvm::SmallString<32> str;
      llvm::MD5::stringifyResult(result, str);
      std::string hashName = prefix.str();
      hashName.append(".").append(str.c_str());
      return uniquer.doGenerated(hashName);
    }
    // "Short" identifiers use a reversible hex string
    std::string nm = prefix.str();
    return uniquer.doGenerated(nm.append(".").append(llvm::toHex(name)));
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
    return cat == Fortran::common::TypeCategory::Integer ||
           cat == Fortran::common::TypeCategory::Real ||
           cat == Fortran::common::TypeCategory::Complex ||
           cat == Fortran::common::TypeCategory::Logical;
  }

  bool isCharacterCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Character;
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
    auto loc = toLocation();
    auto bcc = builder->createConvert(loc, builder->getI1Type(), cond);
    builder->create<mlir::CondBranchOp>(loc, bcc, trueTarget, llvm::None,
                                        falseTarget, llvm::None);
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
    // TODO: This should probably look at the callee interface result instead
    // to know what must be returned.
    mlir::Value retval = resultRef;
    if (!resultRef.getType().isa<fir::BoxCharType>())
      retval = builder->create<fir::LoadOp>(toLocation(), resultRef);
    builder->create<mlir::ReturnOp>(toLocation(), retval);
  }

  /// Argument \p funit is a subroutine that has alternate return specifiers.
  /// Return the variable that contains the result value of a call to \p funit.
  const mlir::Value
  getAltReturnResult(const Fortran::lower::pft::FunctionLikeUnit &funit) {
    const auto &symbol = funit.getSubprogramSymbol();
    assert(Fortran::semantics::HasAlternateReturns(symbol) &&
           "subroutine does not have alternate returns");
    const auto returnValue = lookupSymbol(symbol);
    assert(returnValue && "missing alternate return value");
    return returnValue;
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
      mlir::Value retval =
          builder->create<fir::LoadOp>(toLocation(), getAltReturnResult(funit));
      builder->create<mlir::ReturnOp>(toLocation(), retval);
    } else {
      genExitRoutine();
    }
  }

  //
  // Statements that have control-flow semantics
  //

  template <typename A>
  std::pair<mlir::OpBuilder::InsertPoint, fir::WhereOp>
  genWhereCondition(const A *stmt, bool withElse = true) {
    auto cond = genExprValue(*Fortran::semantics::GetExpr(
        std::get<Fortran::parser::ScalarLogicalExpr>(stmt->t)));
    auto bcc = builder->createConvert(toLocation(), builder->getI1Type(), cond);
    auto where = builder->create<fir::WhereOp>(toLocation(), bcc, withElse);
    auto insPt = builder->saveInsertionPoint();
    builder->setInsertionPointToStart(&where.whereRegion().front());
    return {insPt, where};
  }

  mlir::Value genFIRLoopIndex(const Fortran::parser::ScalarExpr &x,
                              mlir::Type t) {
    mlir::Value v = genExprValue(*Fortran::semantics::GetExpr(x));
    return builder->createConvert(toLocation(), t, v);
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
    auto &eval = getEval();
    setCurrentPosition(stmt.v.source);
    assert(stmt.typedCall && "Call was not analyzed");
    Fortran::semantics::SomeExpr expr{*stmt.typedCall};
    // Call statement lowering shares code with function call lowering.
    auto res = createFIRExpr(toLocation(), &expr);
    if (!res)
      return; // "Normal" subroutine call.
    // Call with alternate return specifiers.
    // The call returns an index that selects an alternate return branch target.
    llvm::SmallVector<int64_t, 5> indexList;
    llvm::SmallVector<mlir::Block *, 5> blockList;
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
    builder->create<fir::SelectOp>(toLocation(), res, indexList, blockList);
  }

  void genFIR(const Fortran::parser::IfStmt &stmt) {
    auto &eval = getEval();
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

  void genFIR(const Fortran::parser::ComputedGotoStmt &stmt) {
    auto &eval = getEval();
    mlir::Value selectExpr = genExprValue(*Fortran::semantics::GetExpr(
        std::get<Fortran::parser::ScalarIntExpr>(stmt.t)));
    llvm::SmallVector<int64_t, 10> indexList;
    llvm::SmallVector<mlir::Block *, 10> blockList;
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
    auto &eval = getEval();
    mlir::Value expr = genExprValue(
        *Fortran::semantics::GetExpr(std::get<Fortran::parser::Expr>(stmt.t)));
    auto exprType = expr.getType();
    auto loc = toLocation();
    if (exprType.isSignlessInteger()) {
      // Arithmetic expression has Integer type.  Generate a SelectCaseOp
      // with ranges {(-inf:-1], 0=default, [1:inf)}.
      MLIRContext *context = builder->getContext();
      llvm::SmallVector<mlir::Attribute, 3> attrList;
      llvm::SmallVector<mlir::Value, 3> valueList;
      llvm::SmallVector<mlir::Block *, 3> blockList;
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
    assert(eval.localBlocks.size() == 1 && "missing arithmetic if block");
    mlir::Value sum = builder->create<fir::AddfOp>(loc, expr, expr);
    mlir::Value zero = builder->create<mlir::ConstantOp>(
        loc, exprType, builder->getFloatAttr(exprType, 0.0));
    mlir::Value cond1 =
        builder->create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::OLT, sum, zero);
    genFIRConditionalBranch(cond1, blockOfLabel(eval, std::get<1>(stmt.t)),
                            eval.localBlocks[0]);
    startBlock(eval.localBlocks[0]);
    mlir::Value cond2 =
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

    auto &eval = getEval();
    const auto &symbolLabelMap =
        eval.getOwningProcedure()->assignSymbolLabelMap;
    const auto &symbol = *std::get<Fortran::parser::Name>(stmt.t).symbol;
    auto variable = lookupSymbol(symbol);
    auto loc = toLocation();
    if (!variable)
      variable = createTemp(loc, symbol);
    auto selectExpr = builder->create<fir::LoadOp>(loc, variable);
    auto iter = symbolLabelMap.find(symbol);
    if (iter == symbolLabelMap.end()) {
      // Fail for a nonconforming program unit that does not have any ASSIGN
      // statements.  The front end should check for this.
      mlir::emitError(loc, "(semantics issue) no assigned goto targets");
      exit(1);
    }
    auto labelSet = iter->second;
    llvm::SmallVector<int64_t, 10> indexList;
    llvm::SmallVector<mlir::Block *, 10> blockList;
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
    if (indexList.empty()) {
      for (auto &label : labelSet) {
        addLabel(label);
      }
    }
    // Add a nop/fallthrough branch to the switch for a nonconforming program
    // unit that violates the program requirement above.
    blockList.push_back(eval.nonNopSuccessor().block); // default
    builder->create<fir::SelectOp>(loc, selectExpr, indexList, blockList);
  }

  /// Generate FIR for a DO construct.  There are six variants:
  ///  - unstructured infinite and while loops
  ///  - structured and unstructured increment loops
  ///  - structured and unstructured concurrent loops
  void genFIR(const Fortran::parser::DoConstruct &) {
    auto &eval = getEval();
    bool unstructuredContext = eval.lowerAsUnstructured();
    Fortran::lower::pft::Evaluation &doStmtEval =
        eval.getFirstNestedEvaluation();
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
    for (auto &e : eval.getNestedEvaluations())
      genFIR(e, unstructuredContext);
    setCurrentEval(eval);

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
    auto loc = toLocation();
    mlir::Type type =
        info.isStructured() ? builder->getIndexType() : info.loopVariableType;
    auto lowerValue = genFIRLoopIndex(info.lowerExpr, type);
    auto upperValue = genFIRLoopIndex(info.upperExpr, type);
    info.stepValue =
        info.stepExpr.has_value() ? genFIRLoopIndex(*info.stepExpr, type)
        : info.isStructured()
            ? builder->create<mlir::ConstantIndexOp>(loc, 1)
            : builder->createIntegerConstant(loc, info.loopVariableType, 1);
    assert(info.stepValue && "step value must be set");
    info.loopVariable = createTemp(loc, *info.loopVariableSym);

    // Structured loop - generate fir.loop.
    if (info.isStructured()) {
      // Perform the default initial assignment of the DO variable.
      info.insertionPoint = builder->saveInsertionPoint();
      info.doLoop = builder->create<fir::LoopOp>(
          loc, lowerValue, upperValue, info.stepValue, /*unordered=*/false,
          ArrayRef<mlir::Value>{lowerValue});
      builder->setInsertionPointToStart(info.doLoop.getBody());
      // Always store iteration ssa-value to the DO variable to avoid missing
      // any aliasing. Note that this assignment can only happen when executing
      // an iteration of the loop.
      auto lcv = builder->createConvert(loc, info.loopVariableType,
                                        info.doLoop.getInductionVar());
      builder->create<fir::StoreOp>(loc, lcv, info.loopVariable);
      return;
    }

    // Unstructured loop preheader code - initialize tripVariable, loopVariable.
    auto distance = builder->create<mlir::SubIOp>(loc, upperValue, lowerValue);
    auto adjusted =
        builder->create<mlir::AddIOp>(loc, distance, info.stepValue);
    auto tripCount =
        builder->create<mlir::SignedDivIOp>(loc, adjusted, info.stepValue);
    info.tripVariable = builder->createTemporary(loc, info.loopVariableType);
    builder->create<fir::StoreOp>(loc, tripCount, info.tripVariable);
    builder->create<fir::StoreOp>(loc, lowerValue, info.loopVariable);

    // Unstructured loop header code - generate loop condition.
    startBlock(info.headerBlock);
    mlir::Value tripVariable =
        builder->create<fir::LoadOp>(loc, info.tripVariable);
    mlir::Value zero =
        builder->createIntegerConstant(loc, info.loopVariableType, 0);
    mlir::Value cond = builder->create<mlir::CmpIOp>(
        loc, mlir::CmpIPredicate::sgt, tripVariable, zero);
    genFIRConditionalBranch(cond, info.bodyBlock, info.successorBlock);
  }

  /// Generate FIR to end a structured or unstructured increment loop.
  void genFIRIncrementLoopEnd(IncrementLoopInfo &info) {
    auto loc = toLocation();
    if (info.isStructured()) {
      // End fir.loop.
      mlir::Value inc = builder->create<mlir::AddIOp>(
          loc, info.doLoop.getInductionVar(), info.doLoop.step());
      builder->create<fir::ResultOp>(loc, inc);
      builder->restoreInsertionPoint(info.insertionPoint);
      auto lcv = builder->createConvert(loc, info.loopVariableType,
                                        info.doLoop.getResult(0));
      builder->create<fir::StoreOp>(loc, lcv, info.loopVariable);
      return;
    }

    // Unstructured loop - increment loopVariable.
    mlir::Value loopVariable =
        builder->create<fir::LoadOp>(loc, info.loopVariable);
    loopVariable =
        builder->create<mlir::AddIOp>(loc, loopVariable, info.stepValue);
    builder->create<fir::StoreOp>(loc, loopVariable, info.loopVariable);

    // Unstructured loop - decrement tripVariable.
    mlir::Value tripVariable =
        builder->create<fir::LoadOp>(loc, info.tripVariable);
    mlir::Value one = builder->create<mlir::ConstantOp>(
        loc, builder->getIntegerAttr(info.loopVariableType, 1));
    tripVariable = builder->create<mlir::SubIOp>(loc, tripVariable, one);
    builder->create<fir::StoreOp>(loc, tripVariable, info.tripVariable);
    genBranch(info.headerBlock);
  }

  /// Generate structured or unstructured FIR for an IF construct.
  void genFIR(const Fortran::parser::IfConstruct &) {
    auto &eval = getEval();
    if (eval.lowerAsStructured()) {
      // Structured fir.where nest.
      fir::WhereOp underWhere;
      mlir::OpBuilder::InsertPoint insPt;
      for (auto &e : eval.getNestedEvaluations()) {
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
      setCurrentEval(eval);
      return;
    }

    // Unstructured branch sequence.
    for (auto &e : eval.getNestedEvaluations()) {
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
    setCurrentEval(eval);
  }

  void genFIR(const Fortran::parser::CaseConstruct &) {
    for (auto &e : getEval().getNestedEvaluations())
      genFIR(e);
  }

  /// Lower FORALL construct (See 10.2.4)
  void genFIR(const Fortran::parser::ForallConstruct &forall) {
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
                genFIR(b.statement);
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
    TODO();
  }

  void genFIR(const Fortran::parser::ForallAssignmentStmt &s) {
    std::visit([&](auto &b) { genFIR(b); }, s.u);
  }

  void genFIR(const Fortran::parser::CompilerDirective &) {
    mlir::emitWarning(toLocation(), "ignoring all compiler directives");
  }

  void genFIR(const Fortran::parser::OpenMPConstruct &omp) {
    genOpenMPConstruct(*this, getEval(), omp);
  }

  void genFIR(const Fortran::parser::OmpEndLoopDirective &omp) {
    genOpenMPEndLoop(*this, getEval(), omp);
  }

  void genFIR(const Fortran::parser::SelectCaseStmt &stmt) {
    auto &eval = getEval();
    using ScalarExpr = Fortran::parser::Scalar<Fortran::parser::Expr>;
    MLIRContext *context = builder->getContext();
    auto loc = toLocation();
    auto selectExpr = genExprValue(
        *Fortran::semantics::GetExpr(std::get<ScalarExpr>(stmt.t)));
    auto selectType = selectExpr.getType();
    Fortran::lower::CharacterExprHelper helper{*builder, loc};
    if (helper.isCharacter(selectExpr.getType())) {
      TODO();
    }
    llvm::SmallVector<mlir::Attribute, 10> attrList;
    llvm::SmallVector<mlir::Value, 10> valueList;
    llvm::SmallVector<mlir::Block *, 10> blockList;
    auto *defaultBlock = eval.parentConstruct->constructExit->block;
    using CaseValue = Fortran::parser::Scalar<Fortran::parser::ConstantExpr>;
    auto addValue = [&](const CaseValue &caseValue) {
      const auto *expr = Fortran::semantics::GetExpr(caseValue.thing);
      const auto v = Fortran::evaluate::ToInt64(*expr);
      valueList.push_back(
          v ? builder->createIntegerConstant(loc, selectType, *v)
            : builder->createConvert(loc, selectType, genExprValue(*expr)));
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

  void genFIR(const Fortran::parser::CaseStmt &) {}       // nop
  void genFIR(const Fortran::parser::EndSelectStmt &) {}  // nop
  void genFIR(const Fortran::parser::NonLabelDoStmt &) {} // nop
  void genFIR(const Fortran::parser::EndDoStmt &) {}      // nop
  void genFIR(const Fortran::parser::IfThenStmt &) {}     // nop
  void genFIR(const Fortran::parser::ElseIfStmt &) {}     // nop
  void genFIR(const Fortran::parser::ElseStmt &) {}       // nop
  void genFIR(const Fortran::parser::EndIfStmt &) {}      // nop

  void genFIR(const Fortran::parser::AssociateConstruct &) { TODO(); }
  void genFIR(const Fortran::parser::AssociateStmt &) { TODO(); }
  void genFIR(const Fortran::parser::EndAssociateStmt &) { TODO(); }

  void genFIR(const Fortran::parser::BlockConstruct &) { TODO(); }
  void genFIR(const Fortran::parser::BlockStmt &) { TODO(); }
  void genFIR(const Fortran::parser::EndBlockStmt &) { TODO(); }

  void genFIR(const Fortran::parser::ChangeTeamConstruct &construct) {
    genChangeTeamConstruct(*this, getEval(), construct);
  }
  void genFIR(const Fortran::parser::ChangeTeamStmt &stmt) {
    genChangeTeamStmt(*this, getEval(), stmt);
  }
  void genFIR(const Fortran::parser::EndChangeTeamStmt &stmt) {
    genEndChangeTeamStmt(*this, getEval(), stmt);
  }

  void genFIR(const Fortran::parser::CriticalConstruct &) { TODO(); }
  void genFIR(const Fortran::parser::CriticalStmt &) { TODO(); }
  void genFIR(const Fortran::parser::EndCriticalStmt &) { TODO(); }

  void genFIR(const Fortran::parser::SelectRankConstruct &) { TODO(); }
  void genFIR(const Fortran::parser::SelectRankStmt &) { TODO(); }
  void genFIR(const Fortran::parser::SelectRankCaseStmt &) { TODO(); }

  void genFIR(const Fortran::parser::SelectTypeConstruct &) { TODO(); }
  void genFIR(const Fortran::parser::SelectTypeStmt &) { TODO(); }
  void genFIR(const Fortran::parser::TypeGuardStmt &) { TODO(); }

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
    genIoConditionBranches(
        getEval(), std::get<std::list<Fortran::parser::InquireSpec>>(stmt.u),
        iostat);
  }
  void genFIR(const Fortran::parser::OpenStmt &stmt) {
    auto iostat = genOpenStatement(*this, stmt);
    genIoConditionBranches(getEval(), stmt.v, iostat);
  }
  void genFIR(const Fortran::parser::PrintStmt &stmt) {
    auto &owningProc = *getEval().getOwningProcedure();
    genPrintStatement(*this, stmt, owningProc.labelEvaluationMap,
                      owningProc.assignSymbolLabelMap);
  }
  void genFIR(const Fortran::parser::ReadStmt &stmt) {
    auto &owningProc = *getEval().getOwningProcedure();
    auto iostat = genReadStatement(*this, stmt, owningProc.labelEvaluationMap,
                                   owningProc.assignSymbolLabelMap);
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
    auto &owningProc = *getEval().getOwningProcedure();
    auto iostat = genWriteStatement(*this, stmt, owningProc.labelEvaluationMap,
                                    owningProc.assignSymbolLabelMap);
    genIoConditionBranches(getEval(), stmt.controls, iostat);
  }

  template <typename A>
  void genIoConditionBranches(Fortran::lower::pft::Evaluation &eval,
                              const A &specList, mlir::Value iostat) {
    if (!iostat)
      return;

    mlir::Block *endBlock{};
    mlir::Block *eorBlock{};
    mlir::Block *errBlock{};
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
    llvm::SmallVector<int64_t, 5> indexList;
    llvm::SmallVector<mlir::Block *, 4> blockList;
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

  void genFIR(const Fortran::parser::AllocateStmt &) { TODO(); }

  void genFIR(const Fortran::parser::DeallocateStmt &) { TODO(); }

  /// Nullify pointer object list
  ///
  /// For each pointer object, reset the pointer to a disassociated status.
  /// We do this by setting each pointer to null.
  void genFIR(const Fortran::parser::NullifyStmt &stmt) {
    auto loc = toLocation();
    for (auto &po : stmt.v) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::Name &sym) {
                auto ty = genType(*sym.symbol);
                auto load = builder->create<fir::LoadOp>(
                    loc, lookupSymbol(*sym.symbol));
                auto idxTy = builder->getIndexType();
                auto zero = builder->create<mlir::ConstantOp>(
                    loc, idxTy, builder->getIntegerAttr(idxTy, 0));
                auto cast = builder->createConvert(loc, ty, zero);
                builder->create<fir::StoreOp>(loc, cast, load);
              },
              [&](const Fortran::parser::StructureComponent &) { TODO(); },
          },
          po.u);
    }
  }

  //===--------------------------------------------------------------------===//

  void genFIR(const Fortran::parser::ContinueStmt &) {
    // do nothing
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

  fir::LoopOp createLoopNest(llvm::SmallVectorImpl<mlir::Value> &lcvs,
                             const Fortran::evaluate::Shape &shape) {
    auto loc = toLocation();
    auto idxTy = builder->getIndexType();
    auto zero = builder->createIntegerConstant(loc, idxTy, 0);
    auto one = builder->createIntegerConstant(loc, idxTy, 1);
    llvm::SmallVector<mlir::Value, 8> extents;

    for (auto s : shape) {
      if (s.has_value()) {
        auto ub = builder->createConvert(
            loc, idxTy,
            genExprValue(Fortran::evaluate::AsGenericExpr(std::move(*s))));
        auto up = builder->create<mlir::SubIOp>(loc, ub, one);
        extents.push_back(up);
      } else {
        TODO();
      }
    }
    // Iteration space is created with outermost columns, innermost rows
    std::reverse(extents.begin(), extents.end());
    fir::LoopOp inner;
    auto insPt = builder->saveInsertionPoint();
    for (auto e : extents) {
      if (inner)
        builder->setInsertionPointToStart(inner.getBody());
      auto loop = builder->create<fir::LoopOp>(loc, zero, e, one);
      lcvs.push_back(loop.getInductionVar());
      if (!inner)
        insPt = builder->saveInsertionPoint();
      inner = loop;
    }
    builder->restoreInsertionPoint(insPt);
    std::reverse(lcvs.begin(), lcvs.end());
    return inner;
  }

  mlir::OpBuilder::InsertPoint
  genPrelude(llvm::SmallVectorImpl<mlir::Value> &lcvs, bool isHeap,
             const Fortran::lower::SomeExpr &lhs,
             const Fortran::lower::SomeExpr &rhs,
             const Fortran::evaluate::Shape &shape) {
    if (isHeap) {
      // does this require a dealloc and realloc?
    }
    if (/*needToMakeCopies*/ false) {
      // make copies
    }
    // create the loop nest
    auto innerLoop = createLoopNest(lcvs, shape);
    assert(innerLoop);
    auto insPt = builder->saveInsertionPoint();
    // move insertion point inside loop nest
    builder->setInsertionPointToStart(innerLoop.getBody());
    return insPt;
  }

  void genPostlude(bool isHeap, const Fortran::lower::SomeExpr &lhs,
                   const Fortran::lower::SomeExpr &rhs,
                   mlir::OpBuilder::InsertPoint insPt) {
    builder->restoreInsertionPoint(insPt);
    if (/*copiesWereMade*/ false) {
      // free buffers
    }
  }

  fir::ExtendedValue genExprEleValue(const Fortran::lower::SomeExpr &expr,
                                     llvm::ArrayRef<mlir::Value> lcvs) {
    return createSomeExtendedExpression(toLocation(), *this, expr, localSymbols,
                                        lcvs);
  }

  fir::ExtendedValue genExprEleAddr(const Fortran::lower::SomeExpr &expr,
                                    llvm::ArrayRef<mlir::Value> lcvs) {
    return createSomeExtendedAddress(toLocation(), *this, expr, localSymbols,
                                     lcvs);
  }

  /// Shared for both assignments and pointer assignments.
  void genAssignment(const Fortran::evaluate::Assignment &assign) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::Assignment::Intrinsic &) {
              auto loc = toLocation();
              const auto *sym =
                  Fortran::evaluate::UnwrapWholeSymbolDataRef(assign.lhs);
              // Assignment of allocatable are more complex, the lhs may need to
              // be deallocated/reallocated. See Fortran 2018 10.2.1.3 p3
              const bool isHeap =
                  sym && Fortran::semantics::IsAllocatable(*sym);
              // Target of the pointer must be assigned. See Fortran
              // 2018 10.2.1.3 p2
              const bool isPointer = sym && Fortran::semantics::IsPointer(*sym);
              auto lhsType = assign.lhs.GetType();
              assert(lhsType && "lhs cannot be typeless");

              if (assign.lhs.Rank() > 0 || (assign.rhs.Rank() > 0 && isHeap)) {
                // Array assignment
                // See Fortran 2018 10.2.1.3 p5, p6, and p7
                auto shape = getShape(assign.lhs);
                assert(shape.has_value() && "array without shape");
                llvm::SmallVector<mlir::Value, 8> lcvs;
                auto insPt =
                    genPrelude(lcvs, isHeap, assign.lhs, assign.rhs, *shape);
                auto valBox = genExprEleValue(assign.rhs, lcvs);
                auto addrBox = genExprEleAddr(assign.lhs, lcvs);
                builder->create<fir::StoreOp>(loc, fir::getBase(valBox),
                                              fir::getBase(addrBox));
                genPostlude(isHeap, assign.lhs, assign.rhs, insPt);
                return;
              }

              // Scalar assignment
              if (isHeap) {
                TODO();
              }
              if (isNumericScalarCategory(lhsType->category())) {
                // Fortran 2018 10.2.1.3 p8 and p9
                // Conversions should have been inserted by semantic analysis,
                // but they can be incorrect between the rhs and lhs. Correct
                // that here.
                mlir::Value addr = isPointer ? genExprValue(assign.lhs)
                                             : genExprAddr(assign.lhs);
                auto val = genExprValue(assign.rhs);
                auto toTy = fir::dyn_cast_ptrEleTy(addr.getType());
                auto cast = builder->convertWithSemantics(loc, toTy, val);
                builder->create<fir::StoreOp>(loc, cast, addr);
                return;
              }
              if (isCharacterCategory(lhsType->category())) {
                // Fortran 2018 10.2.1.3 p10 and p11
                // Generating value for lhs to get fir.boxchar.
                auto lhs = genExprValue(assign.lhs);
                auto rhs = genExprValue(assign.rhs);
                Fortran::lower::CharacterExprHelper{*builder, loc}.createAssign(
                    lhs, rhs);
                return;
              }
              if (lhsType->category() ==
                  Fortran::common::TypeCategory::Derived) {
                // Fortran 2018 10.2.1.3 p12 and p13
                TODO();
              }
              llvm_unreachable("unknown category");
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
        assign.u);
  }

  void genFIR(const Fortran::parser::WhereConstruct &) { TODO(); }
  void genFIR(const Fortran::parser::WhereConstructStmt &) { TODO(); }
  void genFIR(const Fortran::parser::MaskedElsewhereStmt &) { TODO(); }
  void genFIR(const Fortran::parser::ElsewhereStmt &) { TODO(); }
  void genFIR(const Fortran::parser::EndWhereStmt &) { TODO(); }
  void genFIR(const Fortran::parser::WhereStmt &) { TODO(); }

  void genFIR(const Fortran::parser::ForallConstructStmt &) { TODO(); }
  void genFIR(const Fortran::parser::EndForallStmt &) { TODO(); }
  void genFIR(const Fortran::parser::ForallStmt &) { TODO(); }

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
    auto variable = lookupSymbol(symbol);
    auto loc = toLocation();
    if (!variable)
      variable = createTemp(loc, symbol);
    const auto labelValue = builder->createIntegerConstant(
        loc, genType(symbol), std::get<Fortran::parser::Label>(stmt.t));
    builder->create<fir::StoreOp>(loc, labelValue, variable);
  }

  void genFIR(const Fortran::parser::FormatStmt &) {
    // do nothing.

    // FORMAT statements have no semantics. They may be lowered if used by a
    // data transfer statement.
  }

  void genFIR(const Fortran::parser::EntryStmt &) {
    // FIXME: Need to lower this for F77.
    mlir::emitError(toLocation(), "ENTRY statement is not handled.");
    exit(1);
  }

  void genFIR(const Fortran::parser::PauseStmt &stmt) {
    genPauseStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::DataStmt &) {
    // do nothing. The front-end converts to data initializations.
  }

  void genFIR(const Fortran::parser::NamelistStmt &) { TODO(); }

  // call FAIL IMAGE in runtime
  void genFIR(const Fortran::parser::FailImageStmt &stmt) {
    genFailImageStatement(*this);
  }

  // call STOP, ERROR STOP in runtime
  void genFIR(const Fortran::parser::StopStmt &stmt) {
    genStopStatement(*this, stmt);
  }

  // gen expression, if any; share code with END of procedure
  void genFIR(const Fortran::parser::ReturnStmt &stmt) {
    auto &eval = getEval();
    auto *funit = eval.getOwningProcedure();
    assert(funit && "not inside main program, function or subroutine");
    if (funit->isMainProgram()) {
      genExitRoutine();
      return;
    }
    auto loc = toLocation();
    if (stmt.v) {
      // Alternate return statement -- assign alternate return index.
      auto expr = Fortran::semantics::GetExpr(*stmt.v);
      assert(expr && "missing alternate return expression");
      auto altReturnIndex = builder->createConvert(loc, builder->getIndexType(),
                                                   genExprValue(*expr));
      builder->create<fir::StoreOp>(loc, altReturnIndex,
                                    getAltReturnResult(*funit));
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
    genBranch(getEval().controlSuccessor->block);
  }
  void genFIR(const Fortran::parser::ExitStmt &) {
    genBranch(getEval().controlSuccessor->block);
  }
  void genFIR(const Fortran::parser::GotoStmt &) {
    genBranch(getEval().controlSuccessor->block);
  }

  /// Generate the FIR for the Evaluation `eval`.
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              bool unstructuredContext = true) {
    if (eval.skip)
      return; // rhs of {Forall,If,Where}Stmt has already been processed

    setCurrentPosition(eval.position);
    if (unstructuredContext) {
      // When transitioning from unstructured to structured code,
      // the structured code could be a target that starts a new block.
      maybeStartBlock(eval.isConstruct() && eval.lowerAsStructured()
                          ? eval.getFirstNestedEvaluation().block
                          : eval.block);
    }

    setCurrentEval(eval);
    eval.visit([&](const auto &stmt) { genFIR(stmt); });

    if (unstructuredContext && blockIsUnterminated()) {
      // Exit from an unstructured IF or SELECT construct block.
      Fortran::lower::pft::Evaluation *successor{};
      if (eval.isActionStmt()) {
        successor = eval.controlSuccessor;
      } else if (eval.isConstruct() &&
                 eval.getLastNestedEvaluation()
                     .lexicalSuccessor->isIntermediateConstructStmt()) {
        successor = eval.constructExit;
      }
      if (successor && successor->block)
        genBranch(successor->block);
    }
  }

  /// Instantiate a global variable. If it hasn't already been processed, add
  /// the global to the ModuleOp as a new uniqued symbol and initialize it with
  /// the correct value. It will be referenced on demand using `fir.addr_of`.
  void instantiateGlobal(const Fortran::lower::pft::Variable &var) {
    const auto &sym = var.getSymbol();
    std::string globalName = mangleName(sym);
    fir::GlobalOp global;
    bool isConst = sym.attrs().test(Fortran::semantics::Attr::PARAMETER);
    auto loc = toLocation();
    // FIXME: name returned does not consider subprogram's scope, is not unique
    if (builder->getNamedGlobal(globalName))
      return;
    if (const auto *details =
            sym.detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
      if (details->init()) {
        if (!sym.GetType()->AsIntrinsic()) {
          TODO(); // Derived type / polymorphic
        }
        auto symTy = genType(var);
        if (symTy.isa<fir::CharacterType>()) {
          if (auto chLit = getCharacterLiteralCopy(details->init().value())) {
            fir::SequenceType::Shape len;
            len.push_back(std::get<std::size_t>(*chLit));
            symTy = fir::SequenceType::get(len, symTy);
            auto init = builder->getStringAttr(std::get<std::string>(*chLit));
            auto linkage = builder->getStringAttr("internal");
            global = builder->createGlobal(loc, symTy, globalName, linkage,
                                           init, isConst);
          } else {
            llvm::report_fatal_error(
                "global CHARACTER has unexpected initial value");
          }
        } else {
          global = builder->createGlobal(
              loc, symTy, globalName, isConst,
              [&](Fortran::lower::FirOpBuilder &builder) {
                auto initVal = genExprValue(details->init().value());
                auto castTo = builder.createConvert(loc, symTy, initVal);
                builder.create<fir::HasValueOp>(loc, castTo);
              });
        }
      } else {
        global = builder->createGlobal(loc, genType(var), globalName);
      }
      auto addrOf = builder->create<fir::AddrOfOp>(loc, global.resultType(),
                                                   global.getSymbol());
      SymbolBoxAnalyzer sia(sym);
      sia.analyze();
      if (sia.isTrivial()) {
        addSymbol(sym, addrOf);
        return;
      }
      auto idxTy = builder->getIndexType();
      mlir::Value len;
      if (sia.isChar) {
        auto c = sia.getCharLenConst();
        assert(c.hasValue());
        len = builder->createIntegerConstant(loc, idxTy, *c);
      }
      llvm::SmallVector<mlir::Value, 8> extents;
      llvm::SmallVector<mlir::Value, 8> lbounds;
      if (sia.isArray) {
        assert(sia.staticSize);
        for (auto i : sia.staticShape)
          extents.push_back(builder->createIntegerConstant(loc, idxTy, i));
        if (!sia.lboundIsAllOnes())
          for (auto i : sia.staticLBound)
            lbounds.push_back(builder->createIntegerConstant(loc, idxTy, i));
      }
      if (sia.isChar && sia.isArray) {
        localSymbols.addCharSymbolWithBounds(sym, addrOf, len, extents,
                                             lbounds);
      } else if (sia.isChar) {
        localSymbols.addCharSymbol(sym, addrOf, len);
      } else {
        assert(sia.isArray);
        localSymbols.addSymbolWithBounds(sym, addrOf, extents, lbounds);
      }
    } else {
      TODO(); // Procedure pointer
    }
  }

  /// Create a stack slot for a local variable. Precondition: the insertion
  /// point of the builder must be in the entry block, which is currently being
  /// constructed.
  mlir::Value createNewLocal(mlir::Location loc,
                             const Fortran::lower::pft::Variable &var,
                             llvm::ArrayRef<mlir::Value> shape = {}) {
    auto nm = var.getSymbol().name().ToString();
    auto ty = genType(var);
    if (shape.size())
      if (auto arrTy = ty.dyn_cast<fir::SequenceType>()) {
        // elide the constant dimensions before construction
        assert(shape.size() == arrTy.getDimension());
        llvm::SmallVector<mlir::Value, 8> args;
        auto typeShape = arrTy.getShape();
        for (unsigned i = 0, end = arrTy.getDimension(); i < end; ++i)
          if (typeShape[i] == fir::SequenceType::getUnknownExtent())
            args.push_back(shape[i]);
        return builder->allocateLocal(loc, ty, nm, args, var.isTarget());
      }
    auto local = builder->allocateLocal(loc, ty, nm, shape, var.isTarget());
    // Set local pointer/allocatable to null.
    if (var.isHeapAlloc() || var.isPointer()) {
      auto zero =
          builder->createIntegerConstant(loc, builder->getIndexType(), 0);
      auto null = builder->createConvert(loc, ty, zero);
      builder->create<fir::StoreOp>(loc, null, local);
    }
    return local;
  }

  /// Instantiate a local variable. Precondition: Each variable will be visited
  /// such that if it's properties depend on other variables, the variables upon
  /// which its properties depend will already have been visited.
  void instantiateLocal(const Fortran::lower::pft::Variable &var) {
    const auto &sym = var.getSymbol();
    const auto loc = genLocation(sym.name());
    auto idxTy = builder->getIndexType();
    const auto isDummy = Fortran::semantics::IsDummy(sym);
    const auto isResult = Fortran::semantics::IsFunctionResult(sym);
    Fortran::lower::CharacterExprHelper charHelp{*builder, loc};
    SymbolBoxAnalyzer sia(sym);
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
      auto local = createNewLocal(loc, var);
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
      if (isDummy || isResult) {
        auto unboxchar = charHelp.createUnboxChar(addr);
        auto boxAddr = unboxchar.first;
        if (auto c = sia.getCharLenConst()) {
          // Set/override LEN with a constant
          len = builder->createIntegerConstant(loc, idxTy, *c);
          addr = charHelp.createEmboxChar(boxAddr, len);
        } else if (auto e = sia.getCharLenExpr()) {
          // Set/override LEN with an expression
          len = genExprValue(*e);
          addr = charHelp.createEmboxChar(boxAddr, len);
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
          len = builder->createIntegerConstant(loc, idxTy, *c);
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
      llvm::SmallVector<mlir::Value, 8> extents;
      llvm::SmallVector<mlir::Value, 8> lbounds;
      mustBeDummy = !isExplicitShape(sym) &&
                    !Fortran::semantics::IsAllocatableOrPointer(sym);
      if (sia.staticSize) {
        // object shape is constant
        auto castTy = builder->getRefType(genType(var));
        if (addr)
          addr = builder->createConvert(loc, castTy, addr);
        if (sia.lboundIsAllOnes()) {
          // if lower bounds are all ones, build simple shaped object
          llvm::SmallVector<mlir::Value, 8> shape;
          for (auto i : sia.staticShape)
            shape.push_back(builder->createIntegerConstant(loc, idxTy, i));
          if (sia.isChar) {
            if (isDummy || isResult) {
              localSymbols.addCharSymbolWithShape(sym, addr, len, shape, true);
              return;
            }
            // local CHARACTER array with constant size
            auto local = createNewLocal(loc, var);
            localSymbols.addCharSymbolWithShape(sym, local, len, shape);
            return;
          }
          if (isDummy || isResult) {
            localSymbols.addSymbolWithShape(sym, addr, shape, true);
            return;
          }
          // local array with constant size
          auto local = createNewLocal(loc, var);
          localSymbols.addSymbolWithShape(sym, local, shape);
          return;
        }
      } else {
        // cast to the known constant parts from the declaration
        auto castTy = builder->getRefType(genType(var));
        if (addr) {
          // XXX: special handling for boxchar; see proviso above
          if (auto box =
                  dyn_cast_or_null<fir::EmboxCharOp>(addr.getDefiningOp()))
            addr = builder->createConvert(loc, castTy, box.memref());
          else
            addr = builder->createConvert(loc, castTy, addr);
        }
      }
      // construct constants and populate `bounds`
      for (const auto &i : llvm::zip(sia.staticLBound, sia.staticShape)) {
        auto fst = builder->createIntegerConstant(loc, idxTy, std::get<0>(i));
        auto snd = builder->createIntegerConstant(loc, idxTy, std::get<1>(i));
        lbounds.emplace_back(fst);
        extents.emplace_back(snd);
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
          auto one = builder->createIntegerConstant(loc, ty, 1);
          auto sz = builder->create<mlir::AddIOp>(loc, ty, diff, one);
          auto idx = builder->createConvert(loc, idxTy, sz);
          lbounds.emplace_back(lb);
          extents.emplace_back(idx);
          continue;
        }
        if (low && spec->ubound().isAssumed()) {
          // An assumed size array. The extent is not computed.
          auto lb = genExprValue(Fortran::semantics::SomeExpr{*low});
          lbounds.emplace_back(lb);
          extents.emplace_back(mlir::Value{});
        }
        break;
      }

      if (sia.isChar) {
        if (isDummy || isResult) {
          localSymbols.addCharSymbolWithBounds(sym, addr, len, extents, lbounds,
                                               true);
          return;
        }
        // local CHARACTER array with computed bounds
        assert(!mustBeDummy);
        llvm::SmallVector<mlir::Value, 8> shape;
        shape.push_back(len);
        shape.append(extents.begin(), extents.end());
        auto local = createNewLocal(loc, var, shape);
        localSymbols.addCharSymbolWithBounds(sym, local, len, extents, lbounds);
        return;
      }
      if (isDummy || isResult) {
        localSymbols.addSymbolWithBounds(sym, addr, extents, lbounds, true);
        return;
      }
      // local array with computed bounds
      assert(!mustBeDummy);
      auto local = createNewLocal(loc, var, extents);
      localSymbols.addSymbolWithBounds(sym, local, extents, lbounds);
      return;
    }

    // not an array, so process as scalar argument
    if (sia.isChar) {
      if (isDummy || isResult) {
        addCharSymbol(sym, addr, len, true);
        return;
      }
      assert(!mustBeDummy);
      auto charTy = genType(var);
      auto c = sia.getCharLenConst();
      mlir::Value local = c ? charHelp.createCharacterTemp(charTy, *c)
                            : charHelp.createCharacterTemp(charTy, len);
      addCharSymbol(sym, local, len);
      return;
    }
    if (isDummy) {
      addSymbol(sym, addr, true);
      return;
    }
    auto local = createNewLocal(loc, var);
    addSymbol(sym, local);
  }

  void instantiateVar(const Fortran::lower::pft::Variable &var) {
    if (Fortran::semantics::FindCommonBlockContaining(var.getSymbol())) {
      mlir::emitError(toLocation(),
                      "Common blocks not yet handled in lowering");
      exit(1);
    }
    if (var.isGlobal())
      instantiateGlobal(var);
    else
      instantiateLocal(var);
  }

  void mapDummyAndResults(const Fortran::lower::CalleeInterface &callee) {
    assert(builder && "need a builder at this point");
    using PassBy = Fortran::lower::CalleeInterface::PassEntityBy;
    auto mapPassedEntity = [&](const auto arg) -> void {
      if (arg.passBy == PassBy::AddressAndLength) {
        auto loc = toLocation();
        Fortran::lower::CharacterExprHelper charHelp{*builder, loc};
        auto box = charHelp.createEmboxChar(arg.firArgument, arg.firLength);
        addSymbol(arg.entity.get(), box);
      } else {
        addSymbol(arg.entity.get(), arg.firArgument);
      }
    };
    for (const auto &arg : callee.getPassedArguments()) {
      mapPassedEntity(arg);
    }
    if (auto passedResult = callee.getPassedResult()) {
      mapPassedEntity(*passedResult);
    }
  }

  /// Prepare to translate a new function
  void startNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    assert(!builder && "expected nullptr");
    Fortran::lower::CalleeInterface callee(funit, *this);
    mlir::FuncOp func = callee.getFuncOp();
    builder = new Fortran::lower::FirOpBuilder(func, bridge.getKindMap());
    assert(builder && "FirOpBuilder did not instantiate");
    builder->setInsertionPointToStart(&func.front());

    mapDummyAndResults(callee);

    for (const auto &var : funit.getOrderedSymbolTable())
      instantiateVar(var);

    // Create most function blocks in advance.
    createEmptyBlocks(funit.evaluationList);

    // Reinstate entry block as the current insertion point.
    builder->setInsertionPointToEnd(&func.front());

    if (callee.hasAlternateReturns()) {
      // Create a local temp to hold the alternate return index.
      // Give it an integer index type and the subroutine name (for dumps).
      // Attach it to the subroutine symbol in the localSymbols map.
      // Initialize it to zero, the "fallthrough" alternate return value.
      const auto &symbol = funit.getSubprogramSymbol();
      auto loc = toLocation();
      const auto altResult = builder->createTemporary(
          loc, builder->getIndexType(), symbol.name().ToString());
      addSymbol(symbol, altResult);
      const auto zero =
          builder->createIntegerConstant(loc, builder->getIndexType(), 0);
      builder->create<fir::StoreOp>(loc, zero, altResult);
    }
  }

  /// Create empty blocks for the current function.
  void createEmptyBlocks(
      std::list<Fortran::lower::pft::Evaluation> &evaluationList) {
    for (auto &eval : evaluationList) {
      if (eval.isNewBlock)
        eval.block = builder->createBlock(&builder->getRegion());
      for (size_t i = 0, n = eval.localBlocks.size(); i < n; ++i)
        eval.localBlocks[i] = builder->createBlock(&builder->getRegion());
      if (eval.isConstruct() || eval.isDirective()) {
        if (eval.lowerAsUnstructured()) {
          createEmptyBlocks(eval.getNestedEvaluations());
        } else if (eval.hasNestedEvaluations()) {
          // A structured construct that is a target starts a new block.
          auto &constructStmt = eval.getFirstNestedEvaluation();
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

    // immediately throw away any dead code just created
    mlir::simplifyRegions({builder->getRegion()});
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
  fir::NameUniquer &uniquer;
  Fortran::evaluate::FoldingContext foldingContext;
  Fortran::lower::FirOpBuilder *builder = nullptr;
  Fortran::lower::pft::Evaluation *evalPtr = nullptr;
  Fortran::lower::SymMap localSymbols;
  Fortran::parser::CharBlock currentPosition;
};

} // namespace

Fortran::evaluate::FoldingContext
Fortran::lower::LoweringBridge::createFoldingContext() const {
  return {getDefaultKinds(), getIntrinsicTable()};
}

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
    const Fortran::evaluate::IntrinsicProcTable &intrinsics,
    const Fortran::parser::CookedSource &cooked)
    : defaultKinds{defaultKinds}, intrinsics{intrinsics}, cooked{&cooked},
      context{std::make_unique<mlir::MLIRContext>()}, kindMap{context.get()} {
  context.get()->getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
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
  module = std::make_unique<mlir::ModuleOp>(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get())));
}
