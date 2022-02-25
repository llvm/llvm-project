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
#include "flang/Evaluate/tools.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/IterationSpace.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-bridge"

static llvm::cl::opt<bool> dumpBeforeFir(
    "fdebug-dump-pre-fir", llvm::cl::init(false),
    llvm::cl::desc("dump the Pre-FIR tree prior to FIR generation"));

//===----------------------------------------------------------------------===//
// FirConverter
//===----------------------------------------------------------------------===//

namespace {

/// Traverse the pre-FIR tree (PFT) to generate the FIR dialect of MLIR.
class FirConverter : public Fortran::lower::AbstractConverter {
public:
  explicit FirConverter(Fortran::lower::LoweringBridge &bridge)
      : bridge{bridge}, foldingContext{bridge.createFoldingContext()} {}
  virtual ~FirConverter() = default;

  /// Convert the PFT to FIR.
  void run(Fortran::lower::pft::Program &pft) {
    // Primary translation pass.
    for (Fortran::lower::pft::Program::Units &u : pft.getUnits()) {
      std::visit(
          Fortran::common::visitors{
              [&](Fortran::lower::pft::FunctionLikeUnit &f) { lowerFunc(f); },
              [&](Fortran::lower::pft::ModuleLikeUnit &m) {},
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
  }

  //===--------------------------------------------------------------------===//
  // AbstractConverter overrides
  //===--------------------------------------------------------------------===//

  mlir::Value getSymbolAddress(Fortran::lower::SymbolRef sym) override final {
    return lookupSymbol(sym).getAddr();
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
    return Fortran::lower::createMutableBox(loc, *this, expr, localSymbols);
  }

  Fortran::evaluate::FoldingContext &getFoldingContext() override final {
    return foldingContext;
  }

  mlir::Type genType(const Fortran::evaluate::DataRef &) override final {
    TODO_NOLOC("Not implemented genType DataRef. Needed for more complex "
               "expression lowering");
  }
  mlir::Type genType(const Fortran::lower::SomeExpr &expr) override final {
    return Fortran::lower::translateSomeExprToFIRType(*this, expr);
  }
  mlir::Type genType(Fortran::lower::SymbolRef sym) override final {
    return Fortran::lower::translateSymbolToFIRType(*this, sym);
  }
  mlir::Type genType(Fortran::common::TypeCategory tc) override final {
    TODO_NOLOC("Not implemented genType TypeCategory. Needed for more complex "
               "expression lowering");
  }
  mlir::Type genType(Fortran::common::TypeCategory tc,
                     int kind) override final {
    return Fortran::lower::getFIRType(&getMLIRContext(), tc, kind);
  }
  mlir::Type genType(const Fortran::lower::pft::Variable &var) override final {
    return Fortran::lower::translateVariableToFIRType(*this, var);
  }

  void setCurrentPosition(const Fortran::parser::CharBlock &position) {
    if (position != Fortran::parser::CharBlock{})
      currentPosition = position;
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
    assert(evalPtr && "current evaluation not set");
    return *evalPtr;
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
    // Note that setting the insertion point causes the subsequent function
    // call to check the existence of terminator in the newBlock.
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

  /// Map mlir function block arguments to the corresponding Fortran dummy
  /// variables. When the result is passed as a hidden argument, the Fortran
  /// result is also mapped. The symbol map is used to hold this mapping.
  void mapDummiesAndResults(Fortran::lower::pft::FunctionLikeUnit &funit,
                            const Fortran::lower::CalleeInterface &callee) {
    assert(builder && "require a builder object at this point");
    using PassBy = Fortran::lower::CalleeInterface::PassEntityBy;
    auto mapPassedEntity = [&](const auto arg) -> void {
      if (arg.passBy == PassBy::AddressAndLength) {
        // TODO: now that fir call has some attributes regarding character
        // return, PassBy::AddressAndLength should be retired.
        mlir::Location loc = toLocation();
        fir::factory::CharacterExprHelper charHelp{*builder, loc};
        mlir::Value box =
            charHelp.createEmboxChar(arg.firArgument, arg.firLength);
        addSymbol(arg.entity->get(), box);
      } else {
        if (arg.entity.has_value()) {
          addSymbol(arg.entity->get(), arg.firArgument);
        } else {
          // assert(funit.parentHasHostAssoc());
          // funit.parentHostAssoc().internalProcedureBindings(*this,
          //                                                   localSymbols);
        }
      }
    };
    for (const Fortran::lower::CalleeInterface::PassedEntity &arg :
         callee.getPassedArguments())
      mapPassedEntity(arg);

    // Allocate local skeleton instances of dummies from other entry points.
    // Most of these locals will not survive into final generated code, but
    // some will.  It is illegal to reference them at run time if they do.
    for (const Fortran::semantics::Symbol *arg :
         funit.nonUniversalDummyArguments) {
      if (lookupSymbol(*arg))
        continue;
      mlir::Type type = genType(*arg);
      // TODO: Account for VALUE arguments (and possibly other variants).
      type = builder->getRefType(type);
      addSymbol(*arg, builder->create<fir::UndefOp>(toLocation(), type));
    }
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
  void instantiateVar(const Fortran::lower::pft::Variable &var) {
    Fortran::lower::instantiateVariable(*this, var, localSymbols);
  }

  /// Prepare to translate a new function
  void startNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    assert(!builder && "expected nullptr");
    Fortran::lower::CalleeInterface callee(funit, *this);
    mlir::FuncOp func = callee.addEntryBlockAndMapArguments();
    func.setVisibility(mlir::SymbolTable::Visibility::Public);
    builder = new fir::FirOpBuilder(func, bridge.getKindMap());
    assert(builder && "FirOpBuilder did not instantiate");
    builder->setInsertionPointToStart(&func.front());

    mapDummiesAndResults(funit, callee);

    for (const Fortran::lower::pft::Variable &var :
         funit.getOrderedSymbolTable()) {
      const Fortran::semantics::Symbol &sym = var.getSymbol();
      if (!sym.IsFuncResult() || !funit.primaryResult) {
        instantiateVar(var);
      } else if (&sym == funit.primaryResult) {
        instantiateVar(var);
      }
    }

    // Create most function blocks in advance.
    createEmptyGlobalBlocks(funit.evaluationList);

    // Reinstate entry block as the current insertion point.
    builder->setInsertionPointToEnd(&func.front());
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
  void createEmptyGlobalBlocks(
      std::list<Fortran::lower::pft::Evaluation> &evaluationList) {
    mlir::Region *region = &builder->getRegion();
    for (Fortran::lower::pft::Evaluation &eval : evaluationList) {
      if (eval.isNewBlock)
        eval.block = builder->createBlock(region);
      if (eval.isConstruct() || eval.isDirective()) {
        if (eval.lowerAsUnstructured()) {
          createEmptyGlobalBlocks(eval.getNestedEvaluations());
        } else if (eval.hasNestedEvaluations()) {
          TODO(toLocation(), "Constructs with nested evaluations");
        }
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
    for (Fortran::lower::pft::FunctionLikeUnit &f : funit.nestedFunctions)
      lowerFunc(f); // internal procedure
  }

  mlir::Value hostAssocTupleValue() override final { return hostAssocTuple; }

private:
  FirConverter() = delete;
  FirConverter(const FirConverter &) = delete;
  FirConverter &operator=(const FirConverter &) = delete;

  //===--------------------------------------------------------------------===//
  // Helper member functions
  //===--------------------------------------------------------------------===//

  /// Find the symbol in the local map or return null.
  Fortran::lower::SymbolBox
  lookupSymbol(const Fortran::semantics::Symbol &sym) {
    if (Fortran::lower::SymbolBox v = localSymbols.lookupSymbol(sym))
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

  bool isNumericScalarCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Integer ||
           cat == Fortran::common::TypeCategory::Real ||
           cat == Fortran::common::TypeCategory::Complex ||
           cat == Fortran::common::TypeCategory::Logical;
  }
  bool isCharacterCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Character;
  }
  bool isDerivedCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Derived;
  }

  void genFIRBranch(mlir::Block *targetBlock) {
    assert(targetBlock && "missing unconditional target block");
    builder->create<cf::BranchOp>(toLocation(), targetBlock);
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
    const Fortran::semantics::Symbol &resultSym =
        functionSymbol.get<Fortran::semantics::SubprogramDetails>().result();
    Fortran::lower::SymbolBox resultSymBox = lookupSymbol(resultSym);
    mlir::Location loc = toLocation();
    if (!resultSymBox) {
      mlir::emitError(loc, "failed lowering function return");
      return;
    }
    mlir::Value resultVal = resultSymBox.match(
        [&](const fir::CharBoxValue &x) -> mlir::Value {
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
            TODO(loc, "Convert to actual type");
          return builder->create<fir::LoadOp>(loc, resultRef);
        });
    builder->create<mlir::ReturnOp>(loc, resultVal);
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
    } else {
      genExitRoutine();
    }
  }

  [[maybe_unused]] static bool
  isFuncResultDesignator(const Fortran::lower::SomeExpr &expr) {
    const Fortran::semantics::Symbol *sym =
        Fortran::evaluate::GetFirstSymbol(expr);
    return sym && sym->IsFuncResult();
  }

  static bool isWholeAllocatable(const Fortran::lower::SomeExpr &expr) {
    const Fortran::semantics::Symbol *sym =
        Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(expr);
    return sym && Fortran::semantics::IsAllocatable(*sym);
  }

  void genAssignment(const Fortran::evaluate::Assignment &assign) {
    Fortran::lower::StatementContext stmtCtx;
    mlir::Location loc = toLocation();
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
              if (lhsType->IsPolymorphic() && isWholeAllocatable(assign.lhs))
                TODO(loc, "assignment to polymorphic allocatable");

              // Note: No ad-hoc handling for pointers is required here. The
              // target will be assigned as per 2018 10.2.1.3 p2. genExprAddr
              // on a pointer returns the target address and not the address of
              // the pointer variable.

              if (assign.lhs.Rank() > 0) {
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
              bool lhsIsWholeAllocatable = isWholeAllocatable(assign.lhs);
              llvm::Optional<fir::factory::MutableBoxReallocation> lhsRealloc;
              llvm::Optional<fir::MutableBoxValue> lhsMutableBox;
              auto lhs = [&]() -> fir::ExtendedValue {
                if (lhsIsWholeAllocatable) {
                  lhsMutableBox = genExprMutableBox(loc, assign.lhs);
                  llvm::SmallVector<mlir::Value> lengthParams;
                  if (const fir::CharBoxValue *charBox = rhs.getCharBox())
                    lengthParams.push_back(charBox->getLen());
                  else if (fir::isDerivedWithLengthParameters(rhs))
                    TODO(loc, "assignment to derived type allocatable with "
                              "length parameters");
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
                TODO(toLocation(), "Character assignment");
              } else if (isDerivedCategory(lhsType->category())) {
                TODO(toLocation(), "Derived type assignment");
              } else {
                llvm_unreachable("unknown category");
              }
              if (lhsIsWholeAllocatable)
                fir::factory::finalizeRealloc(
                    *builder, loc, lhsMutableBox.getValue(),
                    /*lbounds=*/llvm::None, /*takeLboundsIfRealloc=*/false,
                    lhsRealloc.getValue());
            },

            // [2] User defined assignment. If the context is a scalar
            // expression then call the procedure.
            [&](const Fortran::evaluate::ProcedureRef &procRef) {
              TODO(toLocation(), "User defined assignment");
            },

            // [3] Pointer assignment with possibly empty bounds-spec. R1035: a
            // bounds-spec is a lower bound value.
            [&](const Fortran::evaluate::Assignment::BoundsSpec &lbExprs) {
              TODO(toLocation(),
                   "Pointer assignment with possibly empty bounds-spec");
            },

            // [4] Pointer assignment with bounds-remapping. R1036: a
            // bounds-remapping is a pair, lower bound and upper bound.
            [&](const Fortran::evaluate::Assignment::BoundsRemapping
                    &boundExprs) {
              TODO(toLocation(), "Pointer assignment with bounds-remapping");
            },
        },
        assign.u);
  }

  /// Lowering of CALL statement
  void genFIR(const Fortran::parser::CallStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    setCurrentPosition(stmt.v.source);
    assert(stmt.typedCall && "Call was not analyzed");
    // Call statement lowering shares code with function call lowering.
    mlir::Value res = Fortran::lower::createSubroutineCall(
        *this, *stmt.typedCall, localSymbols, stmtCtx);
    if (!res)
      return; // "Normal" subroutine call.
  }

  void genFIR(const Fortran::parser::ComputedGotoStmt &stmt) {
    TODO(toLocation(), "ComputedGotoStmt lowering");
  }

  void genFIR(const Fortran::parser::ArithmeticIfStmt &stmt) {
    TODO(toLocation(), "ArithmeticIfStmt lowering");
  }

  void genFIR(const Fortran::parser::AssignedGotoStmt &stmt) {
    TODO(toLocation(), "AssignedGotoStmt lowering");
  }

  void genFIR(const Fortran::parser::DoConstruct &doConstruct) {
    TODO(toLocation(), "DoConstruct lowering");
  }

  void genFIR(const Fortran::parser::IfConstruct &) {
    TODO(toLocation(), "IfConstruct lowering");
  }

  void genFIR(const Fortran::parser::CaseConstruct &) {
    TODO(toLocation(), "CaseConstruct lowering");
  }

  void genFIR(const Fortran::parser::ConcurrentHeader &header) {
    TODO(toLocation(), "ConcurrentHeader lowering");
  }

  void genFIR(const Fortran::parser::ForallAssignmentStmt &stmt) {
    TODO(toLocation(), "ForallAssignmentStmt lowering");
  }

  void genFIR(const Fortran::parser::EndForallStmt &) {
    TODO(toLocation(), "EndForallStmt lowering");
  }

  void genFIR(const Fortran::parser::ForallStmt &) {
    TODO(toLocation(), "ForallStmt lowering");
  }

  void genFIR(const Fortran::parser::ForallConstruct &) {
    TODO(toLocation(), "ForallConstruct lowering");
  }

  void genFIR(const Fortran::parser::ForallConstructStmt &) {
    TODO(toLocation(), "ForallConstructStmt lowering");
  }

  void genFIR(const Fortran::parser::CompilerDirective &) {
    TODO(toLocation(), "CompilerDirective lowering");
  }

  void genFIR(const Fortran::parser::OpenACCConstruct &) {
    TODO(toLocation(), "OpenACCConstruct lowering");
  }

  void genFIR(const Fortran::parser::OpenACCDeclarativeConstruct &) {
    TODO(toLocation(), "OpenACCDeclarativeConstruct lowering");
  }

  void genFIR(const Fortran::parser::OpenMPConstruct &) {
    TODO(toLocation(), "OpenMPConstruct lowering");
  }

  void genFIR(const Fortran::parser::OpenMPDeclarativeConstruct &) {
    TODO(toLocation(), "OpenMPDeclarativeConstruct lowering");
  }

  void genFIR(const Fortran::parser::SelectCaseStmt &) {
    TODO(toLocation(), "SelectCaseStmt lowering");
  }

  void genFIR(const Fortran::parser::AssociateConstruct &) {
    TODO(toLocation(), "AssociateConstruct lowering");
  }

  void genFIR(const Fortran::parser::BlockConstruct &blockConstruct) {
    TODO(toLocation(), "BlockConstruct lowering");
  }

  void genFIR(const Fortran::parser::BlockStmt &) {
    TODO(toLocation(), "BlockStmt lowering");
  }

  void genFIR(const Fortran::parser::EndBlockStmt &) {
    TODO(toLocation(), "EndBlockStmt lowering");
  }

  void genFIR(const Fortran::parser::ChangeTeamConstruct &construct) {
    TODO(toLocation(), "ChangeTeamConstruct lowering");
  }

  void genFIR(const Fortran::parser::ChangeTeamStmt &stmt) {
    TODO(toLocation(), "ChangeTeamStmt lowering");
  }

  void genFIR(const Fortran::parser::EndChangeTeamStmt &stmt) {
    TODO(toLocation(), "EndChangeTeamStmt lowering");
  }

  void genFIR(const Fortran::parser::CriticalConstruct &criticalConstruct) {
    TODO(toLocation(), "CriticalConstruct lowering");
  }

  void genFIR(const Fortran::parser::CriticalStmt &) {
    TODO(toLocation(), "CriticalStmt lowering");
  }

  void genFIR(const Fortran::parser::EndCriticalStmt &) {
    TODO(toLocation(), "EndCriticalStmt lowering");
  }

  void genFIR(const Fortran::parser::SelectRankConstruct &selectRankConstruct) {
    TODO(toLocation(), "SelectRankConstruct lowering");
  }

  void genFIR(const Fortran::parser::SelectRankStmt &) {
    TODO(toLocation(), "SelectRankStmt lowering");
  }

  void genFIR(const Fortran::parser::SelectRankCaseStmt &) {
    TODO(toLocation(), "SelectRankCaseStmt lowering");
  }

  void genFIR(const Fortran::parser::SelectTypeConstruct &selectTypeConstruct) {
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
    TODO(toLocation(), "BackspaceStmt lowering");
  }

  void genFIR(const Fortran::parser::CloseStmt &stmt) {
    TODO(toLocation(), "CloseStmt lowering");
  }

  void genFIR(const Fortran::parser::EndfileStmt &stmt) {
    TODO(toLocation(), "EndfileStmt lowering");
  }

  void genFIR(const Fortran::parser::FlushStmt &stmt) {
    TODO(toLocation(), "FlushStmt lowering");
  }

  void genFIR(const Fortran::parser::InquireStmt &stmt) {
    TODO(toLocation(), "InquireStmt lowering");
  }

  void genFIR(const Fortran::parser::OpenStmt &stmt) {
    TODO(toLocation(), "OpenStmt lowering");
  }

  void genFIR(const Fortran::parser::PrintStmt &stmt) {
    TODO(toLocation(), "PrintStmt lowering");
  }

  void genFIR(const Fortran::parser::ReadStmt &stmt) {
    TODO(toLocation(), "ReadStmt lowering");
  }

  void genFIR(const Fortran::parser::RewindStmt &stmt) {
    TODO(toLocation(), "RewindStmt lowering");
  }

  void genFIR(const Fortran::parser::WaitStmt &stmt) {
    TODO(toLocation(), "WaitStmt lowering");
  }

  void genFIR(const Fortran::parser::WriteStmt &stmt) {
    TODO(toLocation(), "WriteStmt lowering");
  }

  //===--------------------------------------------------------------------===//
  // Memory allocation and deallocation
  //===--------------------------------------------------------------------===//

  void genFIR(const Fortran::parser::AllocateStmt &stmt) {
    TODO(toLocation(), "AllocateStmt lowering");
  }

  void genFIR(const Fortran::parser::DeallocateStmt &stmt) {
    TODO(toLocation(), "DeallocateStmt lowering");
  }

  void genFIR(const Fortran::parser::NullifyStmt &stmt) {
    TODO(toLocation(), "NullifyStmt lowering");
  }

  //===--------------------------------------------------------------------===//

  void genFIR(const Fortran::parser::EventPostStmt &stmt) {
    TODO(toLocation(), "EventPostStmt lowering");
  }

  void genFIR(const Fortran::parser::EventWaitStmt &stmt) {
    TODO(toLocation(), "EventWaitStmt lowering");
  }

  void genFIR(const Fortran::parser::FormTeamStmt &stmt) {
    TODO(toLocation(), "FormTeamStmt lowering");
  }

  void genFIR(const Fortran::parser::LockStmt &stmt) {
    TODO(toLocation(), "LockStmt lowering");
  }

  /// Generate an array assignment.
  /// This is an assignment expression with rank > 0. The assignment may or may
  /// not be in a WHERE and/or FORALL context.
  void genArrayAssignment(const Fortran::evaluate::Assignment &assign,
                          Fortran::lower::StatementContext &stmtCtx) {
    if (isWholeAllocatable(assign.lhs)) {
      // Assignment to allocatables may require the lhs to be
      // deallocated/reallocated. See Fortran 2018 10.2.1.3 p3
      Fortran::lower::createAllocatableArrayAssignment(
          *this, assign.lhs, assign.rhs, explicitIterSpace, implicitIterSpace,
          localSymbols, stmtCtx);
      return;
    }

    // No masks and the iteration space is implied by the array, so create a
    // simple array assignment.
    Fortran::lower::createSomeArrayAssignment(*this, assign.lhs, assign.rhs,
                                              localSymbols, stmtCtx);
  }

  void genFIR(const Fortran::parser::WhereConstruct &c) {
    TODO(toLocation(), "WhereConstruct lowering");
  }

  void genFIR(const Fortran::parser::WhereBodyConstruct &body) {
    TODO(toLocation(), "WhereBodyConstruct lowering");
  }

  void genFIR(const Fortran::parser::WhereConstructStmt &stmt) {
    TODO(toLocation(), "WhereConstructStmt lowering");
  }

  void genFIR(const Fortran::parser::WhereConstruct::MaskedElsewhere &ew) {
    TODO(toLocation(), "MaskedElsewhere lowering");
  }

  void genFIR(const Fortran::parser::MaskedElsewhereStmt &stmt) {
    TODO(toLocation(), "MaskedElsewhereStmt lowering");
  }

  void genFIR(const Fortran::parser::WhereConstruct::Elsewhere &ew) {
    TODO(toLocation(), "Elsewhere lowering");
  }

  void genFIR(const Fortran::parser::ElsewhereStmt &stmt) {
    TODO(toLocation(), "ElsewhereStmt lowering");
  }

  void genFIR(const Fortran::parser::EndWhereStmt &) {
    TODO(toLocation(), "EndWhereStmt lowering");
  }

  void genFIR(const Fortran::parser::WhereStmt &stmt) {
    TODO(toLocation(), "WhereStmt lowering");
  }

  void genFIR(const Fortran::parser::PointerAssignmentStmt &stmt) {
    TODO(toLocation(), "PointerAssignmentStmt lowering");
  }

  void genFIR(const Fortran::parser::AssignmentStmt &stmt) {
    genAssignment(*stmt.typedAssignment->v);
  }

  void genFIR(const Fortran::parser::SyncAllStmt &stmt) {
    TODO(toLocation(), "SyncAllStmt lowering");
  }

  void genFIR(const Fortran::parser::SyncImagesStmt &stmt) {
    TODO(toLocation(), "SyncImagesStmt lowering");
  }

  void genFIR(const Fortran::parser::SyncMemoryStmt &stmt) {
    TODO(toLocation(), "SyncMemoryStmt lowering");
  }

  void genFIR(const Fortran::parser::SyncTeamStmt &stmt) {
    TODO(toLocation(), "SyncTeamStmt lowering");
  }

  void genFIR(const Fortran::parser::UnlockStmt &stmt) {
    TODO(toLocation(), "UnlockStmt lowering");
  }

  void genFIR(const Fortran::parser::AssignStmt &stmt) {
    TODO(toLocation(), "AssignStmt lowering");
  }

  void genFIR(const Fortran::parser::FormatStmt &) {
    TODO(toLocation(), "FormatStmt lowering");
  }

  void genFIR(const Fortran::parser::PauseStmt &stmt) {
    genPauseStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::FailImageStmt &stmt) {
    TODO(toLocation(), "FailImageStmt lowering");
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
      TODO(loc, "Alternate return statement");
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
    TODO(toLocation(), "CycleStmt lowering");
  }

  void genFIR(const Fortran::parser::ExitStmt &) {
    TODO(toLocation(), "ExitStmt lowering");
  }

  void genFIR(const Fortran::parser::GotoStmt &) {
    genFIRBranch(getEval().controlSuccessor->block);
  }

  void genFIR(const Fortran::parser::AssociateStmt &) {
    TODO(toLocation(), "AssociateStmt lowering");
  }

  void genFIR(const Fortran::parser::CaseStmt &) {
    TODO(toLocation(), "CaseStmt lowering");
  }

  void genFIR(const Fortran::parser::ContinueStmt &) {
    TODO(toLocation(), "ContinueStmt lowering");
  }

  void genFIR(const Fortran::parser::ElseIfStmt &) {
    TODO(toLocation(), "ElseIfStmt lowering");
  }

  void genFIR(const Fortran::parser::ElseStmt &) {
    TODO(toLocation(), "ElseStmt lowering");
  }

  void genFIR(const Fortran::parser::EndAssociateStmt &) {
    TODO(toLocation(), "EndAssociateStmt lowering");
  }

  void genFIR(const Fortran::parser::EndDoStmt &) {
    TODO(toLocation(), "EndDoStmt lowering");
  }

  void genFIR(const Fortran::parser::EndIfStmt &) {
    TODO(toLocation(), "EndIfStmt lowering");
  }

  void genFIR(const Fortran::parser::EndMpSubprogramStmt &) {
    TODO(toLocation(), "EndMpSubprogramStmt lowering");
  }

  void genFIR(const Fortran::parser::EndSelectStmt &) {
    TODO(toLocation(), "EndSelectStmt lowering");
  }

  // Nop statements - No code, or code is generated at the construct level.
  void genFIR(const Fortran::parser::EndFunctionStmt &) {}   // nop
  void genFIR(const Fortran::parser::EndSubroutineStmt &) {} // nop

  void genFIR(const Fortran::parser::EntryStmt &) {
    TODO(toLocation(), "EntryStmt lowering");
  }

  void genFIR(const Fortran::parser::IfStmt &) {
    TODO(toLocation(), "IfStmt lowering");
  }

  void genFIR(const Fortran::parser::IfThenStmt &) {
    TODO(toLocation(), "IfThenStmt lowering");
  }

  void genFIR(const Fortran::parser::NonLabelDoStmt &) {
    TODO(toLocation(), "NonLabelDoStmt lowering");
  }

  void genFIR(const Fortran::parser::OmpEndLoopDirective &) {
    TODO(toLocation(), "OmpEndLoopDirective lowering");
  }

  void genFIR(const Fortran::parser::NamelistStmt &) {
    TODO(toLocation(), "NamelistStmt lowering");
  }

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
  }

  //===--------------------------------------------------------------------===//

  Fortran::lower::LoweringBridge &bridge;
  Fortran::evaluate::FoldingContext foldingContext;
  fir::FirOpBuilder *builder = nullptr;
  Fortran::lower::pft::Evaluation *evalPtr = nullptr;
  Fortran::lower::SymMap localSymbols;
  Fortran::parser::CharBlock currentPosition;

  /// Tuple of host assoicated variables.
  mlir::Value hostAssocTuple;
  Fortran::lower::ImplicitIterSpace implicitIterSpace;
  Fortran::lower::ExplicitIterSpace explicitIterSpace;
};

} // namespace

Fortran::evaluate::FoldingContext
Fortran::lower::LoweringBridge::createFoldingContext() const {
  return {getDefaultKinds(), getIntrinsicTable()};
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
