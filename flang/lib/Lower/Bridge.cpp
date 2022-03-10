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
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/IO.h"
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
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Runtime/iostat.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-bridge"

using namespace mlir;

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
    //  - Declare all functions that have definitions so that definition
    //    signatures prevail over call site signatures.
    //  - Define module variables and OpenMP/OpenACC declarative construct so
    //    that they are available before lowering any function that may use
    //    them.
    for (Fortran::lower::pft::Program::Units &u : pft.getUnits()) {
      std::visit(Fortran::common::visitors{
                     [&](Fortran::lower::pft::FunctionLikeUnit &f) {
                       declareFunction(f);
                     },
                     [&](Fortran::lower::pft::ModuleLikeUnit &m) {
                       lowerModuleDeclScope(m);
                       for (Fortran::lower::pft::FunctionLikeUnit &f :
                            m.nestedFunctions)
                         declareFunction(f);
                     },
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

    // Primary translation pass.
    for (Fortran::lower::pft::Program::Units &u : pft.getUnits()) {
      std::visit(
          Fortran::common::visitors{
              [&](Fortran::lower::pft::FunctionLikeUnit &f) { lowerFunc(f); },
              [&](Fortran::lower::pft::ModuleLikeUnit &m) { lowerMod(m); },
              [&](Fortran::lower::pft::BlockDataUnit &b) {},
              [&](Fortran::lower::pft::CompilerDirectiveUnit &d) {},
          },
          u);
    }
  }

  /// Declare a function.
  void declareFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    setCurrentPosition(funit.getStartingSourceLoc());
    for (int entryIndex = 0, last = funit.entryPointList.size();
         entryIndex < last; ++entryIndex) {
      funit.setActiveEntry(entryIndex);
      // Calling CalleeInterface ctor will build a declaration mlir::FuncOp with
      // no other side effects.
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

  mlir::Value impliedDoBinding(llvm::StringRef name) override final {
    mlir::Value val = localSymbols.lookupImpliedDo(name);
    if (!val)
      fir::emitFatalError(toLocation(), "ac-do-variable has no binding");
    return val;
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
  fir::ExtendedValue genExprBox(const Fortran::lower::SomeExpr &expr,
                                Fortran::lower::StatementContext &context,
                                mlir::Location loc) override final {
    if (expr.Rank() > 0 && Fortran::evaluate::IsVariable(expr) &&
        !Fortran::evaluate::HasVectorSubscript(expr))
      return Fortran::lower::createSomeArrayBox(*this, expr, localSymbols,
                                                context);
    return fir::BoxValue(
        builder->createBox(loc, genExprAddr(expr, context, &loc)));
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
  mlir::Type
  genType(Fortran::common::TypeCategory tc, int kind,
          llvm::ArrayRef<std::int64_t> lenParameters) override final {
    return Fortran::lower::getFIRType(&getMLIRContext(), tc, kind,
                                      lenParameters);
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
          assert(funit.parentHasHostAssoc());
          funit.parentHostAssoc().internalProcedureBindings(*this,
                                                            localSymbols);
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
  void instantiateVar(const Fortran::lower::pft::Variable &var,
                      Fortran::lower::AggregateStoreMap &storeMap) {
    Fortran::lower::instantiateVariable(*this, var, localSymbols, storeMap);
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

    // If this is a host procedure with host associations, then create the tuple
    // of pointers for passing to the internal procedures.
    if (!funit.getHostAssoc().empty())
      funit.getHostAssoc().hostProcedureBindings(*this, localSymbols);

    /// TODO: should use same mechanism as equivalence?
    /// One blocking point is character entry returns that need special handling
    /// since they are not locally allocated but come as argument. CHARACTER(*)
    /// is not something that fit wells with equivalence lowering.
    for (const Fortran::lower::pft::Variable &altResult :
         deferredFuncResultList) {
      if (std::optional<Fortran::lower::CalleeInterface::PassedEntity>
              passedResult = callee.getPassedResult())
        addSymbol(altResult.getSymbol(), resultArg.getAddr());
      Fortran::lower::StatementContext stmtCtx;
      Fortran::lower::mapSymbolAttributes(*this, altResult, localSymbols,
                                          stmtCtx, primaryFuncResultStorage);
    }

    // Create most function blocks in advance.
    createEmptyGlobalBlocks(funit.evaluationList);

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
          // A structured construct that is a target starts a new block.
          Fortran::lower::pft::Evaluation &constructStmt =
              eval.getFirstNestedEvaluation();
          if (constructStmt.isNewBlock)
            constructStmt.block = builder->createBlock(region);
        }
      }
    }
  }

  /// Lower a procedure (nest).
  void lowerFunc(Fortran::lower::pft::FunctionLikeUnit &funit) {
    if (!funit.isMainProgram()) {
      const Fortran::semantics::Symbol &procSymbol =
          funit.getSubprogramSymbol();
      if (procSymbol.owner().IsSubmodule()) {
        TODO(toLocation(), "support submodules");
        return;
      }
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
    // FIXME: get rid of the bogus function context and instantiate the
    // globals directly into the module.
    MLIRContext *context = &getMLIRContext();
    setCurrentPosition(mod.getStartingSourceLoc());
    mlir::FuncOp func = fir::FirOpBuilder::createFunction(
        mlir::UnknownLoc::get(context), getModuleOp(),
        fir::NameUniquer::doGenerated("ModuleSham"),
        mlir::FunctionType::get(context, llvm::None, llvm::None));
    func.addEntryBlock();
    builder = new fir::FirOpBuilder(func, bridge.getKindMap());
    for (const Fortran::lower::pft::Variable &var :
         mod.getOrderedSymbolTable()) {
      // Only define the variables owned by this module.
      const Fortran::semantics::Scope *owningScope = var.getOwningScope();
      if (!owningScope || mod.getScope() == *owningScope)
        Fortran::lower::defineModuleVariable(*this, var);
    }
    for (auto &eval : mod.evaluationList)
      genFIR(eval);
    if (mlir::Region *region = func.getCallableRegion())
      region->dropAllReferences();
    func.erase();
    delete builder;
    builder = nullptr;
  }

  /// Lower functions contained in a module.
  void lowerMod(Fortran::lower::pft::ModuleLikeUnit &mod) {
    for (Fortran::lower::pft::FunctionLikeUnit &f : mod.nestedFunctions)
      lowerFunc(f);
  }

  mlir::Value hostAssocTupleValue() override final { return hostAssocTuple; }

  /// Record a binding for the ssa-value of the tuple for this function.
  void bindHostAssocTuple(mlir::Value val) override final {
    assert(!hostAssocTuple && val);
    hostAssocTuple = val;
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
    builder->create<cf::BranchOp>(toLocation(), targetBlock);
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
    builder->create<mlir::func::ReturnOp>(loc, resultVal);
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

  static bool
  isArraySectionWithoutVectorSubscript(const Fortran::lower::SomeExpr &expr) {
    return expr.Rank() > 0 && Fortran::evaluate::IsVariable(expr) &&
           !Fortran::evaluate::UnwrapWholeSymbolDataRef(expr) &&
           !Fortran::evaluate::HasVectorSubscript(expr);
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
                // Fortran 2018 10.2.1.3 p10 and p11
                fir::factory::CharacterExprHelper{*builder, loc}.createAssign(
                    lhs, rhs);
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

  void genFIR(const Fortran::parser::DoConstruct &doConstruct) {
    TODO(toLocation(), "DoConstruct lowering");
  }

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

  fir::ExtendedValue
  genAssociateSelector(const Fortran::lower::SomeExpr &selector,
                       Fortran::lower::StatementContext &stmtCtx) {
    return isArraySectionWithoutVectorSubscript(selector)
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
    const Fortran::semantics::Symbol &symbol =
        *std::get<Fortran::parser::Name>(stmt.t).symbol;
    mlir::Location loc = toLocation();
    mlir::Value labelValue = builder->createIntegerConstant(
        loc, genType(symbol), std::get<Fortran::parser::Label>(stmt.t));
    builder->create<fir::StoreOp>(loc, labelValue, getSymbolAddress(symbol));
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

  void genFIR(const Fortran::parser::CaseStmt &) {
    TODO(toLocation(), "CaseStmt lowering");
  }

  void genFIR(const Fortran::parser::ElseIfStmt &) {
    TODO(toLocation(), "ElseIfStmt lowering");
  }

  void genFIR(const Fortran::parser::ElseStmt &) {
    TODO(toLocation(), "ElseStmt lowering");
  }

  void genFIR(const Fortran::parser::EndDoStmt &) {
    TODO(toLocation(), "EndDoStmt lowering");
  }

  void genFIR(const Fortran::parser::EndMpSubprogramStmt &) {
    TODO(toLocation(), "EndMpSubprogramStmt lowering");
  }

  void genFIR(const Fortran::parser::EndSelectStmt &) {
    TODO(toLocation(), "EndSelectStmt lowering");
  }

  // Nop statements - No code, or code is generated at the construct level.
  void genFIR(const Fortran::parser::AssociateStmt &) {}     // nop
  void genFIR(const Fortran::parser::ContinueStmt &) {}      // nop
  void genFIR(const Fortran::parser::EndAssociateStmt &) {}  // nop
  void genFIR(const Fortran::parser::EndFunctionStmt &) {}   // nop
  void genFIR(const Fortran::parser::EndIfStmt &) {}         // nop
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
