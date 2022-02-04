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
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Support/FIRContext.h"
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
                                 mlir::Location *loc = nullptr) override final {
    TODO_NOLOC("Not implemented genExprAddr. Needed for more complex "
               "expression lowering");
  }
  fir::ExtendedValue
  genExprValue(const Fortran::lower::SomeExpr &expr,
               mlir::Location *loc = nullptr) override final {
    return createSomeExtendedExpression(loc ? *loc : toLocation(), *this, expr,
                                        localSymbols);
  }

  Fortran::evaluate::FoldingContext &getFoldingContext() override final {
    return foldingContext;
  }

  mlir::Type genType(const Fortran::evaluate::DataRef &) override final {
    TODO_NOLOC("Not implemented genType DataRef. Needed for more complex "
               "expression lowering");
  }
  mlir::Type genType(const Fortran::lower::SomeExpr &) override final {
    TODO_NOLOC("Not implemented genType SomeExpr. Needed for more complex "
               "expression lowering");
  }
  mlir::Type genType(Fortran::lower::SymbolRef) override final {
    TODO_NOLOC("Not implemented genType SymbolRef. Needed for more complex "
               "expression lowering");
  }
  mlir::Type genType(Fortran::common::TypeCategory tc) override final {
    TODO_NOLOC("Not implemented genType TypeCategory. Needed for more complex "
               "expression lowering");
  }
  mlir::Type genType(Fortran::common::TypeCategory tc,
                     int kind) override final {
    return Fortran::lower::getFIRType(&getMLIRContext(), tc, kind);
  }
  mlir::Type genType(const Fortran::lower::pft::Variable &) override final {
    TODO_NOLOC("Not implemented genType Variable. Needed for more complex "
               "expression lowering");
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
    delete builder;
    builder = nullptr;
    localSymbols.clear();
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

  void genFIRProcedureExit(Fortran::lower::pft::FunctionLikeUnit &funit,
                           const Fortran::semantics::Symbol &symbol) {
    if (Fortran::semantics::IsFunction(symbol)) {
      TODO(toLocation(), "Function lowering");
    } else {
      genExitRoutine();
    }
  }

  void genFIR(const Fortran::parser::CallStmt &stmt) {
    TODO(toLocation(), "CallStmt lowering");
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
    TODO(toLocation(), "AssignmentStmt lowering");
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
    TODO(toLocation(), "ReturnStmt lowering");
  }

  void genFIR(const Fortran::parser::CycleStmt &) {
    TODO(toLocation(), "CycleStmt lowering");
  }

  void genFIR(const Fortran::parser::ExitStmt &) {
    TODO(toLocation(), "ExitStmt lowering");
  }

  void genFIR(const Fortran::parser::GotoStmt &) {
    TODO(toLocation(), "GotoStmt lowering");
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

  void genFIR(const Fortran::parser::EndFunctionStmt &) {
    TODO(toLocation(), "EndFunctionStmt lowering");
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
