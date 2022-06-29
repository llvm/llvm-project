#include "CIRASTConsumer.h"

#include "clang/CIR/Dialect/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "../utils/OptionsUtils.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <algorithm>

using namespace clang;
using namespace clang::tidy;

namespace {
const std::string lifeTimeCheck = "cir-lifetime-check";
} // namespace

namespace cir {
namespace tidy {

CIRASTConsumer::CIRASTConsumer(CompilerInstance &CI, StringRef inputFile,
                               clang::tidy::ClangTidyContext &Context)
    : Context(Context) {
  Gen =
      std::make_unique<CIRGenerator>(CI.getDiagnostics(), CI.getCodeGenOpts());
}

bool CIRASTConsumer::HandleTopLevelDecl(DeclGroupRef D) {
  PrettyStackTraceDecl CrashInfo(*D.begin(), SourceLocation(),
                                 AstContext->getSourceManager(),
                                 "CIR generation of declaration");
  Gen->HandleTopLevelDecl(D);
  return true;
}

void CIRASTConsumer::Initialize(ASTContext &Context) {
  AstContext = &Context;
  Gen->Initialize(Context);
}

void CIRASTConsumer::HandleTranslationUnit(ASTContext &C) {
  Gen->HandleTranslationUnit(C);
  Gen->verifyModule();

  mlir::ModuleOp mlirMod = Gen->getModule();
  std::unique_ptr<mlir::MLIRContext> mlirCtx = Gen->takeContext();

  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(/*prettyForm=*/false);

  SourceManager &SourceMgr = C.getSourceManager();
  FileID MainFileID = SourceMgr.getMainFileID();

  llvm::MemoryBufferRef MainFileBuf = SourceMgr.getBufferOrFake(MainFileID);
  std::unique_ptr<llvm::MemoryBuffer> FileBuf =
      llvm::MemoryBuffer::getMemBuffer(MainFileBuf);

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(FileBuf), llvm::SMLoc());

  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, mlirCtx.get());
  mlir::PassManager pm(mlirCtx.get());
  pm.addPass(mlir::createMergeCleanupsPass());

  clang::tidy::ClangTidyOptions Opts = Context.getOptions();
  static constexpr const char *remarkOptName = "cir-lifetime-check.RemarksList";
  static constexpr const char *histOptName = "cir-lifetime-check.HistoryList";
  auto remarks =
      utils::options::parseStringList(Opts.CheckOptions[remarkOptName].Value);
  auto hist =
      utils::options::parseStringList(Opts.CheckOptions[histOptName].Value);

  if (Context.isCheckEnabled(lifeTimeCheck))
    pm.addPass(mlir::createLifetimeCheckPass(remarks, hist, 1, &C));

  bool Result = !mlir::failed(pm.run(mlirMod));
  if (!Result)
    llvm::report_fatal_error(
        "The pass manager failed to run pass on the module!");
}
} // namespace tidy
} // namespace cir
