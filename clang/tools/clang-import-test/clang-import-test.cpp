//===-- import-test.cpp - ASTImporter/ExternalASTSource testbed -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTImporter.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExternalASTMerger.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Signals.h"

#include <memory>
#include <string>

using namespace clang;

static llvm::cl::opt<std::string> Expression(
    "expression", llvm::cl::Required,
    llvm::cl::desc("Path to a file containing the expression to parse"));

static llvm::cl::list<std::string>
    Imports("import", llvm::cl::ZeroOrMore,
            llvm::cl::desc("Path to a file containing declarations to import"));

static llvm::cl::opt<bool>
    Direct("direct", llvm::cl::Optional,
             llvm::cl::desc("Use the parsed declarations without indirection"));

static llvm::cl::list<std::string>
    ClangArgs("Xcc", llvm::cl::ZeroOrMore,
              llvm::cl::desc("Argument to pass to the CompilerInvocation"),
              llvm::cl::CommaSeparated);

static llvm::cl::opt<bool>
DumpAST("dump-ast", llvm::cl::init(false),
        llvm::cl::desc("Dump combined AST"));

namespace init_convenience {
class TestDiagnosticConsumer : public DiagnosticConsumer {
private:
  std::unique_ptr<TextDiagnosticBuffer> Passthrough;
  const LangOptions *LangOpts = nullptr;

public:
  TestDiagnosticConsumer()
      : Passthrough(llvm::make_unique<TextDiagnosticBuffer>()) {}

  virtual void BeginSourceFile(const LangOptions &LangOpts,
                               const Preprocessor *PP = nullptr) override {
    this->LangOpts = &LangOpts;
    return Passthrough->BeginSourceFile(LangOpts, PP);
  }

  virtual void EndSourceFile() override {
    this->LangOpts = nullptr;
    Passthrough->EndSourceFile();
  }

  virtual bool IncludeInDiagnosticCounts() const override {
    return Passthrough->IncludeInDiagnosticCounts();
  }

private:
  static void PrintSourceForLocation(const SourceLocation &Loc,
                                     SourceManager &SM) {
    const char *LocData = SM.getCharacterData(Loc, /*Invalid=*/nullptr);
    unsigned LocColumn =
        SM.getSpellingColumnNumber(Loc, /*Invalid=*/nullptr) - 1;
    FileID FID = SM.getFileID(Loc);
    llvm::MemoryBuffer *Buffer = SM.getBuffer(FID, Loc, /*Invalid=*/nullptr);

    assert(LocData >= Buffer->getBufferStart() &&
           LocData < Buffer->getBufferEnd());

    const char *LineBegin = LocData - LocColumn;

    assert(LineBegin >= Buffer->getBufferStart());

    const char *LineEnd = nullptr;

    for (LineEnd = LineBegin; *LineEnd != '\n' && *LineEnd != '\r' &&
                              LineEnd < Buffer->getBufferEnd();
         ++LineEnd)
      ;

    llvm::StringRef LineString(LineBegin, LineEnd - LineBegin);

    llvm::errs() << LineString << '\n';
    llvm::errs().indent(LocColumn);
    llvm::errs() << '^';
  }

  virtual void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                                const Diagnostic &Info) override {
    if (Info.hasSourceManager() && LangOpts) {
      SourceManager &SM = Info.getSourceManager();

      if (Info.getLocation().isValid()) {
        Info.getLocation().print(llvm::errs(), SM);
        llvm::errs() << ": ";
      }

      SmallString<16> DiagText;
      Info.FormatDiagnostic(DiagText);
      llvm::errs() << DiagText << '\n';

      if (Info.getLocation().isValid()) {
        PrintSourceForLocation(Info.getLocation(), SM);
      }

      for (const CharSourceRange &Range : Info.getRanges()) {
        bool Invalid = true;
        StringRef Ref = Lexer::getSourceText(Range, SM, *LangOpts, &Invalid);
        if (!Invalid) {
          llvm::errs() << Ref << '\n';
        }
      }
    }
    DiagnosticConsumer::HandleDiagnostic(DiagLevel, Info);
  }
};

std::unique_ptr<CompilerInstance>
BuildCompilerInstance(ArrayRef<const char *> ClangArgv) {
  auto Ins = llvm::make_unique<CompilerInstance>();
  auto DC = llvm::make_unique<TestDiagnosticConsumer>();
  const bool ShouldOwnClient = true;
  Ins->createDiagnostics(DC.release(), ShouldOwnClient);

  auto Inv = llvm::make_unique<CompilerInvocation>();

  CompilerInvocation::CreateFromArgs(*Inv, ClangArgv.data(),
                                     &ClangArgv.data()[ClangArgv.size()],
                                     Ins->getDiagnostics());

  Inv->getLangOpts()->CPlusPlus = true;
  Inv->getLangOpts()->CPlusPlus11 = true;
  Inv->getHeaderSearchOpts().UseLibcxx = true;
  Inv->getLangOpts()->Bool = true;
  Inv->getLangOpts()->WChar = true;
  Inv->getLangOpts()->Blocks = true;
  Inv->getLangOpts()->DebuggerSupport = true;
  Inv->getLangOpts()->SpellChecking = false;
  Inv->getLangOpts()->ThreadsafeStatics = false;
  Inv->getLangOpts()->AccessControl = false;
  Inv->getLangOpts()->DollarIdents = true;
  Inv->getCodeGenOpts().setDebugInfo(codegenoptions::FullDebugInfo);
  Inv->getTargetOpts().Triple = llvm::sys::getDefaultTargetTriple();

  Ins->setInvocation(std::move(Inv));

  TargetInfo *TI = TargetInfo::CreateTargetInfo(
      Ins->getDiagnostics(), Ins->getInvocation().TargetOpts);
  Ins->setTarget(TI);
  Ins->getTarget().adjust(Ins->getLangOpts());
  Ins->createFileManager();
  Ins->createSourceManager(Ins->getFileManager());
  Ins->createPreprocessor(TU_Complete);

  return Ins;
}

std::unique_ptr<CompilerInstance>
BuildCompilerInstance(ArrayRef<std::string> ClangArgs) {
  std::vector<const char *> ClangArgv(ClangArgs.size());
  std::transform(ClangArgs.begin(), ClangArgs.end(), ClangArgv.begin(),
                 [](const std::string &s) -> const char * { return s.data(); });
  return init_convenience::BuildCompilerInstance(ClangArgv);
}

std::unique_ptr<ASTContext>
BuildASTContext(CompilerInstance &CI, SelectorTable &ST, Builtin::Context &BC) {
  auto AST = llvm::make_unique<ASTContext>(
      CI.getLangOpts(), CI.getSourceManager(),
      CI.getPreprocessor().getIdentifierTable(), ST, BC);
  AST->InitBuiltinTypes(CI.getTarget());
  return AST;
}

std::unique_ptr<CodeGenerator> BuildCodeGen(CompilerInstance &CI,
                                            llvm::LLVMContext &LLVMCtx) {
  StringRef ModuleName("$__module");
  return std::unique_ptr<CodeGenerator>(CreateLLVMCodeGen(
      CI.getDiagnostics(), ModuleName, CI.getHeaderSearchOpts(),
      CI.getPreprocessorOpts(), CI.getCodeGenOpts(), LLVMCtx));
}
} // end namespace

namespace {
 
void AddExternalSource(
    CompilerInstance &CI,
    llvm::ArrayRef<std::unique_ptr<CompilerInstance>> Imports) {
  ExternalASTMerger::ImporterEndpoint Target({CI.getASTContext(), CI.getFileManager()});
  llvm::SmallVector<ExternalASTMerger::ImporterEndpoint, 3> Sources;
  for (const std::unique_ptr<CompilerInstance> &CI : Imports) {
    Sources.push_back({CI->getASTContext(), CI->getFileManager()});
  }
  auto ES = llvm::make_unique<ExternalASTMerger>(Target, Sources);
  CI.getASTContext().setExternalSource(ES.release());
  CI.getASTContext().getTranslationUnitDecl()->setHasExternalVisibleStorage();
}

std::unique_ptr<CompilerInstance> BuildIndirect(std::unique_ptr<CompilerInstance> &CI) {
  std::vector<const char *> ClangArgv(ClangArgs.size());
  std::transform(ClangArgs.begin(), ClangArgs.end(), ClangArgv.begin(),
                 [](const std::string &s) -> const char * { return s.data(); });
  std::unique_ptr<CompilerInstance> IndirectCI =
      init_convenience::BuildCompilerInstance(ClangArgv);
  auto ST = llvm::make_unique<SelectorTable>();
  auto BC = llvm::make_unique<Builtin::Context>();
  std::unique_ptr<ASTContext> AST =
      init_convenience::BuildASTContext(*IndirectCI, *ST, *BC);
  IndirectCI->setASTContext(AST.release());
  AddExternalSource(*IndirectCI, CI);
  return IndirectCI;
}

llvm::Error ParseSource(const std::string &Path, CompilerInstance &CI,
                        ASTConsumer &Consumer) {
  SourceManager &SM = CI.getSourceManager();
  const FileEntry *FE = CI.getFileManager().getFile(Path);
  if (!FE) {
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("Couldn't open ", Path), std::error_code());
  }
  SM.setMainFileID(SM.createFileID(FE, SourceLocation(), SrcMgr::C_User));
  ParseAST(CI.getPreprocessor(), &Consumer, CI.getASTContext());
  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<CompilerInstance>>
Parse(const std::string &Path,
      llvm::ArrayRef<std::unique_ptr<CompilerInstance>> Imports,
      bool ShouldDumpAST) {
  std::vector<const char *> ClangArgv(ClangArgs.size());
  std::transform(ClangArgs.begin(), ClangArgs.end(), ClangArgv.begin(),
                 [](const std::string &s) -> const char * { return s.data(); });
  std::unique_ptr<CompilerInstance> CI =
      init_convenience::BuildCompilerInstance(ClangArgv);
  auto ST = llvm::make_unique<SelectorTable>();
  auto BC = llvm::make_unique<Builtin::Context>();
  std::unique_ptr<ASTContext> AST =
      init_convenience::BuildASTContext(*CI, *ST, *BC);
  CI->setASTContext(AST.release());
  if (Imports.size())
    AddExternalSource(*CI, Imports);

  std::vector<std::unique_ptr<ASTConsumer>> ASTConsumers;

  auto LLVMCtx = llvm::make_unique<llvm::LLVMContext>();
  ASTConsumers.push_back(init_convenience::BuildCodeGen(*CI, *LLVMCtx));

  if (ShouldDumpAST)
    ASTConsumers.push_back(CreateASTDumper("", true, false, false));

  CI->getDiagnosticClient().BeginSourceFile(CI->getLangOpts(),
                                            &CI->getPreprocessor());
  MultiplexConsumer Consumers(std::move(ASTConsumers));
  Consumers.Initialize(CI->getASTContext());

  if (llvm::Error PE = ParseSource(Path, *CI, Consumers)) {
    return std::move(PE);
  }
  CI->getDiagnosticClient().EndSourceFile();
  if (CI->getDiagnosticClient().getNumErrors()) {
    return llvm::make_error<llvm::StringError>(
        "Errors occured while parsing the expression.", std::error_code());
  } else {
    return std::move(CI);
  }
}

} // end namespace

int main(int argc, const char **argv) {
  const bool DisableCrashReporting = true;
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0], DisableCrashReporting);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  std::vector<std::unique_ptr<CompilerInstance>> ImportCIs;
  for (auto I : Imports) {
    llvm::Expected<std::unique_ptr<CompilerInstance>> ImportCI =
      Parse(I, {}, false);
    if (auto E = ImportCI.takeError()) {
      llvm::errs() << llvm::toString(std::move(E));
      exit(-1);
    } else {
      ImportCIs.push_back(std::move(*ImportCI));
    }
  }
  std::vector<std::unique_ptr<CompilerInstance>> IndirectCIs;
  if (!Direct) {
    for (auto &ImportCI : ImportCIs) {
      llvm::Expected<std::unique_ptr<CompilerInstance>> IndirectCI =
          BuildIndirect(ImportCI);
      if (auto E = IndirectCI.takeError()) {
        llvm::errs() << llvm::toString(std::move(E));
        exit(-1);
      } else {
        IndirectCIs.push_back(std::move(*IndirectCI));
      }
    }
  }
  llvm::Expected<std::unique_ptr<CompilerInstance>> ExpressionCI =
      Parse(Expression, Direct ? ImportCIs : IndirectCIs, DumpAST);
  if (auto E = ExpressionCI.takeError()) {
    llvm::errs() << llvm::toString(std::move(E));
    exit(-1);
  } else {
    return 0;
  }
}

