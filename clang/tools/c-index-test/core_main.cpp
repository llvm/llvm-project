//===-- core_main.cpp - Core Index Tool testbed ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Index/CodegenNameGenerator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/PrettyStackTrace.h"

using namespace clang;
using namespace clang::index;
using namespace llvm;

extern "C" int indextest_core_main(int argc, const char **argv);

namespace {

enum class ActionType {
  None,
  PrintSourceSymbols,
};

namespace options {

static cl::OptionCategory IndexTestCoreCategory("index-test-core options");

static cl::opt<ActionType>
Action(cl::desc("Action:"), cl::init(ActionType::None),
       cl::values(
          clEnumValN(ActionType::PrintSourceSymbols,
                     "print-source-symbols", "Print symbols from source"),
          clEnumValEnd),
       cl::cat(IndexTestCoreCategory));

static cl::extrahelp MoreHelp(
  "\nAdd \"-- <compiler arguments>\" at the end to setup the compiler "
  "invocation\n"
);

}
} // anonymous namespace

static void printSymbolInfo(SymbolInfo SymInfo, raw_ostream &OS);
static void printSymbolNameAndUSR(const Decl *D, ASTContext &Ctx,
                                  raw_ostream &OS);

namespace {

class PrintIndexDataConsumer : public IndexDataConsumer {
  raw_ostream &OS;
  std::unique_ptr<CodegenNameGenerator> CGNameGen;

public:
  PrintIndexDataConsumer(raw_ostream &OS) : OS(OS) {
  }

  void initialize(ASTContext &Ctx) override {
    CGNameGen.reset(new CodegenNameGenerator(Ctx));
  }

  bool handleDeclOccurence(const Decl *D, SymbolRoleSet Roles,
                           ArrayRef<SymbolRelation> Relations,
                           FileID FID, unsigned Offset,
                           ASTNodeInfo ASTNode) override {
    ASTContext &Ctx = D->getASTContext();
    SourceManager &SM = Ctx.getSourceManager();

    unsigned Line = SM.getLineNumber(FID, Offset);
    unsigned Col = SM.getColumnNumber(FID, Offset);
    OS << Line << ':' << Col << " | ";

    printSymbolInfo(getSymbolInfo(D), OS);
    OS << " | ";

    printSymbolNameAndUSR(D, Ctx, OS);
    OS << " | ";

    if (CGNameGen->writeName(D, OS))
      OS << "<no-cgname>";
    OS << " | ";

    printSymbolRoles(Roles, OS);
    OS << " | ";

    OS << "rel: " << Relations.size() << '\n';

    for (auto &SymRel : Relations) {
      OS << '\t';
      printSymbolRoles(SymRel.Roles, OS);
      OS << " | ";
      printSymbolNameAndUSR(SymRel.RelatedSymbol, Ctx, OS);
      OS << '\n';
    }

    return true;
  }

  bool handleModuleOccurence(const ImportDecl *ImportD, SymbolRoleSet Roles,
                             FileID FID, unsigned Offset) override {
    ASTContext &Ctx = ImportD->getASTContext();
    SourceManager &SM = Ctx.getSourceManager();

    unsigned Line = SM.getLineNumber(FID, Offset);
    unsigned Col = SM.getColumnNumber(FID, Offset);
    OS << Line << ':' << Col << " | ";

    printSymbolInfo(getSymbolInfo(ImportD), OS);
    OS << " | ";

    OS << ImportD->getImportedModule()->getFullModuleName() << " | ";

    printSymbolRoles(Roles, OS);
    OS << " |\n";

    return true;
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Print Source Symbols
//===----------------------------------------------------------------------===//

static bool printSourceSymbols(ArrayRef<const char *> Args) {
  SmallVector<const char *, 4> ArgsWithProgName;
  ArgsWithProgName.push_back("clang");
  ArgsWithProgName.append(Args.begin(), Args.end());
  IntrusiveRefCntPtr<DiagnosticsEngine>
    Diags(CompilerInstance::createDiagnostics(new DiagnosticOptions));
  IntrusiveRefCntPtr<CompilerInvocation>
    CInvok(createInvocationFromCommandLine(ArgsWithProgName, Diags));
  if (!CInvok)
    return true;

  auto DataConsumer = std::make_shared<PrintIndexDataConsumer>(outs());
  IndexingOptions IndexOpts;
  std::unique_ptr<FrontendAction> IndexAction;
  IndexAction = createIndexingAction(DataConsumer, IndexOpts,
                                     /*WrappedAction=*/nullptr);

  auto PCHContainerOps = std::make_shared<PCHContainerOperations>();
  std::unique_ptr<ASTUnit> Unit(ASTUnit::LoadFromCompilerInvocationAction(
      CInvok.get(), PCHContainerOps, Diags, IndexAction.get()));

  if (!Unit)
    return true;

  return false;
}

//===----------------------------------------------------------------------===//
// Helper Utils
//===----------------------------------------------------------------------===//

static void printSymbolInfo(SymbolInfo SymInfo, raw_ostream &OS) {
  OS << getSymbolKindString(SymInfo.Kind);
  if (SymInfo.SubKinds) {
    OS << '(';
    printSymbolSubKinds(SymInfo.SubKinds, OS);
    OS << ')';
  }
  OS << '/' << getSymbolLanguageString(SymInfo.Lang);
}

static void printSymbolNameAndUSR(const Decl *D, ASTContext &Ctx,
                                  raw_ostream &OS) {
  if (printSymbolName(D, Ctx.getLangOpts(), OS)) {
    OS << "<no-name>";
  }
  OS << " | ";

  SmallString<256> USRBuf;
  if (generateUSRForDecl(D, USRBuf)) {
    OS << "<no-usr>";
  } else {
    OS << USRBuf;
  }
}

//===----------------------------------------------------------------------===//
// Command line processing.
//===----------------------------------------------------------------------===//

int indextest_core_main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);

  assert(argv[1] == StringRef("core"));
  ++argv;
  --argc;

  std::vector<const char *> CompArgs;
  const char **DoubleDash = std::find(argv, argv + argc, StringRef("--"));
  if (DoubleDash != argv + argc) {
    CompArgs = std::vector<const char *>(DoubleDash + 1, argv + argc);
    argc = DoubleDash - argv;
  }

  cl::HideUnrelatedOptions(options::IndexTestCoreCategory);
  cl::ParseCommandLineOptions(argc, argv, "index-test-core");

  if (options::Action == ActionType::None) {
    errs() << "error: action required; pass '-help' for options\n";
    return 1;
  }

  if (options::Action == ActionType::PrintSourceSymbols) {
    if (CompArgs.empty()) {
      errs() << "error: missing compiler args; pass '-- <compiler arguments>'\n";
      return 1;
    }
    return printSourceSymbols(CompArgs);
  }

  return 0;
}
