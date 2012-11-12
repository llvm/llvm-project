//===-- arcmt-test.cpp - ARC Migration Tool testbed -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/ARCMigrate/ARCMT.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/VerifyDiagnosticConsumer.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/system_error.h"

using namespace clang;
using namespace arcmt;

static llvm::cl::opt<bool>
CheckOnly("check-only",
      llvm::cl::desc("Just check for issues that need to be handled manually"));

//static llvm::cl::opt<bool>
//TestResultForARC("test-result",
//llvm::cl::desc("Test the result of transformations by parsing it in ARC mode"));

static llvm::cl::opt<bool>
OutputTransformations("output-transformations",
                      llvm::cl::desc("Print the source transformations"));

static llvm::cl::opt<bool>
VerifyDiags("verify",llvm::cl::desc("Verify emitted diagnostics and warnings"));

static llvm::cl::opt<bool>
VerboseOpt("v", llvm::cl::desc("Enable verbose output"));

static llvm::cl::opt<bool>
VerifyTransformedFiles("verify-transformed-files",
llvm::cl::desc("Read pairs of file mappings (typically the output of "
               "c-arcmt-test) and compare their contents with the filenames "
               "provided in command-line"));

static llvm::cl::opt<std::string>
RemappingsFile("remappings-file",
               llvm::cl::desc("Pairs of file mappings (typically the output of "
               "c-arcmt-test)"));

static llvm::cl::list<std::string>
ResultFiles(llvm::cl::Positional, llvm::cl::desc("<filename>..."));

static llvm::cl::extrahelp extraHelp(
  "\nusage with compiler args: arcmt-test [options] --args [compiler flags]\n");

// This function isn't referenced outside its translation unit, but it
// can't use the "static" keyword because its address is used for
// GetMainExecutable (since some platforms don't support taking the
// address of main, and some platforms can't implement GetMainExecutable
// without being given the address of a function in the main executable).
llvm::sys::Path GetExecutablePath(const char *Argv0) {
  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *MainAddr = (void*) (intptr_t) GetExecutablePath;
  return llvm::sys::Path::GetMainExecutable(Argv0, MainAddr);
}

static void printSourceLocation(SourceLocation loc, ASTContext &Ctx,
                                raw_ostream &OS);
static void printSourceRange(CharSourceRange range, ASTContext &Ctx,
                             raw_ostream &OS);

namespace {

class PrintTransforms : public MigrationProcess::RewriteListener {
  ASTContext *Ctx;
  raw_ostream &OS;

public:
  PrintTransforms(raw_ostream &OS)
    : Ctx(0), OS(OS) { }

  virtual void start(ASTContext &ctx) { Ctx = &ctx; }
  virtual void finish() { Ctx = 0; }

  virtual void insert(SourceLocation loc, StringRef text) {
    assert(Ctx);
    OS << "Insert: ";
    printSourceLocation(loc, *Ctx, OS);
    OS << " \"" << text << "\"\n";
  }

  virtual void remove(CharSourceRange range) {
    assert(Ctx);
    OS << "Remove: ";
    printSourceRange(range, *Ctx, OS);
    OS << '\n';
  }
};

} // anonymous namespace

static bool checkForMigration(StringRef resourcesPath,
                              ArrayRef<const char *> Args) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  DiagnosticConsumer *DiagClient =
    new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags(
      new DiagnosticsEngine(DiagID, &*DiagOpts, DiagClient));
  // Chain in -verify checker, if requested.
  VerifyDiagnosticConsumer *verifyDiag = 0;
  if (VerifyDiags) {
    verifyDiag = new VerifyDiagnosticConsumer(*Diags);
    Diags->setClient(verifyDiag);
  }

  CompilerInvocation CI;
  if (!CompilerInvocation::CreateFromArgs(CI, Args.begin(), Args.end(), *Diags))
    return true;

  if (CI.getFrontendOpts().Inputs.empty()) {
    llvm::errs() << "error: no input files\n";
    return true;
  }

  if (!CI.getLangOpts()->ObjC1)
    return false;

  arcmt::checkForManualIssues(CI, CI.getFrontendOpts().Inputs[0], 
                              Diags->getClient());
  return Diags->getClient()->getNumErrors() > 0;
}

static void printResult(FileRemapper &remapper, raw_ostream &OS) {
  PreprocessorOptions PPOpts;
  remapper.applyMappings(PPOpts);
  // The changed files will be in memory buffers, print them.
  for (unsigned i = 0, e = PPOpts.RemappedFileBuffers.size(); i != e; ++i) {
    const llvm::MemoryBuffer *mem = PPOpts.RemappedFileBuffers[i].second;
    OS << mem->getBuffer();
  }
}

static bool performTransformations(StringRef resourcesPath,
                                   ArrayRef<const char *> Args) {
  // Check first.
  if (checkForMigration(resourcesPath, Args))
    return true;

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  DiagnosticConsumer *DiagClient =
    new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  IntrusiveRefCntPtr<DiagnosticsEngine> TopDiags(
      new DiagnosticsEngine(DiagID, &*DiagOpts, &*DiagClient));

  CompilerInvocation origCI;
  if (!CompilerInvocation::CreateFromArgs(origCI, Args.begin(), Args.end(),
                                     *TopDiags))
    return true;

  if (origCI.getFrontendOpts().Inputs.empty()) {
    llvm::errs() << "error: no input files\n";
    return true;
  }

  if (!origCI.getLangOpts()->ObjC1)
    return false;

  MigrationProcess migration(origCI, DiagClient);

  std::vector<TransformFn>
    transforms = arcmt::getAllTransformations(origCI.getLangOpts()->getGC(),
                                 origCI.getMigratorOpts().NoFinalizeRemoval);
  assert(!transforms.empty());

  OwningPtr<PrintTransforms> transformPrinter;
  if (OutputTransformations)
    transformPrinter.reset(new PrintTransforms(llvm::outs()));

  for (unsigned i=0, e = transforms.size(); i != e; ++i) {
    bool err = migration.applyTransform(transforms[i], transformPrinter.get());
    if (err) return true;

    if (VerboseOpt) {
      if (i == e-1)
        llvm::errs() << "\n##### FINAL RESULT #####\n";
      else
        llvm::errs() << "\n##### OUTPUT AFTER "<< i+1 <<". TRANSFORMATION #####\n";
      printResult(migration.getRemapper(), llvm::errs());
      llvm::errs() << "\n##########################\n\n";
    }
  }

  if (!OutputTransformations)
    printResult(migration.getRemapper(), llvm::outs());

  // FIXME: TestResultForARC

  return false;
}

static bool filesCompareEqual(StringRef fname1, StringRef fname2) {
  using namespace llvm;

  OwningPtr<MemoryBuffer> file1;
  MemoryBuffer::getFile(fname1, file1);
  if (!file1)
    return false;
  
  OwningPtr<MemoryBuffer> file2;
  MemoryBuffer::getFile(fname2, file2);
  if (!file2)
    return false;

  return file1->getBuffer() == file2->getBuffer();
}

static bool verifyTransformedFiles(ArrayRef<std::string> resultFiles) {
  using namespace llvm;

  assert(!resultFiles.empty());

  std::map<StringRef, StringRef> resultMap;

  for (ArrayRef<std::string>::iterator
         I = resultFiles.begin(), E = resultFiles.end(); I != E; ++I) {
    StringRef fname(*I);
    if (!fname.endswith(".result")) {
      errs() << "error: filename '" << fname
                   << "' does not have '.result' extension\n";
      return true;
    }
    resultMap[sys::path::stem(fname)] = fname;
  }

  OwningPtr<MemoryBuffer> inputBuf;
  if (RemappingsFile.empty())
    MemoryBuffer::getSTDIN(inputBuf);
  else
    MemoryBuffer::getFile(RemappingsFile, inputBuf);
  if (!inputBuf) {
    errs() << "error: could not read remappings input\n";
    return true;
  }

  SmallVector<StringRef, 8> strs;
  inputBuf->getBuffer().split(strs, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  if (strs.empty()) {
    errs() << "error: no files to verify from stdin\n";
    return true;
  }
  if (strs.size() % 2 != 0) {
    errs() << "error: files to verify are not original/result pairs\n";
    return true;
  }

  for (unsigned i = 0, e = strs.size(); i != e; i += 2) {
    StringRef inputOrigFname = strs[i];
    StringRef inputResultFname = strs[i+1];

    std::map<StringRef, StringRef>::iterator It;
    It = resultMap.find(sys::path::filename(inputOrigFname));
    if (It == resultMap.end()) {
      errs() << "error: '" << inputOrigFname << "' is not in the list of "
             << "transformed files to verify\n";
      return true;
    }

    bool exists = false;
    sys::fs::exists(It->second, exists);
    if (!exists) {
      errs() << "error: '" << It->second << "' does not exist\n";
      return true;
    }
    sys::fs::exists(inputResultFname, exists);
    if (!exists) {
      errs() << "error: '" << inputResultFname << "' does not exist\n";
      return true;
    }

    if (!filesCompareEqual(It->second, inputResultFname)) {
      errs() << "error: '" << It->second << "' is different than "
             << "'" << inputResultFname << "'\n";
      return true;
    }

    resultMap.erase(It);
  }

  if (!resultMap.empty()) {
    for (std::map<StringRef, StringRef>::iterator
           I = resultMap.begin(), E = resultMap.end(); I != E; ++I)
      errs() << "error: '" << I->second << "' was not verified!\n";
    return true;
  }

  return false; 
}

//===----------------------------------------------------------------------===//
// Misc. functions.
//===----------------------------------------------------------------------===//

static void printSourceLocation(SourceLocation loc, ASTContext &Ctx,
                                raw_ostream &OS) {
  SourceManager &SM = Ctx.getSourceManager();
  PresumedLoc PL = SM.getPresumedLoc(loc);

  OS << llvm::sys::path::filename(PL.getFilename());
  OS << ":" << PL.getLine() << ":"
            << PL.getColumn();
}

static void printSourceRange(CharSourceRange range, ASTContext &Ctx,
                             raw_ostream &OS) {
  SourceManager &SM = Ctx.getSourceManager();
  const LangOptions &langOpts = Ctx.getLangOpts();

  PresumedLoc PL = SM.getPresumedLoc(range.getBegin());

  OS << llvm::sys::path::filename(PL.getFilename());
  OS << " [" << PL.getLine() << ":"
             << PL.getColumn();
  OS << " - ";

  SourceLocation end = range.getEnd();
  PL = SM.getPresumedLoc(end);

  unsigned endCol = PL.getColumn() - 1;
  if (!range.isTokenRange())
    endCol += Lexer::MeasureTokenLength(end, SM, langOpts);
  OS << PL.getLine() << ":" << endCol << "]";
}

//===----------------------------------------------------------------------===//
// Command line processing.
//===----------------------------------------------------------------------===//

int main(int argc, const char **argv) {
  void *MainAddr = (void*) (intptr_t) GetExecutablePath;
  llvm::sys::PrintStackTraceOnErrorSignal();

  std::string
    resourcesPath = CompilerInvocation::GetResourcesPath(argv[0], MainAddr);

  int optargc = 0;
  for (; optargc != argc; ++optargc) {
    if (StringRef(argv[optargc]) == "--args")
      break;
  }
  llvm::cl::ParseCommandLineOptions(optargc, argv, "arcmt-test");

  if (VerifyTransformedFiles) {
    if (ResultFiles.empty()) {
      llvm::cl::PrintHelpMessage();
      return 1;
    }
    return verifyTransformedFiles(ResultFiles);
  }

  if (optargc == argc) {
    llvm::cl::PrintHelpMessage();
    return 1;
  }

  ArrayRef<const char*> Args(argv+optargc+1, argc-optargc-1);

  if (CheckOnly)
    return checkForMigration(resourcesPath, Args);

  return performTransformations(resourcesPath, Args);
}
