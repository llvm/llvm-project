//===-- llvm-ar.cpp - LLVM archive librarian utility ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Builds up (relatively) standard unix archive files (.a) containing LLVM
// bitcode or other files.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Archive.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#endif

using namespace llvm;

// The name this program was invoked as.
static StringRef ToolName;

// The basename of this program.
static StringRef Stem;

const char NVLWHelp[] = R"(
OVERVIEW: Clang Nvlink Wrapper

USAGE: clang-nvlink-wrapper [options] <objects>

For descriptions of the options please run 'nvlink --help'
The wrapper extracts any arcive objects and call nvlink with the
individual files instead, plus any other options/object.

)";

void printHelpMessage() { outs() << NVLWHelp; }

// Show the error message and exit.
LLVM_ATTRIBUTE_NORETURN static void fail(Twine Error) {
  WithColor::error(errs(), ToolName) << Error << ".\n";
  printHelpMessage();
  exit(1);
}

static void failIfError(std::error_code EC, Twine Context = "") {
  if (!EC)
    return;

  std::string ContextStr = Context.str();
  if (ContextStr.empty())
    fail(EC.message());
  fail(Context + ": " + EC.message());
}

static bool isArchiveFile(StringRef Arg) {
  if (Arg.startswith("-"))
    return false;

  StringRef Extension = sys::path::extension(Arg);
  bool isArchive = Extension == ".a";
  return isArchive;
}

std::vector<std::unique_ptr<llvm::MemoryBuffer>> ArchiveBuffers;
std::vector<std::unique_ptr<llvm::object::Archive>> Archives;

static object::Archive &readArchive(std::unique_ptr<MemoryBuffer> Buf) {
  ArchiveBuffers.push_back(std::move(Buf));
  auto LibOrErr =
      object::Archive::create(ArchiveBuffers.back()->getMemBufferRef());
  failIfError(errorToErrorCode(LibOrErr.takeError()),
              "Could not parse library");
  Archives.push_back(std::move(*LibOrErr));
  return *Archives.back();
}

static void reportError(Twine Error) {
  errs() << "ERROR: " << Error << "\n";
  // FIXME: Handle Errors here.
  // llvm::errs() << Error << ".\n";
}

static bool reportIfError(std::error_code EC, Twine Context = "") {
  if (!EC)
    return false;

  std::string ContextStr = Context.str();
  if (ContextStr.empty())
    reportError(EC.message());
  reportError(Context + ": " + EC.message());
  return true;
}

static bool reportIfError(llvm::Error E, Twine Context = "") {
  if (!E)
    return false;
  ;

  handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EIB) {
    std::string ContextStr = Context.str();
    if (ContextStr.empty())
      reportError(EIB.message());
    reportError(Context + ": " + EIB.message());
  });
  return true;
}

void printNVLinkCommand(std::vector<StringRef> &Command) {
  for (auto &Arg : Command)
    errs() << Arg << " ";
  errs() << "\n";
}

static void runNVLink(std::string NVLinkPath,
                      SmallVectorImpl<std::string> &Args) {
  int ExecResult = -1;
  const char *NVLProgram = NVLinkPath.c_str();
  std::vector<StringRef> NVLArgs;
  NVLArgs.push_back("nvlink");
  for (auto &Arg : Args) {
    NVLArgs.push_back(Arg);
  }
  printNVLinkCommand(NVLArgs);
  ExecResult = llvm::sys::ExecuteAndWait(NVLProgram, NVLArgs);
  if (ExecResult) {
    errs() << "Error: NVlink encountered a problem\n";
  }
}

static void getArchiveFiles(StringRef Filename,
                            SmallVectorImpl<std::string> &Args,
                            SmallVectorImpl<std::string> &TmpFiles) {
  StringRef IFName = Filename;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(IFName, -1, false);

  if (reportIfError(BufOrErr.getError(), "Can't open file " + IFName))
    return;

  auto &Archive = readArchive(std::move(BufOrErr.get()));
  SmallVector<std::string, 8> SourcePaths;

  llvm::Error Err = llvm::Error::success();
  auto ChildEnd = Archive.child_end();
  for (auto ChildIter = Archive.child_begin(Err); ChildIter != ChildEnd;
       ++ChildIter) {
    auto ChildNameOrErr = (*ChildIter).getName();

    if (reportIfError(ChildNameOrErr.takeError(), "No Child Name")) {
      continue;
    }
    StringRef ChildName = llvm::sys::path::filename(ChildNameOrErr.get());

    auto ChildBufferRefOrErr = (*ChildIter).getMemoryBufferRef();
    reportIfError(ChildBufferRefOrErr.takeError(), "No Child Mem Buf");
    auto ChildBuffer =
        MemoryBuffer::getMemBuffer(ChildBufferRefOrErr.get(), false);
    auto ChildNameSplit = ChildName.split('.');
    SmallString<16> Path;
    int FileDesc;
    std::error_code EC = llvm::sys::fs::createTemporaryFile(
        (ChildNameSplit.first), (ChildNameSplit.second), FileDesc, Path);
    if (reportIfError(EC, "Unable to create temporary file")) {
      continue;
    }
    std::string TmpFileName(Path.str().str());

    Args.push_back(TmpFileName);
    TmpFiles.push_back(TmpFileName);
    raw_fd_ostream OS(FileDesc, true);
    OS << ChildBuffer->getBuffer();
    OS.close();
  }
  reportIfError(std::move(Err));
}

static void cleanupTmpFiles(SmallVectorImpl<std::string> &TmpFiles) {
  for (auto &TmpFile : TmpFiles) {
    std::error_code EC = llvm::sys::fs::remove(TmpFile);
    reportIfError(EC, "Unable to delete temporary file");
  }
}

int main(int argc, char **argv) {
  ToolName = argv[0];
  SmallVector<const char *, 0> Argv(argv, argv + argc);
  SmallVector<std::string, 0> ArgvSubst;
  SmallVector<std::string, 0> TmpFiles;
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  cl::ExpandResponseFiles(Saver, cl::TokenizeGNUCommandLine, Argv);

  for (size_t i = 2; i < Argv.size(); ++i) {
    std::string Arg = Argv[i];
    if (isArchiveFile(Arg)) {
      getArchiveFiles(Arg, ArgvSubst, TmpFiles);
    } else {
      ArgvSubst.push_back(Arg);
    }
  }
  runNVLink(Argv[1], ArgvSubst);
  cleanupTmpFiles(TmpFiles);
  return 0;
}
