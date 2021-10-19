//===-- clang-nvlink-wrapper/ClangNvlinkWrapper.cpp - wrapper over nvlink-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
///
/// \file
/// This tool works as a wrapper over nvlink program. It transparently passes
/// every input option and objects to nvlink except archive files. It reads
/// each input archive file to extract archived cubin files as temporary files.
/// These temp (*.cubin) files are passed to nvlink, because nvlink does not
/// support linking of archive files implicitly.
///
/// During linking of heterogeneous device archive libraries, the
/// clang-offload-bundler creates a device specific archive of cubin files.
/// Such an archive is then passed to this tool to extract cubin files before
/// passing to nvlink.
///
/// Example:
/// clang-nvlink-wrapper -o a.out-openmp-nvptx64 /tmp/libTest-nvptx-sm_50.a
///
/// 1. Extract (libTest-nvptx-sm_50.a) => /tmp/a.cubin /tmp/b.cubin
/// 2. nvlink -o a.out-openmp-nvptx64 /tmp/a.cubin /tmp/b.cubin
//===---------------------------------------------------------------------===//

#include "clang/Basic/Version.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -help)
// will be hidden.
static cl::OptionCategory
    ClangNvlinkWrapperCategory("clang-nvlink-wrapper options");

static cl::opt<std::string> NvlinkUserPath("nvlink-path",
                                           cl::desc("Path of nvlink binary"),
                                           cl::cat(ClangNvlinkWrapperCategory));

// Do not parse nvlink options
static cl::list<std::string>
    NVArgs(cl::Sink, cl::desc("<options to be passed to nvlink>..."));

static Error runNVLink(std::string NVLinkPath,
                       SmallVectorImpl<std::string> &Args) {
  std::vector<StringRef> NVLArgs;
  NVLArgs.push_back(NVLinkPath);
  for (auto &Arg : Args) {
    NVLArgs.push_back(Arg);
  }

  if (sys::ExecuteAndWait(NVLinkPath.c_str(), NVLArgs))
    return createStringError(inconvertibleErrorCode(), "'nvlink' failed");
  return Error::success();
}

static Error extractArchiveFiles(StringRef Filename,
                                 SmallVectorImpl<std::string> &Args,
                                 SmallVectorImpl<std::string> &TmpFiles) {
  std::vector<std::unique_ptr<MemoryBuffer>> ArchiveBuffers;

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename, false, false);
  if (std::error_code EC = BufOrErr.getError())
    return createFileError(Filename, EC);

  ArchiveBuffers.push_back(std::move(*BufOrErr));
  Expected<std::unique_ptr<llvm::object::Archive>> LibOrErr =
      object::Archive::create(ArchiveBuffers.back()->getMemBufferRef());
  if (!LibOrErr)
    return LibOrErr.takeError();

  auto Archive = std::move(*LibOrErr);

  Error Err = Error::success();
  auto ChildEnd = Archive->child_end();
  for (auto ChildIter = Archive->child_begin(Err); ChildIter != ChildEnd;
       ++ChildIter) {
    if (Err)
      return Err;
    auto ChildNameOrErr = (*ChildIter).getName();
    if (!ChildNameOrErr)
      return ChildNameOrErr.takeError();

    StringRef ChildName = sys::path::filename(ChildNameOrErr.get());

    auto ChildBufferRefOrErr = (*ChildIter).getMemoryBufferRef();
    if (!ChildBufferRefOrErr)
      return ChildBufferRefOrErr.takeError();

    auto ChildBuffer =
        MemoryBuffer::getMemBuffer(ChildBufferRefOrErr.get(), false);
    auto ChildNameSplit = ChildName.split('.');

    SmallString<16> Path;
    int FileDesc;
    if (std::error_code EC = sys::fs::createTemporaryFile(
            (ChildNameSplit.first), (ChildNameSplit.second), FileDesc, Path))
      return createFileError(ChildName, EC);

    std::string TmpFileName(Path.str());
    Args.push_back(TmpFileName);
    TmpFiles.push_back(TmpFileName);
    std::error_code EC;
    raw_fd_ostream OS(Path.c_str(), EC, sys::fs::OF_None);
    if (EC)
      return createFileError(TmpFileName, errc::io_error);
    OS << ChildBuffer->getBuffer();
    OS.close();
  }
  return Err;
}

static Error cleanupTmpFiles(SmallVectorImpl<std::string> &TmpFiles) {
  for (auto &TmpFile : TmpFiles) {
    if (std::error_code EC = sys::fs::remove(TmpFile))
      return createFileError(TmpFile, errc::no_such_file_or_directory);
  }
  return Error::success();
}

static void PrintVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("clang-nvlink-wrapper") << '\n';
}

int main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  cl::SetVersionPrinter(PrintVersion);
  cl::HideUnrelatedOptions(ClangNvlinkWrapperCategory);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A wrapper tool over nvlink program. It transparently passes every \n"
      "input option and objects to nvlink except archive files and path of \n"
      "nvlink binary. It reads each input archive file to extract archived \n"
      "cubin files as temporary files.\n");

  if (Help) {
    cl::PrintHelpMessage();
    return 0;
  }

  auto reportError = [argv](Error E) {
    logAllUnhandledErrors(std::move(E), WithColor::error(errs(), argv[0]));
    exit(1);
  };

  std::string NvlinkPath;
  SmallVector<const char *, 0> Argv(argv, argv + argc);
  SmallVector<std::string, 0> ArgvSubst;
  SmallVector<std::string, 0> TmpFiles;
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  cl::ExpandResponseFiles(Saver, cl::TokenizeGNUCommandLine, Argv);

  for (const std::string &Arg : NVArgs) {
    if (sys::path::extension(Arg) == ".a") {
      if (Error Err = extractArchiveFiles(Arg, ArgvSubst, TmpFiles))
        reportError(std::move(Err));
    } else {
      ArgvSubst.push_back(Arg);
    }
  }

  NvlinkPath = NvlinkUserPath;

  // If user hasn't specified nvlink binary then search it in PATH
  if (NvlinkPath.empty()) {
    ErrorOr<std::string> NvlinkPathErr = sys::findProgramByName("nvlink");
    if (!NvlinkPathErr) {
      reportError(createStringError(NvlinkPathErr.getError(),
                                    "unable to find 'nvlink' in path"));
    }
    NvlinkPath = NvlinkPathErr.get();
  }

  if (Error Err = runNVLink(NvlinkPath, ArgvSubst))
    reportError(std::move(Err));
  if (Error Err = cleanupTmpFiles(TmpFiles))
    reportError(std::move(Err));

  return 0;
}
