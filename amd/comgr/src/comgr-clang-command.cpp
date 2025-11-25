//===- comgr-clang-command.cpp - ClangCommand implementation --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the CacheCommandAdaptor interface for
/// clang::driver::Commands that are stored in the cache. These correspond to
/// "clang -cc1" and "lld" invocations.
///
//===----------------------------------------------------------------------===//

#include "comgr-clang-command.h"

#include <clang/Driver/Job.h>
#include <llvm/ADT/StringSet.h>

namespace COMGR {
using namespace llvm;
using namespace clang;
namespace {
bool hasDebugOrProfileInfo(ArrayRef<const char *> Args) {
  // These are too difficult to handle since they generate debug info that
  // refers to the temporary paths used by comgr.
  const StringRef Flags[] = {"-fdebug-info-kind", "-fprofile", "-coverage",
                             "-ftime-trace"};

  for (StringRef Arg : Args) {
    for (StringRef Flag : Flags) {
      if (Arg.starts_with(Flag))
        return true;
    }
  }
  return false;
}

Error addFile(CachedCommandAdaptor::HashAlgorithm &H, StringRef Path) {
  auto BufOrError = MemoryBuffer::getFile(Path);
  if (std::error_code EC = BufOrError.getError()) {
    return errorCodeToError(EC);
  }
  StringRef Buf = BufOrError.get()->getBuffer();

  CachedCommandAdaptor::addFileContents(H, Buf);

  return Error::success();
}

template <typename IteratorTy>
bool skipProblematicFlag(IteratorTy &It, const IteratorTy &End) {
  // Skip include paths, these should have been handled by preprocessing the
  // source first. Sadly, these are passed also to the middle-end commands. Skip
  // debug related flags (they should be ignored) like -dumpdir (used for
  // profiling/coverage/split-dwarf).
  // Skip flags related to opencl-c headers or device-libs builtins.
  StringRef Arg = *It;
  static const StringSet<> FlagsWithPathArg = {"-I", "-dumpdir", "-include",
                                               "-mlink-builtin-bitcode"};
  bool IsFlagWithPathArg = It + 1 != End && FlagsWithPathArg.contains(Arg);
  if (IsFlagWithPathArg) {
    ++It;
    return true;
  }

  // Clang always appends the debug compilation dir,
  // even without debug info (in comgr it matches the current directory). We
  // only consider it if the user specified debug information
  const char *FlagsWithEqArg[] = {"-fcoverage-compilation-dir=",
                                  "-fdebug-compilation-dir="};
  bool IsFlagWithSingleArg = any_of(
      FlagsWithEqArg, [&](const char *Flag) { return Arg.starts_with(Flag); });
  if (IsFlagWithSingleArg) {
    return true;
  }

  return false;
}

SmallVector<StringRef, 1> getInputFiles(driver::Command &Command) {
  const auto &CommandInputs = Command.getInputInfos();

  SmallVector<StringRef, 1> Paths;
  Paths.reserve(CommandInputs.size());

  for (const auto &II : CommandInputs) {
    if (!II.isFilename())
      continue;
    Paths.push_back(II.getFilename());
  }

  return Paths;
}

} // namespace
ClangCommand::ClangCommand(driver::Command &Command,
                           DiagnosticOptions &DiagOpts,
                           IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                           ExecuteFnTy &&ExecuteImpl)
    : Command(Command), DiagOpts(DiagOpts), VFS(VFS),
      ExecuteImpl(std::move(ExecuteImpl)) {}

Error ClangCommand::addInputIdentifier(HashAlgorithm &H) const {
  auto Inputs(getInputFiles(Command));
  for (StringRef Input : Inputs) {
    if (Error E = addFile(H, Input)) {
      // call Error's constructor again to silence copy elision warning
      return Error(std::move(E));
    }
  }
  return Error::success();
}

void ClangCommand::addOptionsIdentifier(HashAlgorithm &H) const {
  auto Inputs(getInputFiles(Command));
  StringRef Output = Command.getOutputFilenames().front();
  ArrayRef<const char *> Arguments = Command.getArguments();
  for (auto It = Arguments.begin(), End = Arguments.end(); It != End; ++It) {
    if (skipProblematicFlag(It, End))
      continue;

    StringRef Arg = *It;

    // input files are considered by their content
    // output files should not be considered at all
    bool IsIOFile = Output == Arg || is_contained(Inputs, Arg);
    if (IsIOFile)
      continue;

#ifndef NDEBUG
    bool IsComgrTmpPath =
        CachedCommandAdaptor::searchComgrTmpModel(Arg).has_value();
    // On debug builds, fail on /tmp/comgr-xxxx/... paths.
    // Implicit dependencies should have been considered before.
    // On release builds, add them to the hash to force a cache miss.
    assert(!IsComgrTmpPath &&
           "Unexpected flag and path to comgr temporary directory");
#endif

    addString(H, Arg);
  }
}

ClangCommand::ActionClass ClangCommand::getClass() const {
  return Command.getSource().getKind();
}

bool ClangCommand::canCache() const {
  bool HasOneOutput = Command.getOutputFilenames().size() == 1;
  bool IsPreprocessorCommand = getClass() == driver::Action::PreprocessJobClass;

  return HasOneOutput && !IsPreprocessorCommand &&
         !hasDebugOrProfileInfo(Command.getArguments());
}

Error ClangCommand::writeExecuteOutput(StringRef CachedBuffer) {
  StringRef OutputFilename = Command.getOutputFilenames().front();
  return CachedCommandAdaptor::writeSingleOutputFile(OutputFilename,
                                                     CachedBuffer);
}

Expected<StringRef> ClangCommand::readExecuteOutput() {
  auto MaybeBuffer = CachedCommandAdaptor::readSingleOutputFile(
      Command.getOutputFilenames().front());
  if (!MaybeBuffer)
    return MaybeBuffer.takeError();
  Output = std::move(*MaybeBuffer);
  return Output->getBuffer();
}

amd_comgr_status_t ClangCommand::execute(raw_ostream &LogS) {
  return ExecuteImpl(Command, LogS, DiagOpts, VFS);
}
} // namespace COMGR
