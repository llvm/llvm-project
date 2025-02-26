#include "comgr-cache-command.h"
#include "comgr-cache.h"
#include "comgr-device-libs.h"
#include "comgr-env.h"
#include "comgr.h"

#include <clang/Basic/Version.h>
#include <clang/Driver/Job.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringSet.h>

#include <optional>

namespace COMGR {
using namespace llvm;
using namespace clang;

namespace {
// std::isalnum is locale dependent and can have issues
// depending on the stdlib version and application. We prefer to avoid it
bool isalnum(char c) {
  char low[] = {'0', 'a', 'A'};
  char hi[] = {'9', 'z', 'Z'};
  for (unsigned i = 0; i != 3; ++i) {
    if (low[i] <= c && c <= hi[i])
      return true;
  }
  return false;
}

std::optional<size_t> searchComgrTmpModel(StringRef S) {
  // Ideally, we would use std::regex_search with the regex
  // "comgr-[[:alnum:]]{6}". However, due to a bug in stdlibc++
  // (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85824) we have to roll our
  // own search of this regular expression. This bug resulted in a crash in
  // luxmarkv3, during the std::regex constructor.
  const StringRef Prefix = "comgr-";
  const size_t AlnumCount = 6;

  size_t N = S.size();
  size_t Pos = S.find(Prefix);

  size_t AlnumStart = Pos + Prefix.size();
  size_t AlnumEnd = AlnumStart + AlnumCount;
  if (Pos == StringRef::npos || N < AlnumEnd)
    return std::nullopt;

  for (size_t i = AlnumStart; i < AlnumEnd; ++i) {
    if (!isalnum(S[i]))
      return std::nullopt;
  }

  return Pos;
}

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

void addString(CachedCommandAdaptor::HashAlgorithm &H, StringRef S) {
  // hash size + contents to avoid collisions
  // for example, we have to ensure that the result of hashing "AA" "BB" is
  // different from "A" "ABB"
  H.update(S.size());
  H.update(S);
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
  // profiling/coverage/split-dwarf)
  StringRef Arg = *It;
  static const StringSet<> FlagsWithPathArg = {"-I", "-dumpdir"};
  bool IsFlagWithPathArg = It + 1 != End && FlagsWithPathArg.contains(Arg);
  if (IsFlagWithPathArg) {
    ++It;
    return true;
  }

  // Clang always appends the debug compilation dir,
  // even without debug info (in comgr it matches the current directory). We
  // only consider it if the user specified debug information
  bool IsFlagWithSingleArg = Arg.starts_with("-fdebug-compilation-dir=");
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

bool isSourceCodeInput(const driver::InputInfo &II) {
  return driver::types::isSrcFile(II.getType());
}
} // namespace

void CachedCommandAdaptor::addFileContents(
    CachedCommandAdaptor::HashAlgorithm &H, StringRef Buf) {
  // this is a workaround temporary paths getting in the output files of the
  // different commands in #line directives in preprocessed files, and the
  // ModuleID or source_filename in the bitcode.
  while (!Buf.empty()) {
    std::optional<size_t> ComgrTmpPos = searchComgrTmpModel(Buf);
    if (!ComgrTmpPos) {
      addString(H, Buf);
      break;
    }

    StringRef ToHash = Buf.substr(0, *ComgrTmpPos);
    addString(H, ToHash);
    Buf = Buf.substr(ToHash.size() + StringRef("comgr-xxxxxx").size());
  }
}

Expected<CachedCommandAdaptor::Identifier>
CachedCommandAdaptor::getIdentifier() const {
  CachedCommandAdaptor::HashAlgorithm H;
  H.update(getClass());
  H.update(env::shouldEmitVerboseLogs());
  addString(H, getClangFullVersion());
  addString(H, getComgrHashIdentifier());
  addString(H, getDeviceLibrariesIdentifier());

  if (Error E = addInputIdentifier(H))
    return E;

  addOptionsIdentifier(H);

  CachedCommandAdaptor::Identifier Id;
  toHex(H.final(), true, Id);
  return Id;
}

ClangCommand::ClangCommand(driver::Command &Command,
                           DiagnosticOptions &DiagOpts,
                           llvm::vfs::FileSystem &VFS,
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
    static const StringSet<> FlagsWithFileArgEmbededInComgr = {
        "-include-pch", "-mlink-builtin-bitcode"};
    if (FlagsWithFileArgEmbededInComgr.contains(Arg)) {
      // The next argument is a path to a "secondary" input-file (pre-compiled
      // header or device-libs builtin)
      // These two files kinds of files are embedded in comgr at compile time,
      // and in normally their remain constant with comgr's build. The user is
      // not able to change them.
      ++It;
      if (It == End)
        break;
      continue;
    }

    // input files are considered by their content
    // output files should not be considered at all
    bool IsIOFile = Output == Arg || is_contained(Inputs, Arg);
    if (IsIOFile)
      continue;

#ifndef NDEBUG
    bool IsComgrTmpPath = searchComgrTmpModel(Arg).has_value();
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

  // This reduces the applicability of the cache, but it helps us deliver
  // something now and deal with the PCH issues later. The cache would still
  // help for spirv compilation (e.g. bitcode->asm) and for intermediate
  // compilation steps
  bool HasSourceCodeInput = any_of(Command.getInputInfos(), isSourceCodeInput);

  return HasOneOutput && !IsPreprocessorCommand && !HasSourceCodeInput &&
         !hasDebugOrProfileInfo(Command.getArguments());
}

Error ClangCommand::writeExecuteOutput(StringRef CachedBuffer) {
  StringRef OutputFilename = Command.getOutputFilenames().front();
  std::error_code EC;
  raw_fd_ostream Out(OutputFilename, EC);
  if (EC) {
    Error E = createStringError(EC, Twine("Failed to open ") + OutputFilename +
                                        " : " + EC.message() + "\n");
    return E;
  }

  Out.write(CachedBuffer.data(), CachedBuffer.size());
  Out.close();
  if (Out.has_error()) {
    Error E = createStringError(EC, Twine("Failed to write ") + OutputFilename +
                                        " : " + EC.message() + "\n");
    return E;
  }

  return Error::success();
}

Expected<StringRef> ClangCommand::readExecuteOutput() {
  StringRef OutputFilename = Command.getOutputFilenames().front();
  ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr =
      MemoryBuffer::getFile(OutputFilename);
  if (!MBOrErr) {
    std::error_code EC = MBOrErr.getError();
    return createStringError(EC, Twine("Failed to open ") + OutputFilename +
                                     " : " + EC.message() + "\n");
  }
  Output = std::move(*MBOrErr);
  return Output->getBuffer();
}

amd_comgr_status_t ClangCommand::execute(llvm::raw_ostream &LogS) {
  return ExecuteImpl(Command, LogS, DiagOpts, VFS);
}
} // namespace COMGR
