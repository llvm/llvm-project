//===- cc1depscand_main.cpp - Clang CC1 Dependency Scanning Daemon --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cc1depscanProtocol.h"
#include "clang/Basic/DiagnosticCAS.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/Stack.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "clang/Config/config.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CompileJobCacheKey.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/DependencyScanning/ScanAndUpdateArgs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/CASProvidingFileSystem.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrefixMapper.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/ScopedDurationTimer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/VirtualOutputBackends.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
#include <mutex>
#include <shared_mutex>

#if LLVM_ON_UNIX
#include <sys/file.h> // FIXME: Unix-only. Not portable.
#include <sys/signal.h> // FIXME: Unix-only. Not portable.

#ifdef CLANG_HAVE_RLIMITS
#include <sys/resource.h>
#endif

using namespace clang;
using namespace llvm::opt;
using cc1depscand::DepscanSharing;
using llvm::Error;

#define DEBUG_TYPE "cc1depscand"

ALWAYS_ENABLED_STATISTIC(NumRequests, "Number of -cc1 update requests");

#ifdef CLANG_HAVE_RLIMITS
#if defined(__linux__) && defined(__PIE__)
static size_t getCurrentStackAllocation() {
  // If we can't compute the current stack usage, allow for 512K of command
  // line arguments and environment.
  size_t Usage = 512 * 1024;
  if (FILE *StatFile = fopen("/proc/self/stat", "r")) {
    // We assume that the stack extends from its current address to the end of
    // the environment space. In reality, there is another string literal (the
    // program name) after the environment, but this is close enough (we only
    // need to be within 100K or so).
    unsigned long StackPtr, EnvEnd;
    // Disable silly GCC -Wformat warning that complains about length
    // modifiers on ignored format specifiers. We want to retain these
    // for documentation purposes even though they have no effect.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat"
#endif
    if (fscanf(StatFile,
               "%*d %*s %*c %*d %*d %*d %*d %*d %*u %*lu %*lu %*lu %*lu %*lu "
               "%*lu %*ld %*ld %*ld %*ld %*ld %*ld %*llu %*lu %*ld %*lu %*lu "
               "%*lu %*lu %lu %*lu %*lu %*lu %*lu %*lu %*llu %*lu %*lu %*d %*d "
               "%*u %*u %*llu %*lu %*ld %*lu %*lu %*lu %*lu %*lu %*lu %lu %*d",
               &StackPtr, &EnvEnd) == 2) {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
      Usage = StackPtr < EnvEnd ? EnvEnd - StackPtr : StackPtr - EnvEnd;
    }
    fclose(StatFile);
  }
  return Usage;
}

#include <alloca.h>

LLVM_ATTRIBUTE_NOINLINE
static void ensureStackAddressSpace() {
  // Linux kernels prior to 4.1 will sometimes locate the heap of a PIE binary
  // relatively close to the stack (they are only guaranteed to be 128MiB
  // apart). This results in crashes if we happen to heap-allocate more than
  // 128MiB before we reach our stack high-water mark.
  //
  // To avoid these crashes, ensure that we have sufficient virtual memory
  // pages allocated before we start running.
  size_t Curr = getCurrentStackAllocation();
  const int kTargetStack = DesiredStackSize - 256 * 1024;
  if (Curr < kTargetStack) {
    volatile char *volatile Alloc =
        static_cast<volatile char *>(alloca(kTargetStack - Curr));
    Alloc[0] = 0;
    Alloc[kTargetStack - Curr - 1] = 0;
  }
}
#else
static void ensureStackAddressSpace() {}
#endif

/// Attempt to ensure that we have at least 8MiB of usable stack space.
static void ensureSufficientStack() {
  struct rlimit rlim;
  if (getrlimit(RLIMIT_STACK, &rlim) != 0)
    return;

  // Increase the soft stack limit to our desired level, if necessary and
  // possible.
  if (rlim.rlim_cur != RLIM_INFINITY &&
      rlim.rlim_cur < rlim_t(DesiredStackSize)) {
    // Try to allocate sufficient stack.
    if (rlim.rlim_max == RLIM_INFINITY ||
        rlim.rlim_max >= rlim_t(DesiredStackSize))
      rlim.rlim_cur = DesiredStackSize;
    else if (rlim.rlim_cur == rlim.rlim_max)
      return;
    else
      rlim.rlim_cur = rlim.rlim_max;

    if (setrlimit(RLIMIT_STACK, &rlim) != 0 ||
        rlim.rlim_cur != DesiredStackSize)
      return;
  }


  // We should now have a stack of size at least DesiredStackSize. Ensure
  // that we can actually use that much, if necessary.
  ensureStackAddressSpace();
}
#else
static void ensureSufficientStack() {}
#endif

namespace {
class OneOffCompilationDatabase : public tooling::CompilationDatabase {
public:
  OneOffCompilationDatabase() = delete;
  template <class... ArgsT>
  OneOffCompilationDatabase(ArgsT &&... Args)
      : Command(std::forward<ArgsT>(Args)...) {}

  std::vector<tooling::CompileCommand>
  getCompileCommands(StringRef FilePath) const override {
    return {Command};
  }

  std::vector<tooling::CompileCommand> getAllCompileCommands() const override {
    return {Command};
  }

private:
  tooling::CompileCommand Command;
};
}

namespace {
class SharedStream {
public:
  SharedStream(raw_ostream &OS) : OS(OS) {}
  void applyLocked(llvm::function_ref<void(raw_ostream &OS)> Fn) {
    std::unique_lock<std::mutex> LockGuard(Lock);
    Fn(OS);
    OS.flush();
  }

private:
  std::mutex Lock;
  raw_ostream &OS;
};
} // namespace

namespace {
/// FIXME: Move to LLVMSupport; probably llvm/Support/Process.h.
///
/// TODO: Get this working on Linux:
/// - Reading `/proc/[pid]/comm` for the command names.
/// - Walk up the `ppid` fields in `/proc/[pid]/stat`.
struct ProcessAncestor {
  uint64_t PID = ~0ULL;
  uint64_t PPID = ~0ULL;
  StringRef Name;
};
class ProcessAncestorIterator
    : public llvm::iterator_facade_base<ProcessAncestorIterator,
                                        std::forward_iterator_tag,
                                        const ProcessAncestor> {

public:
  ProcessAncestorIterator() = default;

  static uint64_t getThisPID();
  static uint64_t getParentPID();

  static ProcessAncestorIterator getThisBegin() {
    return ProcessAncestorIterator().setPID(getThisPID());
  }
  static ProcessAncestorIterator getParentBegin() {
    return ProcessAncestorIterator().setPID(getParentPID());
  }

  const ProcessAncestor &operator*() const { return Ancestor; }
  ProcessAncestorIterator &operator++() { return setPID(Ancestor.PPID); }
  bool operator==(const ProcessAncestorIterator &RHS) const {
    return Ancestor.PID == RHS.Ancestor.PID;
  }

private:
  ProcessAncestorIterator &setPID(uint64_t NewPID);

  ProcessAncestor Ancestor;
#ifdef USE_APPLE_LIBPROC_FOR_DEPSCAN_ANCESTORS
  proc_bsdinfo ProcInfo;
#endif
};
} // end namespace

uint64_t ProcessAncestorIterator::getThisPID() {
  // FIXME: Not portable.
  return ::getpid();
}

uint64_t ProcessAncestorIterator::getParentPID() {
  // FIXME: Not portable.
  return ::getppid();
}

ProcessAncestorIterator &ProcessAncestorIterator::setPID(uint64_t NewPID) {
  // Reset state in case NewPID isn't found.
  Ancestor = ProcessAncestor();

#ifdef USE_APPLE_LIBPROC_FOR_DEPSCAN_ANCESTORS
  pid_t TypeCorrectPID = NewPID;
  if (proc_pidinfo(TypeCorrectPID, PROC_PIDTBSDINFO, 0, &ProcInfo,
                   sizeof(ProcInfo)) != sizeof(ProcInfo))
    return *this; // Not found or no access.

  Ancestor.PID = NewPID;
  Ancestor.PPID = ProcInfo.pbi_ppid;
  Ancestor.Name = StringRef(ProcInfo.pbi_name);
#else
  (void)NewPID;
#endif
  return *this;
}

static std::optional<std::string>
makeDepscanDaemonKey(StringRef Mode, const DepscanSharing &Sharing) {
  auto completeKey = [&Sharing](llvm::BLAKE3 &Hasher) -> std::string {
    // Only share depscan daemons that originated from the same clang version.
    Hasher.update(getClangFullVersion());
    for (const char *Arg : Sharing.CASArgs)
      Hasher.update(StringRef(Arg));
    // Using same hash size as the module cache hash.
    auto Hash = Hasher.final<sizeof(uint64_t)>();
    uint64_t HashVal =
        llvm::support::endian::read<uint64_t, llvm::endianness::native>(
            Hash.data());
    return toString(llvm::APInt(64, HashVal), 36, /*Signed=*/false);
  };

  auto makePIDKey = [&completeKey](uint64_t PID) -> std::string {
    llvm::BLAKE3 Hasher;
    Hasher.update(
        ArrayRef(reinterpret_cast<uint8_t *>(&PID), sizeof(PID)));
    return completeKey(Hasher);
  };
  auto makeIdentifierKey = [&completeKey](StringRef Ident) -> std::string {
    llvm::BLAKE3 Hasher;
    Hasher.update(Ident);
    return completeKey(Hasher);
  };

  if (Sharing.ShareViaIdentifier)
    return makeIdentifierKey(*Sharing.Name);

  if (Sharing.Name) {
    // Check for fast path, which doesn't need to look up process names:
    // -fdepscan-share-parent without -fdepscan-share-stop.
    if (Sharing.Name->empty() && !Sharing.Stop)
      return makePIDKey(ProcessAncestorIterator::getParentPID());

    // Check the parent's process name, and then process ancestors.
    for (ProcessAncestorIterator I = ProcessAncestorIterator::getParentBegin(),
                                 IE;
         I != IE; ++I) {
      if (I->Name == Sharing.Stop)
        break;
      if (Sharing.Name->empty() || I->Name == *Sharing.Name)
        return makePIDKey(I->PID);
      if (Sharing.OnlyShareParent)
        break;
    }

    // Fall through if the process to share isn't found.
  }

  // Still daemonize, but use the PID from this process as the key to avoid
  // sharing state.
  if (Mode == "daemon")
    return makePIDKey(ProcessAncestorIterator::getThisPID());

  // Mode == "auto".
  //
  // TODO: consider returning ThisPID (same as "daemon") once the daemon can
  // share a CAS instance without sharing filesystem caching. Or maybe delete
  // "auto" at that point and make "-fdepscan" default to "-fdepscan=daemon".
  return std::nullopt;
}

static std::optional<std::string>
makeDepscanDaemonPath(StringRef Mode, const DepscanSharing &Sharing) {
  if (Mode == "inline")
    return std::nullopt;

  if (Sharing.Path)
    return Sharing.Path->str();

  if (auto Key = makeDepscanDaemonKey(Mode, Sharing))
    return cc1depscand::getBasePath(*Key);

  return std::nullopt;
}

static Expected<llvm::cas::CASID> scanAndUpdateCC1Inline(
    const char *Exec, ArrayRef<const char *> InputArgs,
    StringRef WorkingDirectory, SmallVectorImpl<const char *> &OutputArgs,
    bool ProduceIncludeTree, bool &DiagnosticErrorOccurred,
    llvm::function_ref<const char *(const Twine &)> SaveArg,
    const CASOptions &CASOpts, std::shared_ptr<llvm::cas::ObjectStore> DB,
    std::shared_ptr<llvm::cas::ActionCache> Cache);

static Expected<llvm::cas::CASID> scanAndUpdateCC1InlineWithTool(
    tooling::dependencies::DependencyScanningTool &Tool,
    DiagnosticConsumer &DiagsConsumer, raw_ostream *VerboseOS, const char *Exec,
    ArrayRef<const char *> InputArgs, StringRef WorkingDirectory,
    SmallVectorImpl<const char *> &OutputArgs, llvm::cas::ObjectStore &DB,
    llvm::function_ref<const char *(const Twine &)> SaveArg);

static llvm::Expected<llvm::cas::CASID> scanAndUpdateCC1UsingDaemon(
    const char *Exec, ArrayRef<const char *> OldArgs,
    StringRef WorkingDirectory, SmallVectorImpl<const char *> &NewArgs,
    bool &DiagnosticErrorOccurred, StringRef Path,
    const DepscanSharing &Sharing,
    llvm::function_ref<const char *(const Twine &)> SaveArg,
    llvm::cas::ObjectStore &CAS) {
  using namespace clang::cc1depscand;

  // FIXME: Skip some of this if -fcas-fs has been passed.

  bool NoSpawnDaemon = (bool)Sharing.Path;
  // llvm::dbgs() << "connecting to daemon...\n";
  auto Daemon = NoSpawnDaemon
                    ? ScanDaemon::connectToDaemonAndShakeHands(Path)
                    : ScanDaemon::constructAndShakeHands(Path, Exec, Sharing);
  if (!Daemon)
    return Daemon.takeError();
  CC1DepScanDProtocol Comms(*Daemon);

  // llvm::dbgs() << "sending request...\n";
  if (auto E = Comms.putCommand(WorkingDirectory, OldArgs))
    return std::move(E);

  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver Saver(Alloc);
  SmallVector<const char *> RawNewArgs;
  StringRef DiagnosticOutput;
  CC1DepScanDProtocol::ResultKind Result;
  StringRef FailedReason;
  StringRef RootID;
  if (auto E = Comms.getScanResult(Saver, Result, FailedReason, RootID,
                                   RawNewArgs, DiagnosticOutput))
    return std::move(E);

  if (Result != CC1DepScanDProtocol::SuccessResult)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "depscan daemon failed: " + FailedReason);

  // FIXME: Avoid this duplication.
  NewArgs.resize(RawNewArgs.size());
  for (int I = 0, E = RawNewArgs.size(); I != E; ++I)
    NewArgs[I] = SaveArg(RawNewArgs[I]);

  DiagnosticErrorOccurred = !DiagnosticOutput.empty();
  if (DiagnosticErrorOccurred) {
    llvm::errs() << DiagnosticOutput;
  }

  return CAS.parseID(RootID);
}

// FIXME: This is a copy of Command::writeResponseFile. Command is too deeply
// tied with clang::Driver to use directly.
static void writeResponseFile(raw_ostream &OS,
                              SmallVectorImpl<const char *> &Arguments) {
  for (const auto *Arg : Arguments) {
    OS << '"';

    for (; *Arg != '\0'; Arg++) {
      if (*Arg == '\"' || *Arg == '\\') {
        OS << '\\';
      }
      OS << *Arg;
    }

    OS << "\" ";
  }
  OS << "\n";
}

static int scanAndUpdateCC1(const char *Exec, ArrayRef<const char *> OldArgs,
                            SmallVectorImpl<const char *> &NewArgs,
                            DiagnosticsEngine &Diag,
                            const llvm::opt::ArgList &Args,
                            const CASOptions &CASOpts,
                            std::shared_ptr<llvm::cas::ObjectStore> DB,
                            std::shared_ptr<llvm::cas::ActionCache> Cache,
                            std::optional<llvm::cas::CASID> &RootID) {
  using namespace clang::driver;

  llvm::ScopedDurationTimer ScopedTime([&Diag](double Seconds) {
    Diag.Report(diag::remark_compile_job_cache_timing_depscan)
        << llvm::format("%.6fs", Seconds);
  });

  StringRef WorkingDirectory;
  SmallString<128> WorkingDirectoryBuf;
  if (auto *Arg =
          Args.getLastArg(clang::driver::options::OPT_working_directory)) {
    WorkingDirectory = Arg->getValue();
  } else {
    if (llvm::Error E = llvm::errorCodeToError(
            llvm::sys::fs::current_path(WorkingDirectoryBuf))) {
      Diag.Report(diag::err_cas_depscan_failed) << std::move(E);
      return 1;
    }
    WorkingDirectory = WorkingDirectoryBuf;
  }

  // Collect these before returning to ensure they're claimed.
  DepscanSharing Sharing;
  if (Arg *A = Args.getLastArg(options::OPT_fdepscan_share_stop_EQ))
    Sharing.Stop = A->getValue();
  if (Arg *A = Args.getLastArg(options::OPT_fdepscan_share_EQ,
                               options::OPT_fdepscan_share_identifier,
                               options::OPT_fdepscan_share_parent,
                               options::OPT_fdepscan_share_parent_EQ,
                               options::OPT_fno_depscan_share)) {
    if (A->getOption().matches(options::OPT_fdepscan_share_EQ) ||
        A->getOption().matches(options::OPT_fdepscan_share_parent_EQ)) {
      Sharing.Name = A->getValue();
      Sharing.OnlyShareParent =
          A->getOption().matches(options::OPT_fdepscan_share_parent_EQ);
    } else if (A->getOption().matches(options::OPT_fdepscan_share_parent)) {
      Sharing.Name = "";
      Sharing.OnlyShareParent = true;
    } else if (A->getOption().matches(options::OPT_fdepscan_share_identifier)) {
      Sharing.Name = A->getValue();
      Sharing.ShareViaIdentifier = true;
    }
  }
  if (Arg *A = Args.getLastArg(options::OPT_fdepscan_daemon_EQ))
    Sharing.Path = A->getValue();

  StringRef Mode = "auto";
  if (Arg *A = Args.getLastArg(clang::driver::options::OPT_fdepscan_EQ)) {
    Mode = A->getValue();
    // Note: -cc1depscan does not accept '-fdepscan=off'.
    if (Mode != "daemon" && Mode != "inline" && Mode != "auto") {
      Diag.Report(diag::err_drv_invalid_argument_to_option)
          << Mode << A->getOption().getName();
      return 1;
    }
  }

  bool ProduceIncludeTree = Args.hasArg(options::OPT_fdepscan_include_tree);

  auto SaveArg = [&Args](const Twine &T) { return Args.MakeArgString(T); };
  CompilerInvocation::GenerateCASArgs(CASOpts, Sharing.CASArgs, SaveArg);
  if (ProduceIncludeTree)
    Sharing.CASArgs.push_back("-fdepscan-include-tree");

  bool DiagnosticErrorOccurred = false;
  auto ScanAndUpdate = [&]() {
    if (std::optional<std::string> DaemonPath =
            makeDepscanDaemonPath(Mode, Sharing))
      return scanAndUpdateCC1UsingDaemon(Exec, OldArgs, WorkingDirectory,
                                         NewArgs, DiagnosticErrorOccurred,
                                         *DaemonPath, Sharing, SaveArg, *DB);
    return scanAndUpdateCC1Inline(Exec, OldArgs, WorkingDirectory, NewArgs,
                                  ProduceIncludeTree, DiagnosticErrorOccurred,
                                  SaveArg, CASOpts, DB, Cache);
  };
  if (llvm::Error E = ScanAndUpdate().moveInto(RootID)) {
    Diag.Report(diag::err_cas_depscan_failed) << std::move(E);
    return 1;
  }
  return DiagnosticErrorOccurred;
}

int cc1depscan_main(ArrayRef<const char *> Argv, const char *Argv0,
                    void *MainAddr) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  {
    auto FoundCC1Args =
        std::find_if(Argv.begin(), Argv.end(), [](const char *Arg) -> bool {
          return StringRef(Arg).equals("-cc1-args");
        });
    if (FoundCC1Args != Argv.end()) {
      SmallVector<const char *, 8> WarnOpts{Argv0};
      WarnOpts.append(FoundCC1Args + 1, Argv.end());
      DiagOpts = CreateAndPopulateDiagOpts(WarnOpts);
    } else {
      DiagOpts = new DiagnosticOptions();
    }
  }
  auto DiagsConsumer = std::make_unique<TextDiagnosticPrinter>(
      llvm::errs(), DiagOpts.get(), false);
  DiagnosticsEngine Diags(new DiagnosticIDs(), DiagOpts);
  Diags.setClient(DiagsConsumer.get(), /*ShouldOwnClient=*/false);
  ProcessWarningOptions(Diags, *DiagOpts);
  if (Diags.hasErrorOccurred())
    return 1;

  // FIXME: Create a new OptionFlag group for cc1depscan.
  const OptTable &Opts = clang::driver::getDriverOptTable();
  unsigned MissingArgIndex, MissingArgCount;
  auto Args = Opts.ParseArgs(Argv, MissingArgIndex, MissingArgCount);
  if (MissingArgCount) {
    Diags.Report(diag::err_drv_missing_argument)
        << Args.getArgString(MissingArgIndex) << MissingArgCount;
    return 1;
  }

  auto *CC1Args = Args.getLastArg(clang::driver::options::OPT_cc1_args);
  if (!CC1Args) {
    llvm::errs() << "missing -cc1-args option\n";
    return 1;
  }

  auto *OutputArg = Args.getLastArg(clang::driver::options::OPT_o);
  std::string OutputPath = OutputArg ? OutputArg->getValue() : "-";

  std::optional<StringRef> DumpDepscanTree;
  if (auto *Arg =
          Args.getLastArg(clang::driver::options::OPT_dump_depscan_tree_EQ))
    DumpDepscanTree = Arg->getValue();

  SmallVector<const char *> NewArgs;
  std::optional<llvm::cas::CASID> RootID;

  CASOptions CASOpts;
  auto ParsedCC1Args =
      Opts.ParseArgs(CC1Args->getValues(), MissingArgIndex, MissingArgCount);
  CompilerInvocation::ParseCASArgs(CASOpts, ParsedCC1Args, Diags);
  CASOpts.ensurePersistentCAS();

  auto [CAS, Cache] = CASOpts.getOrCreateDatabases(Diags);
  if (!CAS || !Cache)
    return 1;

  if (int Ret = scanAndUpdateCC1(Argv0, CC1Args->getValues(), NewArgs, Diags,
                                 Args, CASOpts, CAS, Cache, RootID))
    return Ret;

  // FIXME: Use OutputBackend to OnDisk only now.
  auto OutputBackend =
      llvm::makeIntrusiveRefCnt<llvm::vfs::OnDiskOutputBackend>();
  auto OutputFile = consumeDiscardOnDestroy(
      OutputBackend->createFile(OutputPath, llvm::vfs::OutputConfig()
                                                .setTextWithCRLF(true)
                                                .setDiscardOnSignal(false)
                                                .setAtomicWrite(false)));
  if (!OutputFile) {
    Diags.Report(diag::err_fe_unable_to_open_output)
        << OutputArg->getValue() << llvm::toString(OutputFile.takeError());
    return 1;
  }

  if (DumpDepscanTree) {
    std::error_code EC;
    llvm::raw_fd_ostream RootOS(*DumpDepscanTree, EC);
    if (EC)
      Diags.Report(diag::err_fe_unable_to_open_output)
          << *DumpDepscanTree << EC.message();
    RootOS << RootID->toString() << "\n";
  }
  writeResponseFile(*OutputFile, NewArgs);

  if (auto Err = OutputFile->keep()) {
    llvm::errs() << "failed closing outputfile: "
                 << llvm::toString(std::move(Err)) << "\n";
    return 1;
  }
  return 0;
}

namespace {
struct ScanServer {
  const char *Argv0 = nullptr;
  SmallString<128> BasePath;
  /// List of cas options.
  ArrayRef<const char *> CASArgs;
  int PidFD = -1;
  int ListenSocket = -1;
  /// \p std::nullopt means it runs indefinitely.
  std::optional<unsigned> TimeoutSeconds;
  std::atomic<bool> ShutDown{false};

  ~ScanServer() { shutdown(); }

  void start(bool Exclusive);
  int listen();

  /// Tear down the socket and bind file immediately but wait till all existing
  /// jobs to finish.
  void shutdown() {
    ShutDown.store(true);
    cc1depscand::unlinkBoundSocket(BasePath);
    // Clean up the pidfile when we're done.
    if (PidFD != -1)
      ::close(PidFD);
    ::shutdown(ListenSocket, SHUT_RD);
    ::close(ListenSocket);
  }
};
} // anonymous namespace

static llvm::ExitOnError ExitOnErr("clang -cc1depscand: ");

static void reportError(const llvm::Twine &Message) {
  ExitOnErr(llvm::createStringError(llvm::inconvertibleErrorCode(), Message));
}

/// Accepted options are:
///
/// * -run <path> [-long-running] [-cas-args ...]
/// Runs the daemon until a timeout is reached without a new connection.
/// "-long-running" increases the timeout. stdout/stderr are redirected to files
/// relative to <path>. This is how the clang driver auto-spawns a new daemon.
///
/// * -serve <path> [-cas-args ...]
/// Runs indefinitely (until ctrl+c or the process is killed). There's no
/// stdout/stderr redirection. Useful for debugging and for starting a permanent
/// daemon before directing clang invocations to connect to it.
///
/// * -execute <path> [-cas-args ...] -- <command ...>
/// Sets up the socket path, sets \p CLANG_CACHE_SCAN_DAEMON_SOCKET_PATH
/// enviroment variable to the socket path, and executes the provided command.
/// It exits with the same exit code as the command. Useful for lit testing.
int cc1depscand_main(ArrayRef<const char *> Argv, const char *Argv0,
                     void *MainAddr) {
  ensureSufficientStack();

  if (Argv.size() < 2)
    reportError("missing command and base-path");

  ScanServer Server;
  Server.Argv0 = Argv0;

  StringRef Command = Argv[0];
  Server.BasePath = Argv[1];

  ArrayRef<const char *> CommandArgsToExecute;
  auto Sep = llvm::find_if(
      Argv, [](const char *Arg) { return StringRef(Arg) == "--"; });
  if (Sep != Argv.end()) {
    CommandArgsToExecute = Argv.drop_front(Sep - Argv.begin() + 1);
    Argv = Argv.slice(0, Sep - Argv.begin());
  }

  // Whether the daemon can safely stay alive a longer period of time.
  // FIXME: Consider designing a mechanism to notify daemons, started for a
  // particular "build session", to shutdown, then have it stay alive until the
  // session is finished.
  bool LongRunning = false;

  for (const auto *A = Argv.begin() + 2; A != Argv.end(); ++A) {
    StringRef Arg(*A);
    if (Arg == "-long-running")
      LongRunning = true;
    else if (Arg == "-cas-args") {
      Server.CASArgs = ArrayRef(A + 1, Argv.end());
      break;
    }
  }

  // Create the base directory if necessary.
  StringRef BaseDir = llvm::sys::path::parent_path(Server.BasePath);
  if (std::error_code EC = llvm::sys::fs::create_directories(BaseDir))
    reportError(Twine("cannot create basedir: ") + EC.message());

  if (Command == "-serve") {
    Server.start(/*Exclusive*/ true);
    return Server.listen();

  } else if (Command == "-execute") {
    SmallVector<StringRef, 32> RefArgs;
    RefArgs.reserve(CommandArgsToExecute.size());
    for (const char *Arg : CommandArgsToExecute) {
      RefArgs.push_back(Arg);
    }

    // Make sure to start the server before executing the command.
    Server.start(/*Exclusive*/ true);
    std::thread ServerThread([&Server]() { Server.listen(); });

    setenv("CLANG_CACHE_SCAN_DAEMON_SOCKET_PATH", Server.BasePath.c_str(),
           true);

    std::string ErrMsg;
    int Result = llvm::sys::ExecuteAndWait(
        RefArgs.front(), RefArgs, /*Env*/ std::nullopt,
        /*Redirects*/ {}, /*SecondsToWait*/ 0,
        /*MemoryLimit*/ 0, &ErrMsg);

    Server.shutdown();
    ServerThread.join();

    if (!ErrMsg.empty())
      reportError("failed executing command: " + ErrMsg);
    return Result;

  } else if (Command != "-run") {
    reportError("unknown command '" + Command + "'");
  }

  Server.TimeoutSeconds = LongRunning ? 45 : 15;

  // Daemonize.
  if (::signal(SIGHUP, SIG_IGN) == SIG_ERR)
    reportError("failed to ignore SIGHUP");
  if (::setsid() == -1)
    reportError("setsid failed");

  // Check the pidfile.
  SmallString<128> LogOutPath, LogErrPath;
  (Server.BasePath + ".out").toVector(LogOutPath);
  (Server.BasePath + ".err").toVector(LogErrPath);

  auto openAndReplaceFD = [&](int ReplacedFD, StringRef Path) {
    int FD;
    if (std::error_code EC = llvm::sys::fs::openFile(
            Path, FD, llvm::sys::fs::CD_CreateAlways, llvm::sys::fs::FA_Write,
            llvm::sys::fs::OF_None)) {
      // Ignoring error?
      ::close(ReplacedFD);
      return;
    }
    ::dup2(FD, ReplacedFD);
    ::close(FD);
  };
  openAndReplaceFD(1, LogOutPath);
  openAndReplaceFD(2, LogErrPath);

  Server.start(/*Exclusive*/ false);
  return Server.listen();
}

void ScanServer::start(bool Exclusive) {
  // Check the pidfile.
  SmallString<128> PidPath;
  (BasePath + ".pid").toVector(PidPath);

  [&]() {
    if (std::error_code EC = llvm::sys::fs::openFile(
            PidPath, PidFD, llvm::sys::fs::CD_OpenAlways,
            llvm::sys::fs::FA_Write, llvm::sys::fs::OF_None))
      reportError("cannot open pidfile");

    // Try to lock; failure means there's another daemon running.
    if (::flock(PidFD, LOCK_EX | LOCK_NB)) {
      if (Exclusive)
        reportError("another daemon using the base path");
      ::exit(0);
    }

    // FIXME: Should we actually write the pid here? Maybe we don't care.
  }();

  // Open the socket and start listening.
  ListenSocket = cc1depscand::createSocket();
  if (ListenSocket == -1)
    reportError("cannot open socket");

  if (cc1depscand::bindToSocket(BasePath, ListenSocket))
    reportError(StringRef() + "cannot bind to socket" + ": " + strerror(errno));

  unsigned MaxBacklog =
      llvm::hardware_concurrency().compute_thread_count() * 16;
  if (::listen(ListenSocket, MaxBacklog))
    reportError("cannot listen to socket");
}

int ScanServer::listen() {
  llvm::ThreadPool Pool;

  DiagnosticsEngine Diags(new DiagnosticIDs(), new DiagnosticOptions());
  CASOptions CASOpts;
  const OptTable &Opts = clang::driver::getDriverOptTable();
  unsigned MissingArgIndex, MissingArgCount;
  auto ParsedCASArgs =
      Opts.ParseArgs(CASArgs, MissingArgIndex, MissingArgCount);
  CompilerInvocation::ParseCASArgs(CASOpts, ParsedCASArgs, Diags);
  CASOpts.ensurePersistentCAS();
  bool ProduceIncludeTree =
      ParsedCASArgs.hasArg(driver::options::OPT_fdepscan_include_tree);

  std::shared_ptr<llvm::cas::ObjectStore> CAS;
  std::shared_ptr<llvm::cas::ActionCache> Cache;
  std::tie(CAS, Cache) = CASOpts.getOrCreateDatabases(Diags);
  if (!CAS)
    reportError("cannot create CAS");
  if (!Cache)
    reportError("cannot create ActionCache");

  IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> FS;
  if (!ProduceIncludeTree)
    FS = llvm::cantFail(llvm::cas::createCachingOnDiskFileSystem(*CAS));
  tooling::dependencies::DependencyScanningService Service(
      tooling::dependencies::ScanningMode::DependencyDirectivesScan,
      ProduceIncludeTree
          ? tooling::dependencies::ScanningOutputFormat::IncludeTree
          : tooling::dependencies::ScanningOutputFormat::Tree,
      CASOpts, CAS, Cache, FS,
      /*ReuseFileManager=*/false,
      /*SkipExcludedPPRanges=*/true);

  std::atomic<int> NumRunning(0);

  std::chrono::steady_clock::time_point Start =
      std::chrono::steady_clock::now();
  std::atomic<uint64_t> SecondsSinceLastClose;

  SharedStream SharedOS(llvm::errs());

  auto ServiceLoop = [this, &CAS, &Service, &NumRunning, &Start,
                      &SecondsSinceLastClose, &SharedOS](unsigned I) {
    std::optional<tooling::dependencies::DependencyScanningTool> Tool;
    SmallString<256> Message;
    while (true) {
      if (ShutDown.load())
        return;

      int Data = cc1depscand::acceptSocket(ListenSocket);
      if (Data == -1)
        continue;

      auto CloseData = llvm::make_scope_exit([&]() { ::close(Data); });
      cc1depscand::CC1DepScanDProtocol Comms(Data);

      auto StopRunning = llvm::make_scope_exit([&]() {
        SecondsSinceLastClose.store(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - Start)
                .count());
        --NumRunning;
      });

      {
        ++NumRunning;

        // Check again for shutdown, since the main thread could have
        // requested it before we created the service.
        //
        // FIXME: Return std::optional<ServiceReference> from the map, handling
        // this condition in getOrCreateService().
        if (ShutDown.load()) {
          // Abort the work in shutdown state since the thread can go down
          // anytime.
          return; // FIXME: Tell the client about this?
        }
      }

      // First put a result kind as a handshake.
      if (auto E = Comms.putResultKind(
              cc1depscand::CC1DepScanDProtocol::SuccessResult)) {
        SharedOS.applyLocked([&](raw_ostream &OS) {
          OS << I << ": failed to send handshake\n";
          logAllUnhandledErrors(std::move(E), OS);
        });
        continue; // go back to wait when handshake failed.
      }

      llvm::BumpPtrAllocator Alloc;
      llvm::StringSaver Saver(Alloc);
      StringRef WorkingDirectory;
      SmallVector<const char *> Args;
      if (llvm::Error E = Comms.getCommand(Saver, WorkingDirectory, Args)) {
        SharedOS.applyLocked([&](raw_ostream &OS) {
          OS << I << ": failed to get command\n";
          logAllUnhandledErrors(std::move(E), OS);
        });
        continue; // FIXME: Tell the client something went wrong.
      }

      // cc1 request.
      ++NumRequests;
      auto printScannedCC1 = [&](raw_ostream &OS) {
        OS << I << ": scanned -cc1:";
        for (const char *Arg : Args)
          OS << " " << Arg;
        OS << "\n";
      };

      bool ProduceIncludeTree =
          Service.getFormat() ==
          tooling::dependencies::ScanningOutputFormat::IncludeTree;

      // Is this safe to reuse? Or does DependendencyScanningWorkerFileSystem
      // make some bad assumptions about relative paths?
      if (!Tool) {
        llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> UnderlyingFS =
            llvm::vfs::createPhysicalFileSystem();
        if (ProduceIncludeTree)
          UnderlyingFS = llvm::cas::createCASProvidingFileSystem(
              CAS, std::move(UnderlyingFS));
        Tool.emplace(Service, std::move(UnderlyingFS));
      }

      IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts =
          CreateAndPopulateDiagOpts(Args);
      SmallString<128> DiagsBuffer;
      llvm::raw_svector_ostream DiagsOS(DiagsBuffer);
      DiagsOS.enable_colors(true);
      auto DiagsConsumer = std::make_unique<TextDiagnosticPrinter>(
          DiagsOS, DiagOpts.get(), false);

      SmallVector<const char *> NewArgs;
      auto RootID = scanAndUpdateCC1InlineWithTool(
          *Tool, *DiagsConsumer, &DiagsOS, Argv0, Args, WorkingDirectory,
          NewArgs, *CAS, [&](const Twine &T) { return Saver.save(T).data(); });
      if (!RootID) {
        consumeError(Comms.putScanResultFailed(toString(RootID.takeError())));
        SharedOS.applyLocked([&](raw_ostream &OS) {
          printScannedCC1(OS);
          OS << I << ": failed to create compiler invocation\n";
          OS << DiagsBuffer;
        });
        continue;
      }

      auto printComputedCC1 = [&](raw_ostream &OS) {
        OS << I << ": sending back new -cc1 args:\n";
        for (const char *Arg : NewArgs)
          OS << " " << Arg;
        OS << "\n";
      };
      if (llvm::Error E = Comms.putScanResultSuccess(RootID->toString(),
                                                     NewArgs, DiagsOS.str())) {
        SharedOS.applyLocked([&](raw_ostream &OS) {
          printScannedCC1(OS);
          printComputedCC1(OS);
          logAllUnhandledErrors(std::move(E), OS);
        });
        continue; // FIXME: Tell the client something went wrong.
      }

      // Done!
#ifndef NDEBUG
      // In +asserts mode, print out -cc1s even on success.
      SharedOS.applyLocked([&](raw_ostream &OS) {
        printScannedCC1(OS);
        printComputedCC1(OS);
      });
#endif
    }
  };

  for (unsigned I = 0; I < Pool.getThreadCount(); ++I)
    Pool.async(ServiceLoop, I);

  if (!TimeoutSeconds) {
    Pool.wait();
    return 0;
  }

  // Wait for the work to finish.
  const uint64_t SecondsBetweenAttempts = 5;
  const uint64_t SecondsBeforeDestruction = *TimeoutSeconds;
  uint64_t SleepTime = SecondsBeforeDestruction;
  while (true) {
    ::sleep(SleepTime);
    SleepTime = SecondsBetweenAttempts;

    if (NumRunning.load())
      continue;

    if (ShutDown.load())
      break;

    // Figure out the latest access time that we'll delete.
    uint64_t LastAccessToDestroy =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - Start)
            .count();
    if (LastAccessToDestroy < SecondsBeforeDestruction)
      continue; // In case ::sleep returns slightly early.
    LastAccessToDestroy -= SecondsBeforeDestruction;

    if (LastAccessToDestroy < SecondsSinceLastClose)
      continue;

    shutdown();
  }

  return 0;
}

static Expected<llvm::cas::CASID> scanAndUpdateCC1InlineWithTool(
    tooling::dependencies::DependencyScanningTool &Tool,
    DiagnosticConsumer &DiagsConsumer, raw_ostream *VerboseOS, const char *Exec,
    ArrayRef<const char *> InputArgs, StringRef WorkingDirectory,
    SmallVectorImpl<const char *> &OutputArgs, llvm::cas::ObjectStore &DB,
    llvm::function_ref<const char *(const Twine &)> SaveArg) {
  DiagnosticsEngine Diags(new DiagnosticIDs(), new DiagnosticOptions());
  Diags.setClient(&DiagsConsumer, /*ShouldOwnClient=*/false);
  auto Invocation = std::make_shared<CompilerInvocation>();
  if (!CompilerInvocation::CreateFromArgs(*Invocation, InputArgs, Diags, Exec))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to create compiler invocation");

  Expected<llvm::cas::CASID> Root = scanAndUpdateCC1InlineWithTool(
      Tool, DiagsConsumer, VerboseOS, *Invocation, WorkingDirectory, DB);
  if (!Root)
    return Root;

  OutputArgs.resize(1);
  OutputArgs[0] = "-cc1";
  Invocation->generateCC1CommandLine(OutputArgs, SaveArg);
  return *Root;
}

static Expected<llvm::cas::CASID> scanAndUpdateCC1Inline(
    const char *Exec, ArrayRef<const char *> InputArgs,
    StringRef WorkingDirectory, SmallVectorImpl<const char *> &OutputArgs,
    bool ProduceIncludeTree, bool &DiagnosticErrorOccurred,
    llvm::function_ref<const char *(const Twine &)> SaveArg,
    const CASOptions &CASOpts, std::shared_ptr<llvm::cas::ObjectStore> DB,
    std::shared_ptr<llvm::cas::ActionCache> Cache) {
  IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> FS;
  if (!ProduceIncludeTree)
    FS = llvm::cantFail(llvm::cas::createCachingOnDiskFileSystem(*DB));

  tooling::dependencies::DependencyScanningService Service(
      tooling::dependencies::ScanningMode::DependencyDirectivesScan,
      ProduceIncludeTree
          ? tooling::dependencies::ScanningOutputFormat::IncludeTree
          : tooling::dependencies::ScanningOutputFormat::Tree,
      CASOpts, DB, Cache, FS,
      /*ReuseFileManager=*/false,
      /*SkipExcludedPPRanges=*/true);
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> UnderlyingFS =
      llvm::vfs::createPhysicalFileSystem();
  if (ProduceIncludeTree)
    UnderlyingFS =
        llvm::cas::createCASProvidingFileSystem(DB, std::move(UnderlyingFS));
  tooling::dependencies::DependencyScanningTool Tool(Service,
                                                     std::move(UnderlyingFS));

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts =
      CreateAndPopulateDiagOpts(InputArgs);
  auto DiagsConsumer = std::make_unique<TextDiagnosticPrinter>(
      llvm::errs(), DiagOpts.get(), false);

  auto Result = scanAndUpdateCC1InlineWithTool(
      Tool, *DiagsConsumer, /*VerboseOS*/ nullptr, Exec, InputArgs,
      WorkingDirectory, OutputArgs, *DB, SaveArg);
  DiagnosticErrorOccurred = DiagsConsumer->getNumErrors() != 0;
  return Result;
}
#endif /* LLVM_ON_UNIX */
