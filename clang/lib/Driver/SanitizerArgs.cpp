//===--- SanitizerArgs.cpp - Arguments for sanitizer tools  ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "clang/Driver/SanitizerArgs.h"
#include "Tools.h"
#include "clang/Basic/Sanitizers.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SpecialCaseList.h"
#include <memory>

using namespace clang;
using namespace clang::SanitizerKind;
using namespace clang::driver;
using namespace llvm::opt;

enum : SanitizerMask {
  NeedsUbsanRt = Undefined | Integer | CFI,
  NeedsUbsanCxxRt = Vptr | CFI,
  NotAllowedWithTrap = Vptr,
  RequiresPIE = DataFlow,
  NeedsUnwindTables = Address | Thread | Memory | DataFlow,
  SupportsCoverage = Address | Memory | Leak | Undefined | Integer | DataFlow,
  RecoverableByDefault = Undefined | Integer,
  Unrecoverable = Unreachable | Return,
  LegacyFsanitizeRecoverMask = Undefined | Integer,
  NeedsLTO = CFI,
  TrappingSupported =
      (Undefined & ~Vptr) | UnsignedIntegerOverflow | LocalBounds | CFI,
  TrappingDefault = CFI,
};

enum CoverageFeature {
  CoverageFunc = 1 << 0,
  CoverageBB = 1 << 1,
  CoverageEdge = 1 << 2,
  CoverageIndirCall = 1 << 3,
  CoverageTraceBB = 1 << 4,
  CoverageTraceCmp = 1 << 5,
  Coverage8bitCounters = 1 << 6,
};

/// Parse a -fsanitize= or -fno-sanitize= argument's values, diagnosing any
/// invalid components. Returns a SanitizerMask.
static SanitizerMask parseArgValues(const Driver &D, const llvm::opt::Arg *A,
                                    bool DiagnoseErrors);

/// Parse -f(no-)?sanitize-coverage= flag values, diagnosing any invalid
/// components. Returns OR of members of \c CoverageFeature enumeration.
static int parseCoverageFeatures(const Driver &D, const llvm::opt::Arg *A);

/// Produce an argument string from ArgList \p Args, which shows how it
/// provides some sanitizer kind from \p Mask. For example, the argument list
/// "-fsanitize=thread,vptr -fsanitize=address" with mask \c NeedsUbsanRt
/// would produce "-fsanitize=vptr".
static std::string lastArgumentForMask(const Driver &D,
                                       const llvm::opt::ArgList &Args,
                                       SanitizerMask Mask);

/// Produce an argument string from argument \p A, which shows how it provides
/// a value in \p Mask. For instance, the argument
/// "-fsanitize=address,alignment" with mask \c NeedsUbsanRt would produce
/// "-fsanitize=alignment".
static std::string describeSanitizeArg(const llvm::opt::Arg *A,
                                       SanitizerMask Mask);

/// Produce a string containing comma-separated names of sanitizers in \p
/// Sanitizers set.
static std::string toString(const clang::SanitizerSet &Sanitizers);

static bool getDefaultBlacklist(const Driver &D, SanitizerMask Kinds,
                                std::string &BLPath) {
  const char *BlacklistFile = nullptr;
  if (Kinds & Address)
    BlacklistFile = "asan_blacklist.txt";
  else if (Kinds & Memory)
    BlacklistFile = "msan_blacklist.txt";
  else if (Kinds & Thread)
    BlacklistFile = "tsan_blacklist.txt";
  else if (Kinds & DataFlow)
    BlacklistFile = "dfsan_abilist.txt";
  else if (Kinds & CFI)
    BlacklistFile = "cfi_blacklist.txt";

  if (BlacklistFile) {
    clang::SmallString<64> Path(D.ResourceDir);
    llvm::sys::path::append(Path, BlacklistFile);
    BLPath = Path.str();
    return true;
  }
  return false;
}

/// Sets group bits for every group that has at least one representative already
/// enabled in \p Kinds.
static SanitizerMask setGroupBits(SanitizerMask Kinds) {
#define SANITIZER(NAME, ID)
#define SANITIZER_GROUP(NAME, ID, ALIAS)                                       \
  if (Kinds & SanitizerKind::ID)                                               \
    Kinds |= SanitizerKind::ID##Group;
#include "clang/Basic/Sanitizers.def"
  return Kinds;
}

static SanitizerMask parseSanitizeTrapArgs(const Driver &D,
                                           const llvm::opt::ArgList &Args) {
  SanitizerMask TrapRemove = 0; // During the loop below, the accumulated set of
                                // sanitizers disabled by the current sanitizer
                                // argument or any argument after it.
  SanitizerMask TrappingKinds = 0;
  SanitizerMask TrappingSupportedWithGroups = setGroupBits(TrappingSupported);

  for (ArgList::const_reverse_iterator I = Args.rbegin(), E = Args.rend();
       I != E; ++I) {
    const auto *Arg = *I;
    if (Arg->getOption().matches(options::OPT_fsanitize_trap_EQ)) {
      Arg->claim();
      SanitizerMask Add = parseArgValues(D, Arg, true);
      Add &= ~TrapRemove;
      if (SanitizerMask InvalidValues = Add & ~TrappingSupportedWithGroups) {
        SanitizerSet S;
        S.Mask = InvalidValues;
        D.Diag(diag::err_drv_unsupported_option_argument) << "-fsanitize-trap"
                                                          << toString(S);
      }
      TrappingKinds |= expandSanitizerGroups(Add) & ~TrapRemove;
    } else if (Arg->getOption().matches(options::OPT_fno_sanitize_trap_EQ)) {
      Arg->claim();
      TrapRemove |= expandSanitizerGroups(parseArgValues(D, Arg, true));
    } else if (Arg->getOption().matches(
                   options::OPT_fsanitize_undefined_trap_on_error)) {
      Arg->claim();
      TrappingKinds |=
          expandSanitizerGroups(UndefinedGroup & ~TrapRemove) & ~TrapRemove;
    } else if (Arg->getOption().matches(
                   options::OPT_fno_sanitize_undefined_trap_on_error)) {
      Arg->claim();
      TrapRemove |= expandSanitizerGroups(UndefinedGroup);
    }
  }

  // Apply default trapping behavior.
  TrappingKinds |= TrappingDefault & ~TrapRemove;

  return TrappingKinds;
}

bool SanitizerArgs::needsUbsanRt() const {
  return (Sanitizers.Mask & NeedsUbsanRt & ~TrapSanitizers.Mask) &&
         !Sanitizers.has(Address) &&
         !Sanitizers.has(Memory) &&
         !Sanitizers.has(Thread);
}

bool SanitizerArgs::requiresPIE() const {
  return NeedPIE || (Sanitizers.Mask & RequiresPIE);
}

bool SanitizerArgs::needsUnwindTables() const {
  return Sanitizers.Mask & NeedsUnwindTables;
}

void SanitizerArgs::clear() {
  Sanitizers.clear();
  RecoverableSanitizers.clear();
  TrapSanitizers.clear();
  BlacklistFiles.clear();
  ExtraDeps.clear();
  CoverageFeatures = 0;
  MsanTrackOrigins = 0;
  MsanUseAfterDtor = false;
  NeedPIE = false;
  AsanFieldPadding = 0;
  AsanSharedRuntime = false;
  LinkCXXRuntimes = false;
}

SanitizerArgs::SanitizerArgs(const ToolChain &TC,
                             const llvm::opt::ArgList &Args) {
  clear();
  SanitizerMask AllRemove = 0;  // During the loop below, the accumulated set of
                                // sanitizers disabled by the current sanitizer
                                // argument or any argument after it.
  SanitizerMask AllAddedKinds = 0;  // Mask of all sanitizers ever enabled by
                                    // -fsanitize= flags (directly or via group
                                    // expansion), some of which may be disabled
                                    // later. Used to carefully prune
                                    // unused-argument diagnostics.
  SanitizerMask DiagnosedKinds = 0;  // All Kinds we have diagnosed up to now.
                                     // Used to deduplicate diagnostics.
  SanitizerMask Kinds = 0;
  const SanitizerMask Supported = setGroupBits(TC.getSupportedSanitizers());
  ToolChain::RTTIMode RTTIMode = TC.getRTTIMode();

  const Driver &D = TC.getDriver();
  SanitizerMask TrappingKinds = parseSanitizeTrapArgs(D, Args);
  SanitizerMask InvalidTrappingKinds = TrappingKinds & NotAllowedWithTrap;

  for (ArgList::const_reverse_iterator I = Args.rbegin(), E = Args.rend();
       I != E; ++I) {
    const auto *Arg = *I;
    if (Arg->getOption().matches(options::OPT_fsanitize_EQ)) {
      Arg->claim();
      SanitizerMask Add = parseArgValues(D, Arg, true);
      AllAddedKinds |= expandSanitizerGroups(Add);

      // Avoid diagnosing any sanitizer which is disabled later.
      Add &= ~AllRemove;
      // At this point we have not expanded groups, so any unsupported
      // sanitizers in Add are those which have been explicitly enabled.
      // Diagnose them.
      if (SanitizerMask KindsToDiagnose =
              Add & InvalidTrappingKinds & ~DiagnosedKinds) {
        std::string Desc = describeSanitizeArg(*I, KindsToDiagnose);
        D.Diag(diag::err_drv_argument_not_allowed_with)
            << Desc << "-fsanitize-trap=undefined";
        DiagnosedKinds |= KindsToDiagnose;
      }
      Add &= ~InvalidTrappingKinds;
      if (SanitizerMask KindsToDiagnose = Add & ~Supported & ~DiagnosedKinds) {
        std::string Desc = describeSanitizeArg(*I, KindsToDiagnose);
        D.Diag(diag::err_drv_unsupported_opt_for_target)
            << Desc << TC.getTriple().str();
        DiagnosedKinds |= KindsToDiagnose;
      }
      Add &= Supported;

      // Test for -fno-rtti + explicit -fsanitizer=vptr before expanding groups
      // so we don't error out if -fno-rtti and -fsanitize=undefined were
      // passed.
      if (Add & Vptr &&
          (RTTIMode == ToolChain::RM_DisabledImplicitly ||
           RTTIMode == ToolChain::RM_DisabledExplicitly)) {
        if (RTTIMode == ToolChain::RM_DisabledImplicitly)
          // Warn about not having rtti enabled if the vptr sanitizer is
          // explicitly enabled
          D.Diag(diag::warn_drv_disabling_vptr_no_rtti_default);
        else {
          const llvm::opt::Arg *NoRTTIArg = TC.getRTTIArg();
          assert(NoRTTIArg &&
                 "RTTI disabled explicitly but we have no argument!");
          D.Diag(diag::err_drv_argument_not_allowed_with)
              << "-fsanitize=vptr" << NoRTTIArg->getAsString(Args);
        }

        // Take out the Vptr sanitizer from the enabled sanitizers
        AllRemove |= Vptr;
      }

      Add = expandSanitizerGroups(Add);
      // Group expansion may have enabled a sanitizer which is disabled later.
      Add &= ~AllRemove;
      // Silently discard any unsupported sanitizers implicitly enabled through
      // group expansion.
      Add &= ~InvalidTrappingKinds;
      Add &= Supported;

      Kinds |= Add;
    } else if (Arg->getOption().matches(options::OPT_fno_sanitize_EQ)) {
      Arg->claim();
      SanitizerMask Remove = parseArgValues(D, Arg, true);
      AllRemove |= expandSanitizerGroups(Remove);
    }
  }

  // We disable the vptr sanitizer if it was enabled by group expansion but RTTI
  // is disabled.
  if ((Kinds & Vptr) &&
      (RTTIMode == ToolChain::RM_DisabledImplicitly ||
       RTTIMode == ToolChain::RM_DisabledExplicitly)) {
    Kinds &= ~Vptr;
  }

  // Check that LTO is enabled if we need it.
  if ((Kinds & NeedsLTO) && !D.isUsingLTO()) {
    D.Diag(diag::err_drv_argument_only_allowed_with)
        << lastArgumentForMask(D, Args, Kinds & NeedsLTO) << "-flto";
  }

  // Report error if there are non-trapping sanitizers that require
  // c++abi-specific  parts of UBSan runtime, and they are not provided by the
  // toolchain. We don't have a good way to check the latter, so we just
  // check if the toolchan supports vptr.
  if (~Supported & Vptr) {
    SanitizerMask KindsToDiagnose = Kinds & ~TrappingKinds & NeedsUbsanCxxRt;
    // The runtime library supports the Microsoft C++ ABI, but only well enough
    // for CFI. FIXME: Remove this once we support vptr on Windows.
    if (TC.getTriple().isOSWindows())
      KindsToDiagnose &= ~CFI;
    if (KindsToDiagnose) {
      SanitizerSet S;
      S.Mask = KindsToDiagnose;
      D.Diag(diag::err_drv_unsupported_opt_for_target)
          << ("-fno-sanitize-trap=" + toString(S)) << TC.getTriple().str();
      Kinds &= ~KindsToDiagnose;
    }
  }

  // Warn about incompatible groups of sanitizers.
  std::pair<SanitizerMask, SanitizerMask> IncompatibleGroups[] = {
      std::make_pair(Address, Thread), std::make_pair(Address, Memory),
      std::make_pair(Thread, Memory), std::make_pair(Leak, Thread),
      std::make_pair(Leak, Memory), std::make_pair(KernelAddress, Address),
      std::make_pair(KernelAddress, Leak),
      std::make_pair(KernelAddress, Thread),
      std::make_pair(KernelAddress, Memory)};
  for (auto G : IncompatibleGroups) {
    SanitizerMask Group = G.first;
    if (Kinds & Group) {
      if (SanitizerMask Incompatible = Kinds & G.second) {
        D.Diag(clang::diag::err_drv_argument_not_allowed_with)
            << lastArgumentForMask(D, Args, Group)
            << lastArgumentForMask(D, Args, Incompatible);
        Kinds &= ~Incompatible;
      }
    }
  }
  // FIXME: Currently -fsanitize=leak is silently ignored in the presence of
  // -fsanitize=address. Perhaps it should print an error, or perhaps
  // -f(-no)sanitize=leak should change whether leak detection is enabled by
  // default in ASan?

  // Parse -f(no-)?sanitize-recover flags.
  SanitizerMask RecoverableKinds = RecoverableByDefault;
  SanitizerMask DiagnosedUnrecoverableKinds = 0;
  for (const auto *Arg : Args) {
    const char *DeprecatedReplacement = nullptr;
    if (Arg->getOption().matches(options::OPT_fsanitize_recover)) {
      DeprecatedReplacement = "-fsanitize-recover=undefined,integer";
      RecoverableKinds |= expandSanitizerGroups(LegacyFsanitizeRecoverMask);
      Arg->claim();
    } else if (Arg->getOption().matches(options::OPT_fno_sanitize_recover)) {
      DeprecatedReplacement = "-fno-sanitize-recover=undefined,integer";
      RecoverableKinds &= ~expandSanitizerGroups(LegacyFsanitizeRecoverMask);
      Arg->claim();
    } else if (Arg->getOption().matches(options::OPT_fsanitize_recover_EQ)) {
      SanitizerMask Add = parseArgValues(D, Arg, true);
      // Report error if user explicitly tries to recover from unrecoverable
      // sanitizer.
      if (SanitizerMask KindsToDiagnose =
              Add & Unrecoverable & ~DiagnosedUnrecoverableKinds) {
        SanitizerSet SetToDiagnose;
        SetToDiagnose.Mask |= KindsToDiagnose;
        D.Diag(diag::err_drv_unsupported_option_argument)
            << Arg->getOption().getName() << toString(SetToDiagnose);
        DiagnosedUnrecoverableKinds |= KindsToDiagnose;
      }
      RecoverableKinds |= expandSanitizerGroups(Add);
      Arg->claim();
    } else if (Arg->getOption().matches(options::OPT_fno_sanitize_recover_EQ)) {
      RecoverableKinds &= ~expandSanitizerGroups(parseArgValues(D, Arg, true));
      Arg->claim();
    }
    if (DeprecatedReplacement) {
      D.Diag(diag::warn_drv_deprecated_arg) << Arg->getAsString(Args)
                                            << DeprecatedReplacement;
    }
  }
  RecoverableKinds &= Kinds;
  RecoverableKinds &= ~Unrecoverable;

  TrappingKinds &= Kinds;

  // Setup blacklist files.
  // Add default blacklist from resource directory.
  {
    std::string BLPath;
    if (getDefaultBlacklist(D, Kinds, BLPath) && llvm::sys::fs::exists(BLPath))
      BlacklistFiles.push_back(BLPath);
  }
  // Parse -f(no-)sanitize-blacklist options.
  for (const auto *Arg : Args) {
    if (Arg->getOption().matches(options::OPT_fsanitize_blacklist)) {
      Arg->claim();
      std::string BLPath = Arg->getValue();
      if (llvm::sys::fs::exists(BLPath)) {
        BlacklistFiles.push_back(BLPath);
        ExtraDeps.push_back(BLPath);
      } else
        D.Diag(clang::diag::err_drv_no_such_file) << BLPath;

    } else if (Arg->getOption().matches(options::OPT_fno_sanitize_blacklist)) {
      Arg->claim();
      BlacklistFiles.clear();
      ExtraDeps.clear();
    }
  }
  // Validate blacklists format.
  {
    std::string BLError;
    std::unique_ptr<llvm::SpecialCaseList> SCL(
        llvm::SpecialCaseList::create(BlacklistFiles, BLError));
    if (!SCL.get())
      D.Diag(clang::diag::err_drv_malformed_sanitizer_blacklist) << BLError;
  }

  // Parse -f[no-]sanitize-memory-track-origins[=level] options.
  if (AllAddedKinds & Memory) {
    if (Arg *A =
            Args.getLastArg(options::OPT_fsanitize_memory_track_origins_EQ,
                            options::OPT_fsanitize_memory_track_origins,
                            options::OPT_fno_sanitize_memory_track_origins)) {
      if (A->getOption().matches(options::OPT_fsanitize_memory_track_origins)) {
        MsanTrackOrigins = 2;
      } else if (A->getOption().matches(
                     options::OPT_fno_sanitize_memory_track_origins)) {
        MsanTrackOrigins = 0;
      } else {
        StringRef S = A->getValue();
        if (S.getAsInteger(0, MsanTrackOrigins) || MsanTrackOrigins < 0 ||
            MsanTrackOrigins > 2) {
          D.Diag(clang::diag::err_drv_invalid_value) << A->getAsString(Args) << S;
        }
      }
    }
    MsanUseAfterDtor =
        Args.hasArg(options::OPT_fsanitize_memory_use_after_dtor);
    NeedPIE |= !(TC.getTriple().isOSLinux() &&
                 TC.getTriple().getArch() == llvm::Triple::x86_64);
  }

  // Parse -f(no-)?sanitize-coverage flags if coverage is supported by the
  // enabled sanitizers.
  if (AllAddedKinds & SupportsCoverage) {
    for (const auto *Arg : Args) {
      if (Arg->getOption().matches(options::OPT_fsanitize_coverage)) {
        Arg->claim();
        int LegacySanitizeCoverage;
        if (Arg->getNumValues() == 1 &&
            !StringRef(Arg->getValue(0))
                 .getAsInteger(0, LegacySanitizeCoverage) &&
            LegacySanitizeCoverage >= 0 && LegacySanitizeCoverage <= 4) {
          // TODO: Add deprecation notice for this form.
          switch (LegacySanitizeCoverage) {
          case 0:
            CoverageFeatures = 0;
            break;
          case 1:
            CoverageFeatures = CoverageFunc;
            break;
          case 2:
            CoverageFeatures = CoverageBB;
            break;
          case 3:
            CoverageFeatures = CoverageEdge;
            break;
          case 4:
            CoverageFeatures = CoverageEdge | CoverageIndirCall;
            break;
          }
          continue;
        }
        CoverageFeatures |= parseCoverageFeatures(D, Arg);
      } else if (Arg->getOption().matches(options::OPT_fno_sanitize_coverage)) {
        Arg->claim();
        CoverageFeatures &= ~parseCoverageFeatures(D, Arg);
      }
    }
  }
  // Choose at most one coverage type: function, bb, or edge.
  if ((CoverageFeatures & CoverageFunc) && (CoverageFeatures & CoverageBB))
    D.Diag(clang::diag::err_drv_argument_not_allowed_with)
        << "-fsanitize-coverage=func"
        << "-fsanitize-coverage=bb";
  if ((CoverageFeatures & CoverageFunc) && (CoverageFeatures & CoverageEdge))
    D.Diag(clang::diag::err_drv_argument_not_allowed_with)
        << "-fsanitize-coverage=func"
        << "-fsanitize-coverage=edge";
  if ((CoverageFeatures & CoverageBB) && (CoverageFeatures & CoverageEdge))
    D.Diag(clang::diag::err_drv_argument_not_allowed_with)
        << "-fsanitize-coverage=bb"
        << "-fsanitize-coverage=edge";
  // Basic block tracing and 8-bit counters require some type of coverage
  // enabled.
  int CoverageTypes = CoverageFunc | CoverageBB | CoverageEdge;
  if ((CoverageFeatures & CoverageTraceBB) &&
      !(CoverageFeatures & CoverageTypes))
    D.Diag(clang::diag::err_drv_argument_only_allowed_with)
        << "-fsanitize-coverage=trace-bb"
        << "-fsanitize-coverage=(func|bb|edge)";
  if ((CoverageFeatures & Coverage8bitCounters) &&
      !(CoverageFeatures & CoverageTypes))
    D.Diag(clang::diag::err_drv_argument_only_allowed_with)
        << "-fsanitize-coverage=8bit-counters"
        << "-fsanitize-coverage=(func|bb|edge)";

  if (AllAddedKinds & Address) {
    AsanSharedRuntime =
        Args.hasArg(options::OPT_shared_libasan) || TC.getTriple().isAndroid();
    NeedPIE |= TC.getTriple().isAndroid();
    if (Arg *A =
            Args.getLastArg(options::OPT_fsanitize_address_field_padding)) {
        StringRef S = A->getValue();
        // Legal values are 0 and 1, 2, but in future we may add more levels.
        if (S.getAsInteger(0, AsanFieldPadding) || AsanFieldPadding < 0 ||
            AsanFieldPadding > 2) {
          D.Diag(clang::diag::err_drv_invalid_value) << A->getAsString(Args) << S;
        }
    }

    if (Arg *WindowsDebugRTArg =
            Args.getLastArg(options::OPT__SLASH_MTd, options::OPT__SLASH_MT,
                            options::OPT__SLASH_MDd, options::OPT__SLASH_MD,
                            options::OPT__SLASH_LDd, options::OPT__SLASH_LD)) {
      switch (WindowsDebugRTArg->getOption().getID()) {
      case options::OPT__SLASH_MTd:
      case options::OPT__SLASH_MDd:
      case options::OPT__SLASH_LDd:
        D.Diag(clang::diag::err_drv_argument_not_allowed_with)
            << WindowsDebugRTArg->getAsString(Args)
            << lastArgumentForMask(D, Args, Address);
        D.Diag(clang::diag::note_drv_address_sanitizer_debug_runtime);
      }
    }
  }

  // Parse -link-cxx-sanitizer flag.
  LinkCXXRuntimes =
      Args.hasArg(options::OPT_fsanitize_link_cxx_runtime) || D.CCCIsCXX();

  // Finally, initialize the set of available and recoverable sanitizers.
  Sanitizers.Mask |= Kinds;
  RecoverableSanitizers.Mask |= RecoverableKinds;
  TrapSanitizers.Mask |= TrappingKinds;
}

static std::string toString(const clang::SanitizerSet &Sanitizers) {
  std::string Res;
#define SANITIZER(NAME, ID)                                                    \
  if (Sanitizers.has(ID)) {                                                    \
    if (!Res.empty())                                                          \
      Res += ",";                                                              \
    Res += NAME;                                                               \
  }
#include "clang/Basic/Sanitizers.def"
  return Res;
}

void SanitizerArgs::addArgs(const ToolChain &TC, const llvm::opt::ArgList &Args,
                            llvm::opt::ArgStringList &CmdArgs,
                            types::ID InputType) const {
  if (Sanitizers.empty())
    return;
  CmdArgs.push_back(Args.MakeArgString("-fsanitize=" + toString(Sanitizers)));

  if (!RecoverableSanitizers.empty())
    CmdArgs.push_back(Args.MakeArgString("-fsanitize-recover=" +
                                         toString(RecoverableSanitizers)));

  if (!TrapSanitizers.empty())
    CmdArgs.push_back(
        Args.MakeArgString("-fsanitize-trap=" + toString(TrapSanitizers)));

  for (const auto &BLPath : BlacklistFiles) {
    SmallString<64> BlacklistOpt("-fsanitize-blacklist=");
    BlacklistOpt += BLPath;
    CmdArgs.push_back(Args.MakeArgString(BlacklistOpt));
  }
  for (const auto &Dep : ExtraDeps) {
    SmallString<64> ExtraDepOpt("-fdepfile-entry=");
    ExtraDepOpt += Dep;
    CmdArgs.push_back(Args.MakeArgString(ExtraDepOpt));
  }

  if (MsanTrackOrigins)
    CmdArgs.push_back(Args.MakeArgString("-fsanitize-memory-track-origins=" +
                                         llvm::utostr(MsanTrackOrigins)));

  if (MsanUseAfterDtor)
    CmdArgs.push_back(Args.MakeArgString("-fsanitize-memory-use-after-dtor"));

  if (AsanFieldPadding)
    CmdArgs.push_back(Args.MakeArgString("-fsanitize-address-field-padding=" +
                                         llvm::utostr(AsanFieldPadding)));
  // Translate available CoverageFeatures to corresponding clang-cc1 flags.
  std::pair<int, const char *> CoverageFlags[] = {
    std::make_pair(CoverageFunc, "-fsanitize-coverage-type=1"),
    std::make_pair(CoverageBB, "-fsanitize-coverage-type=2"),
    std::make_pair(CoverageEdge, "-fsanitize-coverage-type=3"),
    std::make_pair(CoverageIndirCall, "-fsanitize-coverage-indirect-calls"),
    std::make_pair(CoverageTraceBB, "-fsanitize-coverage-trace-bb"),
    std::make_pair(CoverageTraceCmp, "-fsanitize-coverage-trace-cmp"),
    std::make_pair(Coverage8bitCounters, "-fsanitize-coverage-8bit-counters")};
  for (auto F : CoverageFlags) {
    if (CoverageFeatures & F.first)
      CmdArgs.push_back(Args.MakeArgString(F.second));
  }


  // MSan: Workaround for PR16386.
  // ASan: This is mainly to help LSan with cases such as
  // https://code.google.com/p/address-sanitizer/issues/detail?id=373
  // We can't make this conditional on -fsanitize=leak, as that flag shouldn't
  // affect compilation.
  if (Sanitizers.has(Memory) || Sanitizers.has(Address))
    CmdArgs.push_back(Args.MakeArgString("-fno-assume-sane-operator-new"));

  if (TC.getTriple().isOSWindows() && needsUbsanRt()) {
    // Instruct the code generator to embed linker directives in the object file
    // that cause the required runtime libraries to be linked.
    CmdArgs.push_back(Args.MakeArgString(
        "--dependent-lib=" + TC.getCompilerRT(Args, "ubsan_standalone")));
    if (types::isCXX(InputType))
      CmdArgs.push_back(Args.MakeArgString(
          "--dependent-lib=" + TC.getCompilerRT(Args, "ubsan_standalone_cxx")));
  }
}

SanitizerMask parseArgValues(const Driver &D, const llvm::opt::Arg *A,
                             bool DiagnoseErrors) {
  assert((A->getOption().matches(options::OPT_fsanitize_EQ) ||
          A->getOption().matches(options::OPT_fno_sanitize_EQ) ||
          A->getOption().matches(options::OPT_fsanitize_recover_EQ) ||
          A->getOption().matches(options::OPT_fno_sanitize_recover_EQ) ||
          A->getOption().matches(options::OPT_fsanitize_trap_EQ) ||
          A->getOption().matches(options::OPT_fno_sanitize_trap_EQ)) &&
         "Invalid argument in parseArgValues!");
  SanitizerMask Kinds = 0;
  for (int i = 0, n = A->getNumValues(); i != n; ++i) {
    const char *Value = A->getValue(i);
    SanitizerMask Kind;
    // Special case: don't accept -fsanitize=all.
    if (A->getOption().matches(options::OPT_fsanitize_EQ) &&
        0 == strcmp("all", Value))
      Kind = 0;
    else
      Kind = parseSanitizerValue(Value, /*AllowGroups=*/true);

    if (Kind)
      Kinds |= Kind;
    else if (DiagnoseErrors)
      D.Diag(clang::diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << Value;
  }
  return Kinds;
}

int parseCoverageFeatures(const Driver &D, const llvm::opt::Arg *A) {
  assert(A->getOption().matches(options::OPT_fsanitize_coverage) ||
         A->getOption().matches(options::OPT_fno_sanitize_coverage));
  int Features = 0;
  for (int i = 0, n = A->getNumValues(); i != n; ++i) {
    const char *Value = A->getValue(i);
    int F = llvm::StringSwitch<int>(Value)
        .Case("func", CoverageFunc)
        .Case("bb", CoverageBB)
        .Case("edge", CoverageEdge)
        .Case("indirect-calls", CoverageIndirCall)
        .Case("trace-bb", CoverageTraceBB)
        .Case("trace-cmp", CoverageTraceCmp)
        .Case("8bit-counters", Coverage8bitCounters)
        .Default(0);
    if (F == 0)
      D.Diag(clang::diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << Value;
    Features |= F;
  }
  return Features;
}

std::string lastArgumentForMask(const Driver &D, const llvm::opt::ArgList &Args,
                                SanitizerMask Mask) {
  for (llvm::opt::ArgList::const_reverse_iterator I = Args.rbegin(),
                                                  E = Args.rend();
       I != E; ++I) {
    const auto *Arg = *I;
    if (Arg->getOption().matches(options::OPT_fsanitize_EQ)) {
      SanitizerMask AddKinds =
          expandSanitizerGroups(parseArgValues(D, Arg, false));
      if (AddKinds & Mask)
        return describeSanitizeArg(Arg, Mask);
    } else if (Arg->getOption().matches(options::OPT_fno_sanitize_EQ)) {
      SanitizerMask RemoveKinds =
          expandSanitizerGroups(parseArgValues(D, Arg, false));
      Mask &= ~RemoveKinds;
    }
  }
  llvm_unreachable("arg list didn't provide expected value");
}

std::string describeSanitizeArg(const llvm::opt::Arg *A, SanitizerMask Mask) {
  assert(A->getOption().matches(options::OPT_fsanitize_EQ)
         && "Invalid argument in describeSanitizerArg!");

  std::string Sanitizers;
  for (int i = 0, n = A->getNumValues(); i != n; ++i) {
    if (expandSanitizerGroups(
            parseSanitizerValue(A->getValue(i), /*AllowGroups=*/true)) &
        Mask) {
      if (!Sanitizers.empty())
        Sanitizers += ",";
      Sanitizers += A->getValue(i);
    }
  }

  assert(!Sanitizers.empty() && "arg didn't provide expected value");
  return "-fsanitize=" + Sanitizers;
}
