//===- CompilerInvocation.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInvocation.h"
#include "TestModuleFileExtension.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/CommentOptions.h"
#include "clang/Basic/DebugInfoOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/ObjCRuntime.h"
#include "clang/Basic/Sanitizers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/Version.h"
#include "clang/Basic/Visibility.h"
#include "clang/Basic/XRayInstr.h"
#include "clang/Config/config.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CommandLineSourceLoc.h"
#include "clang/Frontend/DependencyOutputOptions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/MigratorOptions.h"
#include "clang/Frontend/PreprocessorOutputOptions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/CodeCompleteOptions.h"
#include "clang/Serialization/ASTBitCodes.h"
#include "clang/Serialization/ModuleFileExtension.h"
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptSpecifier.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Remarks/HotnessThresholdParser.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using namespace clang;
using namespace driver;
using namespace options;
using namespace llvm::opt;

//===----------------------------------------------------------------------===//
// Initialization.
//===----------------------------------------------------------------------===//

CompilerInvocationBase::CompilerInvocationBase()
    : LangOpts(new LangOptions()), TargetOpts(new TargetOptions()),
      DiagnosticOpts(new DiagnosticOptions()),
      HeaderSearchOpts(new HeaderSearchOptions()),
      PreprocessorOpts(new PreprocessorOptions()) {}

CompilerInvocationBase::CompilerInvocationBase(const CompilerInvocationBase &X)
    : LangOpts(new LangOptions(*X.getLangOpts())),
      TargetOpts(new TargetOptions(X.getTargetOpts())),
      DiagnosticOpts(new DiagnosticOptions(X.getDiagnosticOpts())),
      HeaderSearchOpts(new HeaderSearchOptions(X.getHeaderSearchOpts())),
      PreprocessorOpts(new PreprocessorOptions(X.getPreprocessorOpts())) {}

CompilerInvocationBase::~CompilerInvocationBase() = default;

//===----------------------------------------------------------------------===//
// Normalizers
//===----------------------------------------------------------------------===//

#define SIMPLE_ENUM_VALUE_TABLE
#include "clang/Driver/Options.inc"
#undef SIMPLE_ENUM_VALUE_TABLE

static llvm::Optional<bool>
normalizeSimpleFlag(OptSpecifier Opt, unsigned TableIndex, const ArgList &Args,
                    DiagnosticsEngine &Diags, bool &Success) {
  if (Args.hasArg(Opt))
    return true;
  return None;
}

static Optional<bool> normalizeSimpleNegativeFlag(OptSpecifier Opt, unsigned,
                                                  const ArgList &Args,
                                                  DiagnosticsEngine &,
                                                  bool &Success) {
  if (Args.hasArg(Opt))
    return false;
  return None;
}

/// The tblgen-erated code passes in a fifth parameter of an arbitrary type, but
/// denormalizeSimpleFlags never looks at it. Avoid bloating compile-time with
/// unnecessary template instantiations and just ignore it with a variadic
/// argument.
static void denormalizeSimpleFlag(SmallVectorImpl<const char *> &Args,
                                  const char *Spelling,
                                  CompilerInvocation::StringAllocator,
                                  Option::OptionClass, unsigned, /*T*/...) {
  Args.push_back(Spelling);
}

template <typename T> static constexpr bool is_uint64_t_convertible() {
  return !std::is_same<T, uint64_t>::value &&
         llvm::is_integral_or_enum<T>::value;
}

template <typename T,
          std::enable_if_t<!is_uint64_t_convertible<T>(), bool> = false>
static auto makeFlagToValueNormalizer(T Value) {
  return [Value](OptSpecifier Opt, unsigned, const ArgList &Args,
                 DiagnosticsEngine &, bool &Success) -> Optional<T> {
    if (Args.hasArg(Opt))
      return Value;
    return None;
  };
}

template <typename T,
          std::enable_if_t<is_uint64_t_convertible<T>(), bool> = false>
static auto makeFlagToValueNormalizer(T Value) {
  return makeFlagToValueNormalizer(uint64_t(Value));
}

static auto makeBooleanOptionNormalizer(bool Value, bool OtherValue,
                                        OptSpecifier OtherOpt) {
  return [Value, OtherValue, OtherOpt](OptSpecifier Opt, unsigned,
                                       const ArgList &Args, DiagnosticsEngine &,
                                       bool &Success) -> Optional<bool> {
    if (const Arg *A = Args.getLastArg(Opt, OtherOpt)) {
      return A->getOption().matches(Opt) ? Value : OtherValue;
    }
    return None;
  };
}

static auto makeBooleanOptionDenormalizer(bool Value) {
  return [Value](SmallVectorImpl<const char *> &Args, const char *Spelling,
                 CompilerInvocation::StringAllocator, Option::OptionClass,
                 unsigned, bool KeyPath) {
    if (KeyPath == Value)
      Args.push_back(Spelling);
  };
}

static void denormalizeStringImpl(SmallVectorImpl<const char *> &Args,
                                  const char *Spelling,
                                  CompilerInvocation::StringAllocator SA,
                                  Option::OptionClass OptClass, unsigned,
                                  Twine Value) {
  switch (OptClass) {
  case Option::SeparateClass:
  case Option::JoinedOrSeparateClass:
    Args.push_back(Spelling);
    Args.push_back(SA(Value));
    break;
  case Option::JoinedClass:
    Args.push_back(SA(Twine(Spelling) + Value));
    break;
  default:
    llvm_unreachable("Cannot denormalize an option with option class "
                     "incompatible with string denormalization.");
  }
}

template <typename T>
static void
denormalizeString(SmallVectorImpl<const char *> &Args, const char *Spelling,
                  CompilerInvocation::StringAllocator SA,
                  Option::OptionClass OptClass, unsigned TableIndex, T Value) {
  denormalizeStringImpl(Args, Spelling, SA, OptClass, TableIndex, Twine(Value));
}

static Optional<SimpleEnumValue>
findValueTableByName(const SimpleEnumValueTable &Table, StringRef Name) {
  for (int I = 0, E = Table.Size; I != E; ++I)
    if (Name == Table.Table[I].Name)
      return Table.Table[I];

  return None;
}

static Optional<SimpleEnumValue>
findValueTableByValue(const SimpleEnumValueTable &Table, unsigned Value) {
  for (int I = 0, E = Table.Size; I != E; ++I)
    if (Value == Table.Table[I].Value)
      return Table.Table[I];

  return None;
}

static llvm::Optional<unsigned>
normalizeSimpleEnum(OptSpecifier Opt, unsigned TableIndex, const ArgList &Args,
                    DiagnosticsEngine &Diags, bool &Success) {
  assert(TableIndex < SimpleEnumValueTablesSize);
  const SimpleEnumValueTable &Table = SimpleEnumValueTables[TableIndex];

  auto *Arg = Args.getLastArg(Opt);
  if (!Arg)
    return None;

  StringRef ArgValue = Arg->getValue();
  if (auto MaybeEnumVal = findValueTableByName(Table, ArgValue))
    return MaybeEnumVal->Value;

  Success = false;
  Diags.Report(diag::err_drv_invalid_value)
      << Arg->getAsString(Args) << ArgValue;
  return None;
}

static void denormalizeSimpleEnumImpl(SmallVectorImpl<const char *> &Args,
                                      const char *Spelling,
                                      CompilerInvocation::StringAllocator SA,
                                      Option::OptionClass OptClass,
                                      unsigned TableIndex, unsigned Value) {
  assert(TableIndex < SimpleEnumValueTablesSize);
  const SimpleEnumValueTable &Table = SimpleEnumValueTables[TableIndex];
  if (auto MaybeEnumVal = findValueTableByValue(Table, Value)) {
    denormalizeString(Args, Spelling, SA, OptClass, TableIndex,
                      MaybeEnumVal->Name);
  } else {
    llvm_unreachable("The simple enum value was not correctly defined in "
                     "the tablegen option description");
  }
}

template <typename T>
static void denormalizeSimpleEnum(SmallVectorImpl<const char *> &Args,
                                  const char *Spelling,
                                  CompilerInvocation::StringAllocator SA,
                                  Option::OptionClass OptClass,
                                  unsigned TableIndex, T Value) {
  return denormalizeSimpleEnumImpl(Args, Spelling, SA, OptClass, TableIndex,
                                   static_cast<unsigned>(Value));
}

static Optional<std::string> normalizeString(OptSpecifier Opt, int TableIndex,
                                             const ArgList &Args,
                                             DiagnosticsEngine &Diags,
                                             bool &Success) {
  auto *Arg = Args.getLastArg(Opt);
  if (!Arg)
    return None;
  return std::string(Arg->getValue());
}

template <typename IntTy>
static Optional<IntTy>
normalizeStringIntegral(OptSpecifier Opt, int, const ArgList &Args,
                        DiagnosticsEngine &Diags, bool &Success) {
  auto *Arg = Args.getLastArg(Opt);
  if (!Arg)
    return None;
  IntTy Res;
  if (StringRef(Arg->getValue()).getAsInteger(0, Res)) {
    Success = false;
    Diags.Report(diag::err_drv_invalid_int_value)
        << Arg->getAsString(Args) << Arg->getValue();
    return None;
  }
  return Res;
}

static Optional<std::vector<std::string>>
normalizeStringVector(OptSpecifier Opt, int, const ArgList &Args,
                      DiagnosticsEngine &, bool &Success) {
  return Args.getAllArgValues(Opt);
}

static void denormalizeStringVector(SmallVectorImpl<const char *> &Args,
                                    const char *Spelling,
                                    CompilerInvocation::StringAllocator SA,
                                    Option::OptionClass OptClass,
                                    unsigned TableIndex,
                                    const std::vector<std::string> &Values) {
  switch (OptClass) {
  case Option::CommaJoinedClass: {
    std::string CommaJoinedValue;
    if (!Values.empty()) {
      CommaJoinedValue.append(Values.front());
      for (const std::string &Value : llvm::drop_begin(Values, 1)) {
        CommaJoinedValue.append(",");
        CommaJoinedValue.append(Value);
      }
    }
    denormalizeString(Args, Spelling, SA, Option::OptionClass::JoinedClass,
                      TableIndex, CommaJoinedValue);
    break;
  }
  case Option::JoinedClass:
  case Option::SeparateClass:
  case Option::JoinedOrSeparateClass:
    for (const std::string &Value : Values)
      denormalizeString(Args, Spelling, SA, OptClass, TableIndex, Value);
    break;
  default:
    llvm_unreachable("Cannot denormalize an option with option class "
                     "incompatible with string vector denormalization.");
  }
}

static Optional<std::string> normalizeTriple(OptSpecifier Opt, int TableIndex,
                                             const ArgList &Args,
                                             DiagnosticsEngine &Diags,
                                             bool &Success) {
  auto *Arg = Args.getLastArg(Opt);
  if (!Arg)
    return None;
  return llvm::Triple::normalize(Arg->getValue());
}

template <typename T, typename U>
static T mergeForwardValue(T KeyPath, U Value) {
  return static_cast<T>(Value);
}

template <typename T, typename U> static T mergeMaskValue(T KeyPath, U Value) {
  return KeyPath | Value;
}

template <typename T> static T extractForwardValue(T KeyPath) {
  return KeyPath;
}

template <typename T, typename U, U Value>
static T extractMaskValue(T KeyPath) {
  return KeyPath & Value;
}

#define PARSE_OPTION_WITH_MARSHALLING(ARGS, DIAGS, SUCCESS, ID, FLAGS, PARAM,  \
                                      SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,    \
                                      IMPLIED_CHECK, IMPLIED_VALUE,            \
                                      NORMALIZER, MERGER, TABLE_INDEX)         \
  if ((FLAGS)&options::CC1Option) {                                            \
    KEYPATH = MERGER(KEYPATH, DEFAULT_VALUE);                                  \
    if (IMPLIED_CHECK)                                                         \
      KEYPATH = MERGER(KEYPATH, IMPLIED_VALUE);                                \
    if (SHOULD_PARSE)                                                          \
      if (auto MaybeValue =                                                    \
              NORMALIZER(OPT_##ID, TABLE_INDEX, ARGS, DIAGS, SUCCESS))         \
        KEYPATH =                                                              \
            MERGER(KEYPATH, static_cast<decltype(KEYPATH)>(*MaybeValue));      \
  }

// Capture the extracted value as a lambda argument to avoid potential issues
// with lifetime extension of the reference.
#define GENERATE_OPTION_WITH_MARSHALLING(                                      \
    ARGS, STRING_ALLOCATOR, KIND, FLAGS, SPELLING, ALWAYS_EMIT, KEYPATH,       \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, DENORMALIZER, EXTRACTOR,      \
    TABLE_INDEX)                                                               \
  if ((FLAGS)&options::CC1Option) {                                            \
    [&](const auto &Extracted) {                                               \
      if (ALWAYS_EMIT ||                                                       \
          (Extracted !=                                                        \
           static_cast<decltype(KEYPATH)>((IMPLIED_CHECK) ? (IMPLIED_VALUE)    \
                                                          : (DEFAULT_VALUE)))) \
        DENORMALIZER(ARGS, SPELLING, STRING_ALLOCATOR, Option::KIND##Class,    \
                     TABLE_INDEX, Extracted);                                  \
    }(EXTRACTOR(KEYPATH));                                                     \
  }

static const StringRef GetInputKindName(InputKind IK);

static void FixupInvocation(CompilerInvocation &Invocation,
                            DiagnosticsEngine &Diags, const InputArgList &Args,
                            InputKind IK) {
  LangOptions &LangOpts = *Invocation.getLangOpts();
  CodeGenOptions &CodeGenOpts = Invocation.getCodeGenOpts();
  TargetOptions &TargetOpts = Invocation.getTargetOpts();
  FrontendOptions &FrontendOpts = Invocation.getFrontendOpts();
  CodeGenOpts.XRayInstrumentFunctions = LangOpts.XRayInstrument;
  CodeGenOpts.XRayAlwaysEmitCustomEvents = LangOpts.XRayAlwaysEmitCustomEvents;
  CodeGenOpts.XRayAlwaysEmitTypedEvents = LangOpts.XRayAlwaysEmitTypedEvents;
  CodeGenOpts.DisableFree = FrontendOpts.DisableFree;
  FrontendOpts.GenerateGlobalModuleIndex = FrontendOpts.UseGlobalModuleIndex;

  LangOpts.ForceEmitVTables = CodeGenOpts.ForceEmitVTables;
  LangOpts.SpeculativeLoadHardening = CodeGenOpts.SpeculativeLoadHardening;
  LangOpts.CurrentModule = LangOpts.ModuleName;

  llvm::Triple T(TargetOpts.Triple);
  llvm::Triple::ArchType Arch = T.getArch();

  CodeGenOpts.CodeModel = TargetOpts.CodeModel;

  if (LangOpts.getExceptionHandling() != llvm::ExceptionHandling::None &&
      T.isWindowsMSVCEnvironment())
    Diags.Report(diag::err_fe_invalid_exception_model)
        << static_cast<unsigned>(LangOpts.getExceptionHandling()) << T.str();

  if (LangOpts.AppleKext && !LangOpts.CPlusPlus)
    Diags.Report(diag::warn_c_kext);

  if (Args.hasArg(OPT_fconcepts_ts))
    Diags.Report(diag::warn_fe_concepts_ts_flag);

  if (LangOpts.NewAlignOverride &&
      !llvm::isPowerOf2_32(LangOpts.NewAlignOverride)) {
    Arg *A = Args.getLastArg(OPT_fnew_alignment_EQ);
    Diags.Report(diag::err_fe_invalid_alignment)
        << A->getAsString(Args) << A->getValue();
    LangOpts.NewAlignOverride = 0;
  }

  if (Args.hasArg(OPT_fgnu89_inline) && LangOpts.CPlusPlus)
    Diags.Report(diag::err_drv_argument_not_allowed_with)
        << "-fgnu89-inline" << GetInputKindName(IK);

  if (Args.hasArg(OPT_fgpu_allow_device_init) && !LangOpts.HIP)
    Diags.Report(diag::warn_ignored_hip_only_option)
        << Args.getLastArg(OPT_fgpu_allow_device_init)->getAsString(Args);

  if (Args.hasArg(OPT_gpu_max_threads_per_block_EQ) && !LangOpts.HIP)
    Diags.Report(diag::warn_ignored_hip_only_option)
        << Args.getLastArg(OPT_gpu_max_threads_per_block_EQ)->getAsString(Args);

  // -cl-strict-aliasing needs to emit diagnostic in the case where CL > 1.0.
  // This option should be deprecated for CL > 1.0 because
  // this option was added for compatibility with OpenCL 1.0.
  if (Args.getLastArg(OPT_cl_strict_aliasing) && LangOpts.OpenCLVersion > 100)
    Diags.Report(diag::warn_option_invalid_ocl_version)
        << LangOpts.getOpenCLVersionTuple().getAsString()
        << Args.getLastArg(OPT_cl_strict_aliasing)->getAsString(Args);

  if (Arg *A = Args.getLastArg(OPT_fdefault_calling_conv_EQ)) {
    auto DefaultCC = LangOpts.getDefaultCallingConv();

    bool emitError = (DefaultCC == LangOptions::DCC_FastCall ||
                      DefaultCC == LangOptions::DCC_StdCall) &&
                     Arch != llvm::Triple::x86;
    emitError |= (DefaultCC == LangOptions::DCC_VectorCall ||
                  DefaultCC == LangOptions::DCC_RegCall) &&
                 !T.isX86();
    if (emitError)
      Diags.Report(diag::err_drv_argument_not_allowed_with)
          << A->getSpelling() << T.getTriple();
  }

  // Report if OpenMP host file does not exist.
  if (!LangOpts.OMPHostIRFile.empty() &&
      !llvm::sys::fs::exists(LangOpts.OMPHostIRFile))
    Diags.Report(diag::err_drv_omp_host_ir_file_not_found)
        << LangOpts.OMPHostIRFile;

  if (!CodeGenOpts.ProfileRemappingFile.empty() && CodeGenOpts.LegacyPassManager)
    Diags.Report(diag::err_drv_argument_only_allowed_with)
        << Args.getLastArg(OPT_fprofile_remapping_file_EQ)->getAsString(Args)
        << "-fno-legacy-pass-manager";
}

//===----------------------------------------------------------------------===//
// Deserialization (from args)
//===----------------------------------------------------------------------===//

static unsigned getOptimizationLevel(ArgList &Args, InputKind IK,
                                     DiagnosticsEngine &Diags) {
  unsigned DefaultOpt = llvm::CodeGenOpt::None;
  if (IK.getLanguage() == Language::OpenCL && !Args.hasArg(OPT_cl_opt_disable))
    DefaultOpt = llvm::CodeGenOpt::Default;

  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    if (A->getOption().matches(options::OPT_O0))
      return llvm::CodeGenOpt::None;

    if (A->getOption().matches(options::OPT_Ofast))
      return llvm::CodeGenOpt::Aggressive;

    assert(A->getOption().matches(options::OPT_O));

    StringRef S(A->getValue());
    if (S == "s" || S == "z")
      return llvm::CodeGenOpt::Default;

    if (S == "g")
      return llvm::CodeGenOpt::Less;

    return getLastArgIntValue(Args, OPT_O, DefaultOpt, Diags);
  }

  return DefaultOpt;
}

static unsigned getOptimizationLevelSize(ArgList &Args) {
  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    if (A->getOption().matches(options::OPT_O)) {
      switch (A->getValue()[0]) {
      default:
        return 0;
      case 's':
        return 1;
      case 'z':
        return 2;
      }
    }
  }
  return 0;
}

static std::string GetOptName(llvm::opt::OptSpecifier OptSpecifier) {
  static const OptTable &OptTable = getDriverOptTable();
  return OptTable.getOption(OptSpecifier).getPrefixedName();
}

static void addDiagnosticArgs(ArgList &Args, OptSpecifier Group,
                              OptSpecifier GroupWithValue,
                              std::vector<std::string> &Diagnostics) {
  for (auto *A : Args.filtered(Group)) {
    if (A->getOption().getKind() == Option::FlagClass) {
      // The argument is a pure flag (such as OPT_Wall or OPT_Wdeprecated). Add
      // its name (minus the "W" or "R" at the beginning) to the warning list.
      Diagnostics.push_back(
          std::string(A->getOption().getName().drop_front(1)));
    } else if (A->getOption().matches(GroupWithValue)) {
      // This is -Wfoo= or -Rfoo=, where foo is the name of the diagnostic group.
      Diagnostics.push_back(
          std::string(A->getOption().getName().drop_front(1).rtrim("=-")));
    } else {
      // Otherwise, add its value (for OPT_W_Joined and similar).
      for (const auto *Arg : A->getValues())
        Diagnostics.emplace_back(Arg);
    }
  }
}

// Parse the Static Analyzer configuration. If \p Diags is set to nullptr,
// it won't verify the input.
static void parseAnalyzerConfigs(AnalyzerOptions &AnOpts,
                                 DiagnosticsEngine *Diags);

static void getAllNoBuiltinFuncValues(ArgList &Args,
                                      std::vector<std::string> &Funcs) {
  SmallVector<const char *, 8> Values;
  for (const auto &Arg : Args) {
    const Option &O = Arg->getOption();
    if (O.matches(options::OPT_fno_builtin_)) {
      const char *FuncName = Arg->getValue();
      if (Builtin::Context::isBuiltinFunc(FuncName))
        Values.push_back(FuncName);
    }
  }
  Funcs.insert(Funcs.end(), Values.begin(), Values.end());
}

static bool ParseAnalyzerArgs(AnalyzerOptions &Opts, ArgList &Args,
                              DiagnosticsEngine &Diags) {
  bool Success = true;
  if (Arg *A = Args.getLastArg(OPT_analyzer_store)) {
    StringRef Name = A->getValue();
    AnalysisStores Value = llvm::StringSwitch<AnalysisStores>(Name)
#define ANALYSIS_STORE(NAME, CMDFLAG, DESC, CREATFN) \
      .Case(CMDFLAG, NAME##Model)
#include "clang/StaticAnalyzer/Core/Analyses.def"
      .Default(NumStores);
    if (Value == NumStores) {
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << Name;
      Success = false;
    } else {
      Opts.AnalysisStoreOpt = Value;
    }
  }

  if (Arg *A = Args.getLastArg(OPT_analyzer_constraints)) {
    StringRef Name = A->getValue();
    AnalysisConstraints Value = llvm::StringSwitch<AnalysisConstraints>(Name)
#define ANALYSIS_CONSTRAINTS(NAME, CMDFLAG, DESC, CREATFN) \
      .Case(CMDFLAG, NAME##Model)
#include "clang/StaticAnalyzer/Core/Analyses.def"
      .Default(NumConstraints);
    if (Value == NumConstraints) {
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << Name;
      Success = false;
    } else {
      Opts.AnalysisConstraintsOpt = Value;
    }
  }

  if (Arg *A = Args.getLastArg(OPT_analyzer_output)) {
    StringRef Name = A->getValue();
    AnalysisDiagClients Value = llvm::StringSwitch<AnalysisDiagClients>(Name)
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATFN) \
      .Case(CMDFLAG, PD_##NAME)
#include "clang/StaticAnalyzer/Core/Analyses.def"
      .Default(NUM_ANALYSIS_DIAG_CLIENTS);
    if (Value == NUM_ANALYSIS_DIAG_CLIENTS) {
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << Name;
      Success = false;
    } else {
      Opts.AnalysisDiagOpt = Value;
    }
  }

  if (Arg *A = Args.getLastArg(OPT_analyzer_purge)) {
    StringRef Name = A->getValue();
    AnalysisPurgeMode Value = llvm::StringSwitch<AnalysisPurgeMode>(Name)
#define ANALYSIS_PURGE(NAME, CMDFLAG, DESC) \
      .Case(CMDFLAG, NAME)
#include "clang/StaticAnalyzer/Core/Analyses.def"
      .Default(NumPurgeModes);
    if (Value == NumPurgeModes) {
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << Name;
      Success = false;
    } else {
      Opts.AnalysisPurgeOpt = Value;
    }
  }

  if (Arg *A = Args.getLastArg(OPT_analyzer_inlining_mode)) {
    StringRef Name = A->getValue();
    AnalysisInliningMode Value = llvm::StringSwitch<AnalysisInliningMode>(Name)
#define ANALYSIS_INLINING_MODE(NAME, CMDFLAG, DESC) \
      .Case(CMDFLAG, NAME)
#include "clang/StaticAnalyzer/Core/Analyses.def"
      .Default(NumInliningModes);
    if (Value == NumInliningModes) {
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << Name;
      Success = false;
    } else {
      Opts.InliningMode = Value;
    }
  }

  Opts.CheckersAndPackages.clear();
  for (const Arg *A :
       Args.filtered(OPT_analyzer_checker, OPT_analyzer_disable_checker)) {
    A->claim();
    bool IsEnabled = A->getOption().getID() == OPT_analyzer_checker;
    // We can have a list of comma separated checker names, e.g:
    // '-analyzer-checker=cocoa,unix'
    StringRef CheckerAndPackageList = A->getValue();
    SmallVector<StringRef, 16> CheckersAndPackages;
    CheckerAndPackageList.split(CheckersAndPackages, ",");
    for (const StringRef &CheckerOrPackage : CheckersAndPackages)
      Opts.CheckersAndPackages.emplace_back(std::string(CheckerOrPackage),
                                            IsEnabled);
  }

  // Go through the analyzer configuration options.
  for (const auto *A : Args.filtered(OPT_analyzer_config)) {

    // We can have a list of comma separated config names, e.g:
    // '-analyzer-config key1=val1,key2=val2'
    StringRef configList = A->getValue();
    SmallVector<StringRef, 4> configVals;
    configList.split(configVals, ",");
    for (const auto &configVal : configVals) {
      StringRef key, val;
      std::tie(key, val) = configVal.split("=");
      if (val.empty()) {
        Diags.Report(SourceLocation(),
                     diag::err_analyzer_config_no_value) << configVal;
        Success = false;
        break;
      }
      if (val.find('=') != StringRef::npos) {
        Diags.Report(SourceLocation(),
                     diag::err_analyzer_config_multiple_values)
          << configVal;
        Success = false;
        break;
      }

      // TODO: Check checker options too, possibly in CheckerRegistry.
      // Leave unknown non-checker configs unclaimed.
      if (!key.contains(":") && Opts.isUnknownAnalyzerConfig(key)) {
        if (Opts.ShouldEmitErrorsOnInvalidConfigValue)
          Diags.Report(diag::err_analyzer_config_unknown) << key;
        continue;
      }

      A->claim();
      Opts.Config[key] = std::string(val);
    }
  }

  if (Opts.ShouldEmitErrorsOnInvalidConfigValue)
    parseAnalyzerConfigs(Opts, &Diags);
  else
    parseAnalyzerConfigs(Opts, nullptr);

  llvm::raw_string_ostream os(Opts.FullCompilerInvocation);
  for (unsigned i = 0; i < Args.getNumInputArgStrings(); ++i) {
    if (i != 0)
      os << " ";
    os << Args.getArgString(i);
  }
  os.flush();

  return Success;
}

static StringRef getStringOption(AnalyzerOptions::ConfigTable &Config,
                                 StringRef OptionName, StringRef DefaultVal) {
  return Config.insert({OptionName, std::string(DefaultVal)}).first->second;
}

static void initOption(AnalyzerOptions::ConfigTable &Config,
                       DiagnosticsEngine *Diags,
                       StringRef &OptionField, StringRef Name,
                       StringRef DefaultVal) {
  // String options may be known to invalid (e.g. if the expected string is a
  // file name, but the file does not exist), those will have to be checked in
  // parseConfigs.
  OptionField = getStringOption(Config, Name, DefaultVal);
}

static void initOption(AnalyzerOptions::ConfigTable &Config,
                       DiagnosticsEngine *Diags,
                       bool &OptionField, StringRef Name, bool DefaultVal) {
  auto PossiblyInvalidVal = llvm::StringSwitch<Optional<bool>>(
                 getStringOption(Config, Name, (DefaultVal ? "true" : "false")))
      .Case("true", true)
      .Case("false", false)
      .Default(None);

  if (!PossiblyInvalidVal) {
    if (Diags)
      Diags->Report(diag::err_analyzer_config_invalid_input)
        << Name << "a boolean";
    else
      OptionField = DefaultVal;
  } else
    OptionField = PossiblyInvalidVal.getValue();
}

static void initOption(AnalyzerOptions::ConfigTable &Config,
                       DiagnosticsEngine *Diags,
                       unsigned &OptionField, StringRef Name,
                       unsigned DefaultVal) {

  OptionField = DefaultVal;
  bool HasFailed = getStringOption(Config, Name, std::to_string(DefaultVal))
                     .getAsInteger(0, OptionField);
  if (Diags && HasFailed)
    Diags->Report(diag::err_analyzer_config_invalid_input)
      << Name << "an unsigned";
}

static void parseAnalyzerConfigs(AnalyzerOptions &AnOpts,
                                 DiagnosticsEngine *Diags) {
  // TODO: There's no need to store the entire configtable, it'd be plenty
  // enough tostore checker options.

#define ANALYZER_OPTION(TYPE, NAME, CMDFLAG, DESC, DEFAULT_VAL)                \
  initOption(AnOpts.Config, Diags, AnOpts.NAME, CMDFLAG, DEFAULT_VAL);

#define ANALYZER_OPTION_DEPENDS_ON_USER_MODE(TYPE, NAME, CMDFLAG, DESC,        \
                                           SHALLOW_VAL, DEEP_VAL)              \
  switch (AnOpts.getUserMode()) {                                              \
  case UMK_Shallow:                                                            \
    initOption(AnOpts.Config, Diags, AnOpts.NAME, CMDFLAG, SHALLOW_VAL);       \
    break;                                                                     \
  case UMK_Deep:                                                               \
    initOption(AnOpts.Config, Diags, AnOpts.NAME, CMDFLAG, DEEP_VAL);          \
    break;                                                                     \
  }                                                                            \

#include "clang/StaticAnalyzer/Core/AnalyzerOptions.def"
#undef ANALYZER_OPTION
#undef ANALYZER_OPTION_DEPENDS_ON_USER_MODE

  // At this point, AnalyzerOptions is configured. Let's validate some options.

  // FIXME: Here we try to validate the silenced checkers or packages are valid.
  // The current approach only validates the registered checkers which does not
  // contain the runtime enabled checkers and optimally we would validate both.
  if (!AnOpts.RawSilencedCheckersAndPackages.empty()) {
    std::vector<StringRef> Checkers =
        AnOpts.getRegisteredCheckers(/*IncludeExperimental=*/true);
    std::vector<StringRef> Packages =
        AnOpts.getRegisteredPackages(/*IncludeExperimental=*/true);

    SmallVector<StringRef, 16> CheckersAndPackages;
    AnOpts.RawSilencedCheckersAndPackages.split(CheckersAndPackages, ";");

    for (const StringRef &CheckerOrPackage : CheckersAndPackages) {
      if (Diags) {
        bool IsChecker = CheckerOrPackage.contains('.');
        bool IsValidName =
            IsChecker
                ? llvm::find(Checkers, CheckerOrPackage) != Checkers.end()
                : llvm::find(Packages, CheckerOrPackage) != Packages.end();

        if (!IsValidName)
          Diags->Report(diag::err_unknown_analyzer_checker_or_package)
              << CheckerOrPackage;
      }

      AnOpts.SilencedCheckersAndPackages.emplace_back(CheckerOrPackage);
    }
  }

  if (!Diags)
    return;

  if (AnOpts.ShouldTrackConditionsDebug && !AnOpts.ShouldTrackConditions)
    Diags->Report(diag::err_analyzer_config_invalid_input)
        << "track-conditions-debug" << "'track-conditions' to also be enabled";

  if (!AnOpts.CTUDir.empty() && !llvm::sys::fs::is_directory(AnOpts.CTUDir))
    Diags->Report(diag::err_analyzer_config_invalid_input) << "ctu-dir"
                                                           << "a filename";

  if (!AnOpts.ModelPath.empty() &&
      !llvm::sys::fs::is_directory(AnOpts.ModelPath))
    Diags->Report(diag::err_analyzer_config_invalid_input) << "model-path"
                                                           << "a filename";
}

/// Create a new Regex instance out of the string value in \p RpassArg.
/// It returns a pointer to the newly generated Regex instance.
static std::shared_ptr<llvm::Regex>
GenerateOptimizationRemarkRegex(DiagnosticsEngine &Diags, ArgList &Args,
                                Arg *RpassArg) {
  StringRef Val = RpassArg->getValue();
  std::string RegexError;
  std::shared_ptr<llvm::Regex> Pattern = std::make_shared<llvm::Regex>(Val);
  if (!Pattern->isValid(RegexError)) {
    Diags.Report(diag::err_drv_optimization_remark_pattern)
        << RegexError << RpassArg->getAsString(Args);
    Pattern.reset();
  }
  return Pattern;
}

static bool parseDiagnosticLevelMask(StringRef FlagName,
                                     const std::vector<std::string> &Levels,
                                     DiagnosticsEngine &Diags,
                                     DiagnosticLevelMask &M) {
  bool Success = true;
  for (const auto &Level : Levels) {
    DiagnosticLevelMask const PM =
      llvm::StringSwitch<DiagnosticLevelMask>(Level)
        .Case("note",    DiagnosticLevelMask::Note)
        .Case("remark",  DiagnosticLevelMask::Remark)
        .Case("warning", DiagnosticLevelMask::Warning)
        .Case("error",   DiagnosticLevelMask::Error)
        .Default(DiagnosticLevelMask::None);
    if (PM == DiagnosticLevelMask::None) {
      Success = false;
      Diags.Report(diag::err_drv_invalid_value) << FlagName << Level;
    }
    M = M | PM;
  }
  return Success;
}

static void parseSanitizerKinds(StringRef FlagName,
                                const std::vector<std::string> &Sanitizers,
                                DiagnosticsEngine &Diags, SanitizerSet &S) {
  for (const auto &Sanitizer : Sanitizers) {
    SanitizerMask K = parseSanitizerValue(Sanitizer, /*AllowGroups=*/false);
    if (K == SanitizerMask())
      Diags.Report(diag::err_drv_invalid_value) << FlagName << Sanitizer;
    else
      S.set(K, true);
  }
}

static void parseXRayInstrumentationBundle(StringRef FlagName, StringRef Bundle,
                                           ArgList &Args, DiagnosticsEngine &D,
                                           XRayInstrSet &S) {
  llvm::SmallVector<StringRef, 2> BundleParts;
  llvm::SplitString(Bundle, BundleParts, ",");
  for (const auto &B : BundleParts) {
    auto Mask = parseXRayInstrValue(B);
    if (Mask == XRayInstrKind::None)
      if (B != "none")
        D.Report(diag::err_drv_invalid_value) << FlagName << Bundle;
      else
        S.Mask = Mask;
    else if (Mask == XRayInstrKind::All)
      S.Mask = Mask;
    else
      S.set(Mask, true);
  }
}

// Set the profile kind using fprofile-instrument-use-path.
static void setPGOUseInstrumentor(CodeGenOptions &Opts,
                                  const Twine &ProfileName) {
  auto ReaderOrErr = llvm::IndexedInstrProfReader::create(ProfileName);
  // In error, return silently and let Clang PGOUse report the error message.
  if (auto E = ReaderOrErr.takeError()) {
    llvm::consumeError(std::move(E));
    Opts.setProfileUse(CodeGenOptions::ProfileClangInstr);
    return;
  }
  std::unique_ptr<llvm::IndexedInstrProfReader> PGOReader =
    std::move(ReaderOrErr.get());
  if (PGOReader->isIRLevelProfile()) {
    if (PGOReader->hasCSIRLevelProfile())
      Opts.setProfileUse(CodeGenOptions::ProfileCSIRInstr);
    else
      Opts.setProfileUse(CodeGenOptions::ProfileIRInstr);
  } else
    Opts.setProfileUse(CodeGenOptions::ProfileClangInstr);
}

bool CompilerInvocation::ParseCodeGenArgs(CodeGenOptions &Opts, ArgList &Args,
                                          InputKind IK,
                                          DiagnosticsEngine &Diags,
                                          const llvm::Triple &T,
                                          const std::string &OutputFile,
                                          const LangOptions &LangOptsRef) {
  bool Success = true;

  unsigned OptimizationLevel = getOptimizationLevel(Args, IK, Diags);
  // TODO: This could be done in Driver
  unsigned MaxOptLevel = 3;
  if (OptimizationLevel > MaxOptLevel) {
    // If the optimization level is not supported, fall back on the default
    // optimization
    Diags.Report(diag::warn_drv_optimization_value)
        << Args.getLastArg(OPT_O)->getAsString(Args) << "-O" << MaxOptLevel;
    OptimizationLevel = MaxOptLevel;
  }
  Opts.OptimizationLevel = OptimizationLevel;

  // The key paths of codegen options defined in Options.td start with
  // "CodeGenOpts.". Let's provide the expected variable name and type.
  CodeGenOptions &CodeGenOpts = Opts;
  // Some codegen options depend on language options. Let's provide the expected
  // variable name and type.
  const LangOptions *LangOpts = &LangOptsRef;

#define CODEGEN_OPTION_WITH_MARSHALLING(                                       \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(Args, Diags, Success, ID, FLAGS, PARAM,        \
                                SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,          \
                                IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER,      \
                                MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef CODEGEN_OPTION_WITH_MARSHALLING

  // At O0 we want to fully disable inlining outside of cases marked with
  // 'alwaysinline' that are required for correctness.
  Opts.setInlining((Opts.OptimizationLevel == 0)
                       ? CodeGenOptions::OnlyAlwaysInlining
                       : CodeGenOptions::NormalInlining);
  // Explicit inlining flags can disable some or all inlining even at
  // optimization levels above zero.
  if (Arg *InlineArg = Args.getLastArg(
          options::OPT_finline_functions, options::OPT_finline_hint_functions,
          options::OPT_fno_inline_functions, options::OPT_fno_inline)) {
    if (Opts.OptimizationLevel > 0) {
      const Option &InlineOpt = InlineArg->getOption();
      if (InlineOpt.matches(options::OPT_finline_functions))
        Opts.setInlining(CodeGenOptions::NormalInlining);
      else if (InlineOpt.matches(options::OPT_finline_hint_functions))
        Opts.setInlining(CodeGenOptions::OnlyHintInlining);
      else
        Opts.setInlining(CodeGenOptions::OnlyAlwaysInlining);
    }
  }

  // PIC defaults to -fno-direct-access-external-data while non-PIC defaults to
  // -fdirect-access-external-data.
  Opts.DirectAccessExternalData =
      Args.hasArg(OPT_fdirect_access_external_data) ||
      (!Args.hasArg(OPT_fno_direct_access_external_data) &&
       getLastArgIntValue(Args, OPT_pic_level, 0, Diags) == 0);

  // If -fuse-ctor-homing is set and limited debug info is already on, then use
  // constructor homing.
  if (Args.getLastArg(OPT_fuse_ctor_homing))
    if (Opts.getDebugInfo() == codegenoptions::LimitedDebugInfo)
      Opts.setDebugInfo(codegenoptions::DebugInfoConstructor);

  for (const auto &Arg : Args.getAllArgValues(OPT_fdebug_prefix_map_EQ)) {
    auto Split = StringRef(Arg).split('=');
    Opts.DebugPrefixMap.insert(
        {std::string(Split.first), std::string(Split.second)});
  }

  for (const auto &Arg : Args.getAllArgValues(OPT_fprofile_prefix_map_EQ)) {
    auto Split = StringRef(Arg).split('=');
    Opts.ProfilePrefixMap.insert(
        {std::string(Split.first), std::string(Split.second)});
  }

  const llvm::Triple::ArchType DebugEntryValueArchs[] = {
      llvm::Triple::x86, llvm::Triple::x86_64, llvm::Triple::aarch64,
      llvm::Triple::arm, llvm::Triple::armeb, llvm::Triple::mips,
      llvm::Triple::mipsel, llvm::Triple::mips64, llvm::Triple::mips64el};

  if (Opts.OptimizationLevel > 0 && Opts.hasReducedDebugInfo() &&
      llvm::is_contained(DebugEntryValueArchs, T.getArch()))
    Opts.EmitCallSiteInfo = true;

  Opts.NewStructPathTBAA = !Args.hasArg(OPT_no_struct_path_tbaa) &&
                           Args.hasArg(OPT_new_struct_path_tbaa);
  Opts.OptimizeSize = getOptimizationLevelSize(Args);
  Opts.SimplifyLibCalls = !(Args.hasArg(OPT_fno_builtin) ||
                            Args.hasArg(OPT_ffreestanding));
  if (Opts.SimplifyLibCalls)
    getAllNoBuiltinFuncValues(Args, Opts.NoBuiltinFuncs);
  Opts.UnrollLoops =
      Args.hasFlag(OPT_funroll_loops, OPT_fno_unroll_loops,
                   (Opts.OptimizationLevel > 1));

  Opts.BinutilsVersion =
      std::string(Args.getLastArgValue(OPT_fbinutils_version_EQ));

  Opts.DebugNameTable = static_cast<unsigned>(
      Args.hasArg(OPT_ggnu_pubnames)
          ? llvm::DICompileUnit::DebugNameTableKind::GNU
          : Args.hasArg(OPT_gpubnames)
                ? llvm::DICompileUnit::DebugNameTableKind::Default
                : llvm::DICompileUnit::DebugNameTableKind::None);

  if (!Opts.ProfileInstrumentUsePath.empty())
    setPGOUseInstrumentor(Opts, Opts.ProfileInstrumentUsePath);

  if (const Arg *A = Args.getLastArg(OPT_ftime_report, OPT_ftime_report_EQ)) {
    Opts.TimePasses = true;

    // -ftime-report= is only for new pass manager.
    if (A->getOption().getID() == OPT_ftime_report_EQ) {
      if (Opts.LegacyPassManager)
        Diags.Report(diag::err_drv_argument_only_allowed_with)
            << A->getAsString(Args) << "-fno-legacy-pass-manager";

      StringRef Val = A->getValue();
      if (Val == "per-pass")
        Opts.TimePassesPerRun = false;
      else if (Val == "per-pass-run")
        Opts.TimePassesPerRun = true;
      else
        Diags.Report(diag::err_drv_invalid_value)
            << A->getAsString(Args) << A->getValue();
    }
  }

  // Basic Block Sections implies Function Sections.
  Opts.FunctionSections =
      Args.hasArg(OPT_ffunction_sections) ||
      (Opts.BBSections != "none" && Opts.BBSections != "labels");

  Opts.PrepareForLTO = Args.hasArg(OPT_flto, OPT_flto_EQ);
  Opts.PrepareForThinLTO = false;
  if (Arg *A = Args.getLastArg(OPT_flto_EQ)) {
    StringRef S = A->getValue();
    if (S == "thin")
      Opts.PrepareForThinLTO = true;
    else if (S != "full")
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << S;
  }
  if (Arg *A = Args.getLastArg(OPT_fthinlto_index_EQ)) {
    if (IK.getLanguage() != Language::LLVM_IR)
      Diags.Report(diag::err_drv_argument_only_allowed_with)
          << A->getAsString(Args) << "-x ir";
    Opts.ThinLTOIndexFile =
        std::string(Args.getLastArgValue(OPT_fthinlto_index_EQ));
  }
  if (Arg *A = Args.getLastArg(OPT_save_temps_EQ))
    Opts.SaveTempsFilePrefix =
        llvm::StringSwitch<std::string>(A->getValue())
            .Case("obj", OutputFile)
            .Default(llvm::sys::path::filename(OutputFile).str());

  // The memory profile runtime appends the pid to make this name more unique.
  const char *MemProfileBasename = "memprof.profraw";
  if (Args.hasArg(OPT_fmemory_profile_EQ)) {
    SmallString<128> Path(
        std::string(Args.getLastArgValue(OPT_fmemory_profile_EQ)));
    llvm::sys::path::append(Path, MemProfileBasename);
    Opts.MemoryProfileOutput = std::string(Path);
  } else if (Args.hasArg(OPT_fmemory_profile))
    Opts.MemoryProfileOutput = MemProfileBasename;

  if (Opts.EmitGcovArcs || Opts.EmitGcovNotes) {
    if (Args.hasArg(OPT_coverage_version_EQ)) {
      StringRef CoverageVersion = Args.getLastArgValue(OPT_coverage_version_EQ);
      if (CoverageVersion.size() != 4) {
        Diags.Report(diag::err_drv_invalid_value)
            << Args.getLastArg(OPT_coverage_version_EQ)->getAsString(Args)
            << CoverageVersion;
      } else {
        memcpy(Opts.CoverageVersion, CoverageVersion.data(), 4);
      }
    }
  }
  // FIXME: For backend options that are not yet recorded as function
  // attributes in the IR, keep track of them so we can embed them in a
  // separate data section and use them when building the bitcode.
  for (const auto &A : Args) {
    // Do not encode output and input.
    if (A->getOption().getID() == options::OPT_o ||
        A->getOption().getID() == options::OPT_INPUT ||
        A->getOption().getID() == options::OPT_x ||
        A->getOption().getID() == options::OPT_fembed_bitcode ||
        A->getOption().matches(options::OPT_W_Group))
      continue;
    ArgStringList ASL;
    A->render(Args, ASL);
    for (const auto &arg : ASL) {
      StringRef ArgStr(arg);
      Opts.CmdArgs.insert(Opts.CmdArgs.end(), ArgStr.begin(), ArgStr.end());
      // using \00 to separate each commandline options.
      Opts.CmdArgs.push_back('\0');
    }
  }

  auto XRayInstrBundles =
      Args.getAllArgValues(OPT_fxray_instrumentation_bundle);
  if (XRayInstrBundles.empty())
    Opts.XRayInstrumentationBundle.Mask = XRayInstrKind::All;
  else
    for (const auto &A : XRayInstrBundles)
      parseXRayInstrumentationBundle("-fxray-instrumentation-bundle=", A, Args,
                                     Diags, Opts.XRayInstrumentationBundle);

  if (const Arg *A = Args.getLastArg(OPT_fcf_protection_EQ)) {
    StringRef Name = A->getValue();
    if (Name == "full") {
      Opts.CFProtectionReturn = 1;
      Opts.CFProtectionBranch = 1;
    } else if (Name == "return")
      Opts.CFProtectionReturn = 1;
    else if (Name == "branch")
      Opts.CFProtectionBranch = 1;
    else if (Name != "none") {
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Name;
      Success = false;
    }
  }

  for (auto *A :
       Args.filtered(OPT_mlink_bitcode_file, OPT_mlink_builtin_bitcode)) {
    CodeGenOptions::BitcodeFileToLink F;
    F.Filename = A->getValue();
    if (A->getOption().matches(OPT_mlink_builtin_bitcode)) {
      F.LinkFlags = llvm::Linker::Flags::LinkOnlyNeeded;
      // When linking CUDA bitcode, propagate function attributes so that
      // e.g. libdevice gets fast-math attrs if we're building with fast-math.
      F.PropagateAttrs = true;
      F.Internalize = true;
    }
    Opts.LinkBitcodeFiles.push_back(F);
  }

  if (Args.getLastArg(OPT_femulated_tls) ||
      Args.getLastArg(OPT_fno_emulated_tls)) {
    Opts.ExplicitEmulatedTLS = true;
  }

  if (Arg *A = Args.getLastArg(OPT_fdenormal_fp_math_EQ)) {
    StringRef Val = A->getValue();
    Opts.FPDenormalMode = llvm::parseDenormalFPAttribute(Val);
    if (!Opts.FPDenormalMode.isValid())
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Val;
  }

  if (Arg *A = Args.getLastArg(OPT_fdenormal_fp_math_f32_EQ)) {
    StringRef Val = A->getValue();
    Opts.FP32DenormalMode = llvm::parseDenormalFPAttribute(Val);
    if (!Opts.FP32DenormalMode.isValid())
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Val;
  }

  // X86_32 has -fppc-struct-return and -freg-struct-return.
  // PPC32 has -maix-struct-return and -msvr4-struct-return.
  if (Arg *A =
          Args.getLastArg(OPT_fpcc_struct_return, OPT_freg_struct_return,
                          OPT_maix_struct_return, OPT_msvr4_struct_return)) {
    // TODO: We might want to consider enabling these options on AIX in the
    // future.
    if (T.isOSAIX())
      Diags.Report(diag::err_drv_unsupported_opt_for_target)
          << A->getSpelling() << T.str();

    const Option &O = A->getOption();
    if (O.matches(OPT_fpcc_struct_return) ||
        O.matches(OPT_maix_struct_return)) {
      Opts.setStructReturnConvention(CodeGenOptions::SRCK_OnStack);
    } else {
      assert(O.matches(OPT_freg_struct_return) ||
             O.matches(OPT_msvr4_struct_return));
      Opts.setStructReturnConvention(CodeGenOptions::SRCK_InRegs);
    }
  }

  if (T.isOSAIX() && (Args.hasArg(OPT_mignore_xcoff_visibility) ||
                      !Args.hasArg(OPT_fvisibility)))
    Opts.IgnoreXCOFFVisibility = 1;

  if (Arg *A =
          Args.getLastArg(OPT_mabi_EQ_vec_default, OPT_mabi_EQ_vec_extabi)) {
    if (!T.isOSAIX())
      Diags.Report(diag::err_drv_unsupported_opt_for_target)
          << A->getSpelling() << T.str();

    const Option &O = A->getOption();
    if (O.matches(OPT_mabi_EQ_vec_default))
      Diags.Report(diag::err_aix_default_altivec_abi)
          << A->getSpelling() << T.str();
    else {
      assert(O.matches(OPT_mabi_EQ_vec_extabi));
      Opts.EnableAIXExtendedAltivecABI = 1;
    }
  }

  bool NeedLocTracking = false;

  if (!Opts.OptRecordFile.empty())
    NeedLocTracking = true;

  if (Arg *A = Args.getLastArg(OPT_opt_record_passes)) {
    Opts.OptRecordPasses = A->getValue();
    NeedLocTracking = true;
  }

  if (Arg *A = Args.getLastArg(OPT_opt_record_format)) {
    Opts.OptRecordFormat = A->getValue();
    NeedLocTracking = true;
  }

  if (Arg *A = Args.getLastArg(OPT_Rpass_EQ)) {
    Opts.OptimizationRemarkPattern =
        GenerateOptimizationRemarkRegex(Diags, Args, A);
    NeedLocTracking = true;
  }

  if (Arg *A = Args.getLastArg(OPT_Rpass_missed_EQ)) {
    Opts.OptimizationRemarkMissedPattern =
        GenerateOptimizationRemarkRegex(Diags, Args, A);
    NeedLocTracking = true;
  }

  if (Arg *A = Args.getLastArg(OPT_Rpass_analysis_EQ)) {
    Opts.OptimizationRemarkAnalysisPattern =
        GenerateOptimizationRemarkRegex(Diags, Args, A);
    NeedLocTracking = true;
  }

  bool UsingSampleProfile = !Opts.SampleProfileFile.empty();
  bool UsingProfile = UsingSampleProfile ||
      (Opts.getProfileUse() != CodeGenOptions::ProfileNone);

  if (Opts.DiagnosticsWithHotness && !UsingProfile &&
      // An IR file will contain PGO as metadata
      IK.getLanguage() != Language::LLVM_IR)
    Diags.Report(diag::warn_drv_diagnostics_hotness_requires_pgo)
        << "-fdiagnostics-show-hotness";

  // Parse remarks hotness threshold. Valid value is either integer or 'auto'.
  if (auto *arg =
          Args.getLastArg(options::OPT_fdiagnostics_hotness_threshold_EQ)) {
    auto ResultOrErr =
        llvm::remarks::parseHotnessThresholdOption(arg->getValue());

    if (!ResultOrErr) {
      Diags.Report(diag::err_drv_invalid_diagnotics_hotness_threshold)
          << "-fdiagnostics-hotness-threshold=";
    } else {
      Opts.DiagnosticsHotnessThreshold = *ResultOrErr;
      if ((!Opts.DiagnosticsHotnessThreshold.hasValue() ||
           Opts.DiagnosticsHotnessThreshold.getValue() > 0) &&
          !UsingProfile)
        Diags.Report(diag::warn_drv_diagnostics_hotness_requires_pgo)
            << "-fdiagnostics-hotness-threshold=";
    }
  }

  // If the user requested to use a sample profile for PGO, then the
  // backend will need to track source location information so the profile
  // can be incorporated into the IR.
  if (UsingSampleProfile)
    NeedLocTracking = true;

  // If the user requested a flag that requires source locations available in
  // the backend, make sure that the backend tracks source location information.
  if (NeedLocTracking && Opts.getDebugInfo() == codegenoptions::NoDebugInfo)
    Opts.setDebugInfo(codegenoptions::LocTrackingOnly);

  // Parse -fsanitize-recover= arguments.
  // FIXME: Report unrecoverable sanitizers incorrectly specified here.
  parseSanitizerKinds("-fsanitize-recover=",
                      Args.getAllArgValues(OPT_fsanitize_recover_EQ), Diags,
                      Opts.SanitizeRecover);
  parseSanitizerKinds("-fsanitize-trap=",
                      Args.getAllArgValues(OPT_fsanitize_trap_EQ), Diags,
                      Opts.SanitizeTrap);

  Opts.EmitVersionIdentMetadata = Args.hasFlag(OPT_Qy, OPT_Qn, true);

  return Success;
}

static void ParseDependencyOutputArgs(DependencyOutputOptions &Opts,
                                      ArgList &Args) {
  if (Args.hasArg(OPT_show_includes)) {
    // Writing both /showIncludes and preprocessor output to stdout
    // would produce interleaved output, so use stderr for /showIncludes.
    // This behaves the same as cl.exe, when /E, /EP or /P are passed.
    if (Args.hasArg(options::OPT_E) || Args.hasArg(options::OPT_P))
      Opts.ShowIncludesDest = ShowIncludesDestination::Stderr;
    else
      Opts.ShowIncludesDest = ShowIncludesDestination::Stdout;
  } else {
    Opts.ShowIncludesDest = ShowIncludesDestination::None;
  }
  // Add sanitizer blacklists as extra dependencies.
  // They won't be discovered by the regular preprocessor, so
  // we let make / ninja to know about this implicit dependency.
  if (!Args.hasArg(OPT_fno_sanitize_blacklist)) {
    for (const auto *A : Args.filtered(OPT_fsanitize_blacklist)) {
      StringRef Val = A->getValue();
      if (Val.find('=') == StringRef::npos)
        Opts.ExtraDeps.push_back(std::string(Val));
    }
    if (Opts.IncludeSystemHeaders) {
      for (const auto *A : Args.filtered(OPT_fsanitize_system_blacklist)) {
        StringRef Val = A->getValue();
        if (Val.find('=') == StringRef::npos)
          Opts.ExtraDeps.push_back(std::string(Val));
      }
    }
  }

  // -fprofile-list= dependencies.
  for (const auto &Filename : Args.getAllArgValues(OPT_fprofile_list_EQ))
    Opts.ExtraDeps.push_back(Filename);

  // Propagate the extra dependencies.
  for (const auto *A : Args.filtered(OPT_fdepfile_entry)) {
    Opts.ExtraDeps.push_back(A->getValue());
  }

  // Only the -fmodule-file=<file> form.
  for (const auto *A : Args.filtered(OPT_fmodule_file)) {
    StringRef Val = A->getValue();
    if (Val.find('=') == StringRef::npos)
      Opts.ExtraDeps.push_back(std::string(Val));
  }
}

static bool parseShowColorsArgs(const ArgList &Args, bool DefaultColor) {
  // Color diagnostics default to auto ("on" if terminal supports) in the driver
  // but default to off in cc1, needing an explicit OPT_fdiagnostics_color.
  // Support both clang's -f[no-]color-diagnostics and gcc's
  // -f[no-]diagnostics-colors[=never|always|auto].
  enum {
    Colors_On,
    Colors_Off,
    Colors_Auto
  } ShowColors = DefaultColor ? Colors_Auto : Colors_Off;
  for (auto *A : Args) {
    const Option &O = A->getOption();
    if (O.matches(options::OPT_fcolor_diagnostics) ||
        O.matches(options::OPT_fdiagnostics_color)) {
      ShowColors = Colors_On;
    } else if (O.matches(options::OPT_fno_color_diagnostics) ||
               O.matches(options::OPT_fno_diagnostics_color)) {
      ShowColors = Colors_Off;
    } else if (O.matches(options::OPT_fdiagnostics_color_EQ)) {
      StringRef Value(A->getValue());
      if (Value == "always")
        ShowColors = Colors_On;
      else if (Value == "never")
        ShowColors = Colors_Off;
      else if (Value == "auto")
        ShowColors = Colors_Auto;
    }
  }
  return ShowColors == Colors_On ||
         (ShowColors == Colors_Auto &&
          llvm::sys::Process::StandardErrHasColors());
}

static bool checkVerifyPrefixes(const std::vector<std::string> &VerifyPrefixes,
                                DiagnosticsEngine &Diags) {
  bool Success = true;
  for (const auto &Prefix : VerifyPrefixes) {
    // Every prefix must start with a letter and contain only alphanumeric
    // characters, hyphens, and underscores.
    auto BadChar = llvm::find_if(Prefix, [](char C) {
      return !isAlphanumeric(C) && C != '-' && C != '_';
    });
    if (BadChar != Prefix.end() || !isLetter(Prefix[0])) {
      Success = false;
      Diags.Report(diag::err_drv_invalid_value) << "-verify=" << Prefix;
      Diags.Report(diag::note_drv_verify_prefix_spelling);
    }
  }
  return Success;
}

bool CompilerInvocation::parseSimpleArgs(const ArgList &Args,
                                         DiagnosticsEngine &Diags) {
  bool Success = true;

#define OPTION_WITH_MARSHALLING(                                               \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(Args, Diags, Success, ID, FLAGS, PARAM,        \
                                SHOULD_PARSE, this->KEYPATH, DEFAULT_VALUE,    \
                                IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER,      \
                                MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef OPTION_WITH_MARSHALLING

  return Success;
}

bool clang::ParseDiagnosticArgs(DiagnosticOptions &Opts, ArgList &Args,
                                DiagnosticsEngine *Diags,
                                bool DefaultDiagColor) {
  Optional<DiagnosticsEngine> IgnoringDiags;
  if (!Diags) {
    IgnoringDiags.emplace(new DiagnosticIDs(), new DiagnosticOptions(),
                          new IgnoringDiagConsumer());
    Diags = &*IgnoringDiags;
  }

  // The key paths of diagnostic options defined in Options.td start with
  // "DiagnosticOpts->". Let's provide the expected variable name and type.
  DiagnosticOptions *DiagnosticOpts = &Opts;
  bool Success = true;

#define DIAG_OPTION_WITH_MARSHALLING(                                          \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(Args, *Diags, Success, ID, FLAGS, PARAM,       \
                                SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,          \
                                IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER,      \
                                MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef DIAG_OPTION_WITH_MARSHALLING

  llvm::sys::Process::UseANSIEscapeCodes(Opts.UseANSIEscapeCodes);

  if (Arg *A =
          Args.getLastArg(OPT_diagnostic_serialized_file, OPT__serialize_diags))
    Opts.DiagnosticSerializationFile = A->getValue();
  Opts.ShowColors = parseShowColorsArgs(Args, DefaultDiagColor);

  if (Args.getLastArgValue(OPT_fdiagnostics_format) == "msvc-fallback")
    Opts.CLFallbackMode = true;

  Opts.VerifyDiagnostics = Args.hasArg(OPT_verify) || Args.hasArg(OPT_verify_EQ);
  if (Args.hasArg(OPT_verify))
    Opts.VerifyPrefixes.push_back("expected");
  // Keep VerifyPrefixes in its original order for the sake of diagnostics, and
  // then sort it to prepare for fast lookup using std::binary_search.
  if (!checkVerifyPrefixes(Opts.VerifyPrefixes, *Diags)) {
    Opts.VerifyDiagnostics = false;
    Success = false;
  }
  else
    llvm::sort(Opts.VerifyPrefixes);
  DiagnosticLevelMask DiagMask = DiagnosticLevelMask::None;
  Success &= parseDiagnosticLevelMask("-verify-ignore-unexpected=",
    Args.getAllArgValues(OPT_verify_ignore_unexpected_EQ),
    *Diags, DiagMask);
  if (Args.hasArg(OPT_verify_ignore_unexpected))
    DiagMask = DiagnosticLevelMask::All;
  Opts.setVerifyIgnoreUnexpected(DiagMask);
  if (Opts.TabStop == 0 || Opts.TabStop > DiagnosticOptions::MaxTabStop) {
    Opts.TabStop = DiagnosticOptions::DefaultTabStop;
    Diags->Report(diag::warn_ignoring_ftabstop_value)
        << Opts.TabStop << DiagnosticOptions::DefaultTabStop;
  }

  addDiagnosticArgs(Args, OPT_W_Group, OPT_W_value_Group, Opts.Warnings);
  addDiagnosticArgs(Args, OPT_R_Group, OPT_R_value_Group, Opts.Remarks);

  return Success;
}

/// Parse the argument to the -ftest-module-file-extension
/// command-line argument.
///
/// \returns true on error, false on success.
static bool parseTestModuleFileExtensionArg(StringRef Arg,
                                            std::string &BlockName,
                                            unsigned &MajorVersion,
                                            unsigned &MinorVersion,
                                            bool &Hashed,
                                            std::string &UserInfo) {
  SmallVector<StringRef, 5> Args;
  Arg.split(Args, ':', 5);
  if (Args.size() < 5)
    return true;

  BlockName = std::string(Args[0]);
  if (Args[1].getAsInteger(10, MajorVersion)) return true;
  if (Args[2].getAsInteger(10, MinorVersion)) return true;
  if (Args[3].getAsInteger(2, Hashed)) return true;
  if (Args.size() > 4)
    UserInfo = std::string(Args[4]);
  return false;
}

static InputKind ParseFrontendArgs(FrontendOptions &Opts, ArgList &Args,
                                   DiagnosticsEngine &Diags,
                                   bool &IsHeaderFile) {
  Opts.ProgramAction = frontend::ParseSyntaxOnly;
  if (const Arg *A = Args.getLastArg(OPT_Action_Group)) {
    switch (A->getOption().getID()) {
    default:
      llvm_unreachable("Invalid option in group!");
    case OPT_ast_list:
      Opts.ProgramAction = frontend::ASTDeclList; break;
    case OPT_ast_dump_all_EQ:
    case OPT_ast_dump_EQ: {
      unsigned Val = llvm::StringSwitch<unsigned>(A->getValue())
                         .CaseLower("default", ADOF_Default)
                         .CaseLower("json", ADOF_JSON)
                         .Default(std::numeric_limits<unsigned>::max());

      if (Val != std::numeric_limits<unsigned>::max())
        Opts.ASTDumpFormat = static_cast<ASTDumpOutputFormat>(Val);
      else {
        Diags.Report(diag::err_drv_invalid_value)
            << A->getAsString(Args) << A->getValue();
        Opts.ASTDumpFormat = ADOF_Default;
      }
      LLVM_FALLTHROUGH;
    }
    case OPT_ast_dump:
    case OPT_ast_dump_all:
    case OPT_ast_dump_lookups:
    case OPT_ast_dump_decl_types:
      Opts.ProgramAction = frontend::ASTDump; break;
    case OPT_ast_print:
      Opts.ProgramAction = frontend::ASTPrint; break;
    case OPT_ast_view:
      Opts.ProgramAction = frontend::ASTView; break;
    case OPT_compiler_options_dump:
      Opts.ProgramAction = frontend::DumpCompilerOptions; break;
    case OPT_dump_raw_tokens:
      Opts.ProgramAction = frontend::DumpRawTokens; break;
    case OPT_dump_tokens:
      Opts.ProgramAction = frontend::DumpTokens; break;
    case OPT_S:
      Opts.ProgramAction = frontend::EmitAssembly; break;
    case OPT_emit_llvm_bc:
      Opts.ProgramAction = frontend::EmitBC; break;
    case OPT_emit_html:
      Opts.ProgramAction = frontend::EmitHTML; break;
    case OPT_emit_llvm:
      Opts.ProgramAction = frontend::EmitLLVM; break;
    case OPT_emit_llvm_only:
      Opts.ProgramAction = frontend::EmitLLVMOnly; break;
    case OPT_emit_codegen_only:
      Opts.ProgramAction = frontend::EmitCodeGenOnly; break;
    case OPT_emit_obj:
      Opts.ProgramAction = frontend::EmitObj; break;
    case OPT_fixit_EQ:
      Opts.FixItSuffix = A->getValue();
      LLVM_FALLTHROUGH;
    case OPT_fixit:
      Opts.ProgramAction = frontend::FixIt; break;
    case OPT_emit_module:
      Opts.ProgramAction = frontend::GenerateModule; break;
    case OPT_emit_module_interface:
      Opts.ProgramAction = frontend::GenerateModuleInterface; break;
    case OPT_emit_header_module:
      Opts.ProgramAction = frontend::GenerateHeaderModule; break;
    case OPT_emit_pch:
      Opts.ProgramAction = frontend::GeneratePCH; break;
    case OPT_emit_interface_stubs: {
      StringRef ArgStr =
          Args.hasArg(OPT_interface_stub_version_EQ)
              ? Args.getLastArgValue(OPT_interface_stub_version_EQ)
              : "experimental-ifs-v2";
      if (ArgStr == "experimental-yaml-elf-v1" ||
          ArgStr == "experimental-ifs-v1" ||
          ArgStr == "experimental-tapi-elf-v1") {
        std::string ErrorMessage =
            "Invalid interface stub format: " + ArgStr.str() +
            " is deprecated.";
        Diags.Report(diag::err_drv_invalid_value)
            << "Must specify a valid interface stub format type, ie: "
               "-interface-stub-version=experimental-ifs-v2"
            << ErrorMessage;
      } else if (!ArgStr.startswith("experimental-ifs-")) {
        std::string ErrorMessage =
            "Invalid interface stub format: " + ArgStr.str() + ".";
        Diags.Report(diag::err_drv_invalid_value)
            << "Must specify a valid interface stub format type, ie: "
               "-interface-stub-version=experimental-ifs-v2"
            << ErrorMessage;
      } else {
        Opts.ProgramAction = frontend::GenerateInterfaceStubs;
      }
      break;
    }
    case OPT_init_only:
      Opts.ProgramAction = frontend::InitOnly; break;
    case OPT_fsyntax_only:
      Opts.ProgramAction = frontend::ParseSyntaxOnly; break;
    case OPT_module_file_info:
      Opts.ProgramAction = frontend::ModuleFileInfo; break;
    case OPT_verify_pch:
      Opts.ProgramAction = frontend::VerifyPCH; break;
    case OPT_print_preamble:
      Opts.ProgramAction = frontend::PrintPreamble; break;
    case OPT_E:
      Opts.ProgramAction = frontend::PrintPreprocessedInput; break;
    case OPT_templight_dump:
      Opts.ProgramAction = frontend::TemplightDump; break;
    case OPT_rewrite_macros:
      Opts.ProgramAction = frontend::RewriteMacros; break;
    case OPT_rewrite_objc:
      Opts.ProgramAction = frontend::RewriteObjC; break;
    case OPT_rewrite_test:
      Opts.ProgramAction = frontend::RewriteTest; break;
    case OPT_analyze:
      Opts.ProgramAction = frontend::RunAnalysis; break;
    case OPT_migrate:
      Opts.ProgramAction = frontend::MigrateSource; break;
    case OPT_Eonly:
      Opts.ProgramAction = frontend::RunPreprocessorOnly; break;
    case OPT_print_dependency_directives_minimized_source:
      Opts.ProgramAction =
          frontend::PrintDependencyDirectivesSourceMinimizerOutput;
      break;
    }
  }

  if (const Arg* A = Args.getLastArg(OPT_plugin)) {
    Opts.Plugins.emplace_back(A->getValue(0));
    Opts.ProgramAction = frontend::PluginAction;
    Opts.ActionName = A->getValue();
  }
  for (const auto *AA : Args.filtered(OPT_plugin_arg))
    Opts.PluginArgs[AA->getValue(0)].emplace_back(AA->getValue(1));

  for (const std::string &Arg :
         Args.getAllArgValues(OPT_ftest_module_file_extension_EQ)) {
    std::string BlockName;
    unsigned MajorVersion;
    unsigned MinorVersion;
    bool Hashed;
    std::string UserInfo;
    if (parseTestModuleFileExtensionArg(Arg, BlockName, MajorVersion,
                                        MinorVersion, Hashed, UserInfo)) {
      Diags.Report(diag::err_test_module_file_extension_format) << Arg;

      continue;
    }

    // Add the testing module file extension.
    Opts.ModuleFileExtensions.push_back(
        std::make_shared<TestModuleFileExtension>(
            BlockName, MajorVersion, MinorVersion, Hashed, UserInfo));
  }

  if (const Arg *A = Args.getLastArg(OPT_code_completion_at)) {
    Opts.CodeCompletionAt =
      ParsedSourceLocation::FromString(A->getValue());
    if (Opts.CodeCompletionAt.FileName.empty())
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue();
  }

  Opts.Plugins = Args.getAllArgValues(OPT_load);
  Opts.ASTDumpDecls = Args.hasArg(OPT_ast_dump, OPT_ast_dump_EQ);
  Opts.ASTDumpAll = Args.hasArg(OPT_ast_dump_all, OPT_ast_dump_all_EQ);
  // Only the -fmodule-file=<file> form.
  for (const auto *A : Args.filtered(OPT_fmodule_file)) {
    StringRef Val = A->getValue();
    if (Val.find('=') == StringRef::npos)
      Opts.ModuleFiles.push_back(std::string(Val));
  }

  if (Opts.ProgramAction != frontend::GenerateModule && Opts.IsSystemModule)
    Diags.Report(diag::err_drv_argument_only_allowed_with) << "-fsystem-module"
                                                           << "-emit-module";

  if (Args.hasArg(OPT_aux_target_cpu))
    Opts.AuxTargetCPU = std::string(Args.getLastArgValue(OPT_aux_target_cpu));
  if (Args.hasArg(OPT_aux_target_feature))
    Opts.AuxTargetFeatures = Args.getAllArgValues(OPT_aux_target_feature);

  if (Opts.ARCMTAction != FrontendOptions::ARCMT_None &&
      Opts.ObjCMTAction != FrontendOptions::ObjCMT_None) {
    Diags.Report(diag::err_drv_argument_not_allowed_with)
      << "ARC migration" << "ObjC migration";
  }

  InputKind DashX(Language::Unknown);
  if (const Arg *A = Args.getLastArg(OPT_x)) {
    StringRef XValue = A->getValue();

    // Parse suffixes: '<lang>(-header|[-module-map][-cpp-output])'.
    // FIXME: Supporting '<lang>-header-cpp-output' would be useful.
    bool Preprocessed = XValue.consume_back("-cpp-output");
    bool ModuleMap = XValue.consume_back("-module-map");
    IsHeaderFile = !Preprocessed && !ModuleMap &&
                   XValue != "precompiled-header" &&
                   XValue.consume_back("-header");

    // Principal languages.
    DashX = llvm::StringSwitch<InputKind>(XValue)
                .Case("c", Language::C)
                .Case("cl", Language::OpenCL)
                .Case("cuda", Language::CUDA)
                .Case("hip", Language::HIP)
                .Case("c++", Language::CXX)
                .Case("objective-c", Language::ObjC)
                .Case("objective-c++", Language::ObjCXX)
                .Case("renderscript", Language::RenderScript)
                .Default(Language::Unknown);

    // "objc[++]-cpp-output" is an acceptable synonym for
    // "objective-c[++]-cpp-output".
    if (DashX.isUnknown() && Preprocessed && !IsHeaderFile && !ModuleMap)
      DashX = llvm::StringSwitch<InputKind>(XValue)
                  .Case("objc", Language::ObjC)
                  .Case("objc++", Language::ObjCXX)
                  .Default(Language::Unknown);

    // Some special cases cannot be combined with suffixes.
    if (DashX.isUnknown() && !Preprocessed && !ModuleMap && !IsHeaderFile)
      DashX = llvm::StringSwitch<InputKind>(XValue)
                  .Case("cpp-output", InputKind(Language::C).getPreprocessed())
                  .Case("assembler-with-cpp", Language::Asm)
                  .Cases("ast", "pcm", "precompiled-header",
                         InputKind(Language::Unknown, InputKind::Precompiled))
                  .Case("ir", Language::LLVM_IR)
                  .Default(Language::Unknown);

    if (DashX.isUnknown())
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue();

    if (Preprocessed)
      DashX = DashX.getPreprocessed();
    if (ModuleMap)
      DashX = DashX.withFormat(InputKind::ModuleMap);
  }

  // '-' is the default input if none is given.
  std::vector<std::string> Inputs = Args.getAllArgValues(OPT_INPUT);
  Opts.Inputs.clear();
  if (Inputs.empty())
    Inputs.push_back("-");
  for (unsigned i = 0, e = Inputs.size(); i != e; ++i) {
    InputKind IK = DashX;
    if (IK.isUnknown()) {
      IK = FrontendOptions::getInputKindForExtension(
        StringRef(Inputs[i]).rsplit('.').second);
      // FIXME: Warn on this?
      if (IK.isUnknown())
        IK = Language::C;
      // FIXME: Remove this hack.
      if (i == 0)
        DashX = IK;
    }

    bool IsSystem = false;

    // The -emit-module action implicitly takes a module map.
    if (Opts.ProgramAction == frontend::GenerateModule &&
        IK.getFormat() == InputKind::Source) {
      IK = IK.withFormat(InputKind::ModuleMap);
      IsSystem = Opts.IsSystemModule;
    }

    Opts.Inputs.emplace_back(std::move(Inputs[i]), IK, IsSystem);
  }

  return DashX;
}

std::string CompilerInvocation::GetResourcesPath(const char *Argv0,
                                                 void *MainAddr) {
  std::string ClangExecutable =
      llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
  return Driver::GetResourcesPath(ClangExecutable, CLANG_RESOURCE_DIR);
}

static void GenerateHeaderSearchArgs(const HeaderSearchOptions &Opts,
                                     SmallVectorImpl<const char *> &Args,
                                     CompilerInvocation::StringAllocator SA) {
  const HeaderSearchOptions *HeaderSearchOpts = &Opts;
#define HEADER_SEARCH_OPTION_WITH_MARSHALLING(                                 \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  GENERATE_OPTION_WITH_MARSHALLING(                                            \
      Args, SA, KIND, FLAGS, SPELLING, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE,    \
      IMPLIED_CHECK, IMPLIED_VALUE, DENORMALIZER, EXTRACTOR, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef HEADER_SEARCH_OPTION_WITH_MARSHALLING
}

static void ParseHeaderSearchArgs(HeaderSearchOptions &Opts, ArgList &Args,
                                  DiagnosticsEngine &Diags,
                                  const std::string &WorkingDir) {
  HeaderSearchOptions *HeaderSearchOpts = &Opts;
  bool Success = true;

#define HEADER_SEARCH_OPTION_WITH_MARSHALLING(                                 \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(Args, Diags, Success, ID, FLAGS, PARAM,        \
                                SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,          \
                                IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER,      \
                                MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef HEADER_SEARCH_OPTION_WITH_MARSHALLING

  if (const Arg *A = Args.getLastArg(OPT_stdlib_EQ))
    Opts.UseLibcxx = (strcmp(A->getValue(), "libc++") == 0);

  // Canonicalize -fmodules-cache-path before storing it.
  SmallString<128> P(Args.getLastArgValue(OPT_fmodules_cache_path));
  if (!(P.empty() || llvm::sys::path::is_absolute(P))) {
    if (WorkingDir.empty())
      llvm::sys::fs::make_absolute(P);
    else
      llvm::sys::fs::make_absolute(WorkingDir, P);
  }
  llvm::sys::path::remove_dots(P);
  Opts.ModuleCachePath = std::string(P.str());

  // Only the -fmodule-file=<name>=<file> form.
  for (const auto *A : Args.filtered(OPT_fmodule_file)) {
    StringRef Val = A->getValue();
    if (Val.find('=') != StringRef::npos){
      auto Split = Val.split('=');
      Opts.PrebuiltModuleFiles.insert(
          {std::string(Split.first), std::string(Split.second)});
    }
  }
  for (const auto *A : Args.filtered(OPT_fprebuilt_module_path))
    Opts.AddPrebuiltModulePath(A->getValue());

  for (const auto *A : Args.filtered(OPT_fmodules_ignore_macro)) {
    StringRef MacroDef = A->getValue();
    Opts.ModulesIgnoreMacros.insert(
        llvm::CachedHashString(MacroDef.split('=').first));
  }

  // Add -I..., -F..., and -index-header-map options in order.
  bool IsIndexHeaderMap = false;
  bool IsSysrootSpecified =
      Args.hasArg(OPT__sysroot_EQ) || Args.hasArg(OPT_isysroot);
  for (const auto *A : Args.filtered(OPT_I, OPT_F, OPT_index_header_map)) {
    if (A->getOption().matches(OPT_index_header_map)) {
      // -index-header-map applies to the next -I or -F.
      IsIndexHeaderMap = true;
      continue;
    }

    frontend::IncludeDirGroup Group =
        IsIndexHeaderMap ? frontend::IndexHeaderMap : frontend::Angled;

    bool IsFramework = A->getOption().matches(OPT_F);
    std::string Path = A->getValue();

    if (IsSysrootSpecified && !IsFramework && A->getValue()[0] == '=') {
      SmallString<32> Buffer;
      llvm::sys::path::append(Buffer, Opts.Sysroot,
                              llvm::StringRef(A->getValue()).substr(1));
      Path = std::string(Buffer.str());
    }

    Opts.AddPath(Path, Group, IsFramework,
                 /*IgnoreSysroot*/ true);
    IsIndexHeaderMap = false;
  }

  // Add -iprefix/-iwithprefix/-iwithprefixbefore options.
  StringRef Prefix = ""; // FIXME: This isn't the correct default prefix.
  for (const auto *A :
       Args.filtered(OPT_iprefix, OPT_iwithprefix, OPT_iwithprefixbefore)) {
    if (A->getOption().matches(OPT_iprefix))
      Prefix = A->getValue();
    else if (A->getOption().matches(OPT_iwithprefix))
      Opts.AddPath(Prefix.str() + A->getValue(), frontend::After, false, true);
    else
      Opts.AddPath(Prefix.str() + A->getValue(), frontend::Angled, false, true);
  }

  for (const auto *A : Args.filtered(OPT_idirafter))
    Opts.AddPath(A->getValue(), frontend::After, false, true);
  for (const auto *A : Args.filtered(OPT_iquote))
    Opts.AddPath(A->getValue(), frontend::Quoted, false, true);
  for (const auto *A : Args.filtered(OPT_isystem, OPT_iwithsysroot))
    Opts.AddPath(A->getValue(), frontend::System, false,
                 !A->getOption().matches(OPT_iwithsysroot));
  for (const auto *A : Args.filtered(OPT_iframework))
    Opts.AddPath(A->getValue(), frontend::System, true, true);
  for (const auto *A : Args.filtered(OPT_iframeworkwithsysroot))
    Opts.AddPath(A->getValue(), frontend::System, /*IsFramework=*/true,
                 /*IgnoreSysRoot=*/false);

  // Add the paths for the various language specific isystem flags.
  for (const auto *A : Args.filtered(OPT_c_isystem))
    Opts.AddPath(A->getValue(), frontend::CSystem, false, true);
  for (const auto *A : Args.filtered(OPT_cxx_isystem))
    Opts.AddPath(A->getValue(), frontend::CXXSystem, false, true);
  for (const auto *A : Args.filtered(OPT_objc_isystem))
    Opts.AddPath(A->getValue(), frontend::ObjCSystem, false,true);
  for (const auto *A : Args.filtered(OPT_objcxx_isystem))
    Opts.AddPath(A->getValue(), frontend::ObjCXXSystem, false, true);

  // Add the internal paths from a driver that detects standard include paths.
  for (const auto *A :
       Args.filtered(OPT_internal_isystem, OPT_internal_externc_isystem)) {
    frontend::IncludeDirGroup Group = frontend::System;
    if (A->getOption().matches(OPT_internal_externc_isystem))
      Group = frontend::ExternCSystem;
    Opts.AddPath(A->getValue(), Group, false, true);
  }

  // Add the path prefixes which are implicitly treated as being system headers.
  for (const auto *A :
       Args.filtered(OPT_system_header_prefix, OPT_no_system_header_prefix))
    Opts.AddSystemHeaderPrefix(
        A->getValue(), A->getOption().matches(OPT_system_header_prefix));

  for (const auto *A : Args.filtered(OPT_ivfsoverlay))
    Opts.AddVFSOverlayFile(A->getValue());
}

void CompilerInvocation::setLangDefaults(LangOptions &Opts, InputKind IK,
                                         const llvm::Triple &T,
                                         std::vector<std::string> &Includes,
                                         LangStandard::Kind LangStd) {
  // Set some properties which depend solely on the input kind; it would be nice
  // to move these to the language standard, and have the driver resolve the
  // input kind + language standard.
  //
  // FIXME: Perhaps a better model would be for a single source file to have
  // multiple language standards (C / C++ std, ObjC std, OpenCL std, OpenMP std)
  // simultaneously active?
  if (IK.getLanguage() == Language::Asm) {
    Opts.AsmPreprocessor = 1;
  } else if (IK.isObjectiveC()) {
    Opts.ObjC = 1;
  }

  if (LangStd == LangStandard::lang_unspecified) {
    // Based on the base language, pick one.
    switch (IK.getLanguage()) {
    case Language::Unknown:
    case Language::LLVM_IR:
      llvm_unreachable("Invalid input kind!");
    case Language::OpenCL:
      LangStd = LangStandard::lang_opencl10;
      break;
    case Language::CUDA:
      LangStd = LangStandard::lang_cuda;
      break;
    case Language::Asm:
    case Language::C:
#if defined(CLANG_DEFAULT_STD_C)
      LangStd = CLANG_DEFAULT_STD_C;
#else
      // The PS4 uses C99 as the default C standard.
      if (T.isPS4())
        LangStd = LangStandard::lang_gnu99;
      else
        LangStd = LangStandard::lang_gnu17;
#endif
      break;
    case Language::ObjC:
#if defined(CLANG_DEFAULT_STD_C)
      LangStd = CLANG_DEFAULT_STD_C;
#else
      LangStd = LangStandard::lang_gnu11;
#endif
      break;
    case Language::CXX:
    case Language::ObjCXX:
#if defined(CLANG_DEFAULT_STD_CXX)
      LangStd = CLANG_DEFAULT_STD_CXX;
#else
      LangStd = LangStandard::lang_gnucxx14;
#endif
      break;
    case Language::RenderScript:
      LangStd = LangStandard::lang_c99;
      break;
    case Language::HIP:
      LangStd = LangStandard::lang_hip;
      break;
    }
  }

  const LangStandard &Std = LangStandard::getLangStandardForKind(LangStd);
  Opts.LangStd = LangStd;
  Opts.LineComment = Std.hasLineComments();
  Opts.C99 = Std.isC99();
  Opts.C11 = Std.isC11();
  Opts.C17 = Std.isC17();
  Opts.C2x = Std.isC2x();
  Opts.CPlusPlus = Std.isCPlusPlus();
  Opts.CPlusPlus11 = Std.isCPlusPlus11();
  Opts.CPlusPlus14 = Std.isCPlusPlus14();
  Opts.CPlusPlus17 = Std.isCPlusPlus17();
  Opts.CPlusPlus20 = Std.isCPlusPlus20();
  Opts.CPlusPlus2b = Std.isCPlusPlus2b();
  Opts.GNUMode = Std.isGNUMode();
  Opts.GNUCVersion = 0;
  Opts.HexFloats = Std.hasHexFloats();
  Opts.ImplicitInt = Std.hasImplicitInt();

  Opts.CPlusPlusModules = Opts.CPlusPlus20;

  // Set OpenCL Version.
  Opts.OpenCL = Std.isOpenCL();
  if (LangStd == LangStandard::lang_opencl10)
    Opts.OpenCLVersion = 100;
  else if (LangStd == LangStandard::lang_opencl11)
    Opts.OpenCLVersion = 110;
  else if (LangStd == LangStandard::lang_opencl12)
    Opts.OpenCLVersion = 120;
  else if (LangStd == LangStandard::lang_opencl20)
    Opts.OpenCLVersion = 200;
  else if (LangStd == LangStandard::lang_opencl30)
    Opts.OpenCLVersion = 300;
  else if (LangStd == LangStandard::lang_openclcpp)
    Opts.OpenCLCPlusPlusVersion = 100;

  // OpenCL has some additional defaults.
  if (Opts.OpenCL) {
    Opts.AltiVec = 0;
    Opts.ZVector = 0;
    Opts.setDefaultFPContractMode(LangOptions::FPM_On);
    Opts.OpenCLCPlusPlus = Opts.CPlusPlus;

    // Include default header file for OpenCL.
    if (Opts.IncludeDefaultHeader) {
      if (Opts.DeclareOpenCLBuiltins) {
        // Only include base header file for builtin types and constants.
        Includes.push_back("opencl-c-base.h");
      } else {
        Includes.push_back("opencl-c.h");
      }
    }
  }

  Opts.HIP = IK.getLanguage() == Language::HIP;
  Opts.CUDA = IK.getLanguage() == Language::CUDA || Opts.HIP;
  if (Opts.HIP) {
    // HIP toolchain does not support 'Fast' FPOpFusion in backends since it
    // fuses multiplication/addition instructions without contract flag from
    // device library functions in LLVM bitcode, which causes accuracy loss in
    // certain math functions, e.g. tan(-1e20) becomes -0.933 instead of 0.8446.
    // For device library functions in bitcode to work, 'Strict' or 'Standard'
    // FPOpFusion options in backends is needed. Therefore 'fast-honor-pragmas'
    // FP contract option is used to allow fuse across statements in frontend
    // whereas respecting contract flag in backend.
    Opts.setDefaultFPContractMode(LangOptions::FPM_FastHonorPragmas);
  } else if (Opts.CUDA) {
    // Allow fuse across statements disregarding pragmas.
    Opts.setDefaultFPContractMode(LangOptions::FPM_Fast);
  }

  Opts.RenderScript = IK.getLanguage() == Language::RenderScript;

  // OpenCL and C++ both have bool, true, false keywords.
  Opts.Bool = Opts.OpenCL || Opts.CPlusPlus;

  // OpenCL has half keyword
  Opts.Half = Opts.OpenCL;
}

/// Check if input file kind and language standard are compatible.
static bool IsInputCompatibleWithStandard(InputKind IK,
                                          const LangStandard &S) {
  switch (IK.getLanguage()) {
  case Language::Unknown:
  case Language::LLVM_IR:
    llvm_unreachable("should not parse language flags for this input");

  case Language::C:
  case Language::ObjC:
  case Language::RenderScript:
    return S.getLanguage() == Language::C;

  case Language::OpenCL:
    return S.getLanguage() == Language::OpenCL;

  case Language::CXX:
  case Language::ObjCXX:
    return S.getLanguage() == Language::CXX;

  case Language::CUDA:
    // FIXME: What -std= values should be permitted for CUDA compilations?
    return S.getLanguage() == Language::CUDA ||
           S.getLanguage() == Language::CXX;

  case Language::HIP:
    return S.getLanguage() == Language::CXX || S.getLanguage() == Language::HIP;

  case Language::Asm:
    // Accept (and ignore) all -std= values.
    // FIXME: The -std= value is not ignored; it affects the tokenization
    // and preprocessing rules if we're preprocessing this asm input.
    return true;
  }

  llvm_unreachable("unexpected input language");
}

/// Get language name for given input kind.
static const StringRef GetInputKindName(InputKind IK) {
  switch (IK.getLanguage()) {
  case Language::C:
    return "C";
  case Language::ObjC:
    return "Objective-C";
  case Language::CXX:
    return "C++";
  case Language::ObjCXX:
    return "Objective-C++";
  case Language::OpenCL:
    return "OpenCL";
  case Language::CUDA:
    return "CUDA";
  case Language::RenderScript:
    return "RenderScript";
  case Language::HIP:
    return "HIP";

  case Language::Asm:
    return "Asm";
  case Language::LLVM_IR:
    return "LLVM IR";

  case Language::Unknown:
    break;
  }
  llvm_unreachable("unknown input language");
}

static void GenerateLangArgs(const LangOptions &Opts,
                             SmallVectorImpl<const char *> &Args,
                             CompilerInvocation::StringAllocator SA) {
  if (Opts.IncludeDefaultHeader)
    Args.push_back(SA(GetOptName(OPT_finclude_default_header)));
  if (Opts.DeclareOpenCLBuiltins)
    Args.push_back(SA(GetOptName(OPT_fdeclare_opencl_builtins)));
}

void CompilerInvocation::ParseLangArgs(LangOptions &Opts, ArgList &Args,
                                       InputKind IK, const llvm::Triple &T,
                                       std::vector<std::string> &Includes,
                                       DiagnosticsEngine &Diags) {
  // FIXME: Cleanup per-file based stuff.
  LangStandard::Kind LangStd = LangStandard::lang_unspecified;
  if (const Arg *A = Args.getLastArg(OPT_std_EQ)) {
    LangStd = LangStandard::getLangKind(A->getValue());
    if (LangStd == LangStandard::lang_unspecified) {
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue();
      // Report supported standards with short description.
      for (unsigned KindValue = 0;
           KindValue != LangStandard::lang_unspecified;
           ++KindValue) {
        const LangStandard &Std = LangStandard::getLangStandardForKind(
          static_cast<LangStandard::Kind>(KindValue));
        if (IsInputCompatibleWithStandard(IK, Std)) {
          auto Diag = Diags.Report(diag::note_drv_use_standard);
          Diag << Std.getName() << Std.getDescription();
          unsigned NumAliases = 0;
#define LANGSTANDARD(id, name, lang, desc, features)
#define LANGSTANDARD_ALIAS(id, alias) \
          if (KindValue == LangStandard::lang_##id) ++NumAliases;
#define LANGSTANDARD_ALIAS_DEPR(id, alias)
#include "clang/Basic/LangStandards.def"
          Diag << NumAliases;
#define LANGSTANDARD(id, name, lang, desc, features)
#define LANGSTANDARD_ALIAS(id, alias) \
          if (KindValue == LangStandard::lang_##id) Diag << alias;
#define LANGSTANDARD_ALIAS_DEPR(id, alias)
#include "clang/Basic/LangStandards.def"
        }
      }
    } else {
      // Valid standard, check to make sure language and standard are
      // compatible.
      const LangStandard &Std = LangStandard::getLangStandardForKind(LangStd);
      if (!IsInputCompatibleWithStandard(IK, Std)) {
        Diags.Report(diag::err_drv_argument_not_allowed_with)
          << A->getAsString(Args) << GetInputKindName(IK);
      }
    }
  }

  // -cl-std only applies for OpenCL language standards.
  // Override the -std option in this case.
  if (const Arg *A = Args.getLastArg(OPT_cl_std_EQ)) {
    LangStandard::Kind OpenCLLangStd
      = llvm::StringSwitch<LangStandard::Kind>(A->getValue())
        .Cases("cl", "CL", LangStandard::lang_opencl10)
        .Cases("cl1.0", "CL1.0", LangStandard::lang_opencl10)
        .Cases("cl1.1", "CL1.1", LangStandard::lang_opencl11)
        .Cases("cl1.2", "CL1.2", LangStandard::lang_opencl12)
        .Cases("cl2.0", "CL2.0", LangStandard::lang_opencl20)
        .Cases("cl3.0", "CL3.0", LangStandard::lang_opencl30)
        .Cases("clc++", "CLC++", LangStandard::lang_openclcpp)
        .Default(LangStandard::lang_unspecified);

    if (OpenCLLangStd == LangStandard::lang_unspecified) {
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue();
    }
    else
      LangStd = OpenCLLangStd;
  }

  // These need to be parsed now. They are used to set OpenCL defaults.
  Opts.IncludeDefaultHeader = Args.hasArg(OPT_finclude_default_header);
  Opts.DeclareOpenCLBuiltins = Args.hasArg(OPT_fdeclare_opencl_builtins);

  CompilerInvocation::setLangDefaults(Opts, IK, T, Includes, LangStd);

  // The key paths of codegen options defined in Options.td start with
  // "LangOpts->". Let's provide the expected variable name and type.
  LangOptions *LangOpts = &Opts;
  bool Success = true;

#define LANG_OPTION_WITH_MARSHALLING(                                          \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(Args, Diags, Success, ID, FLAGS, PARAM,        \
                                SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,          \
                                IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER,      \
                                MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef LANG_OPTION_WITH_MARSHALLING

  if (const Arg *A = Args.getLastArg(OPT_fcf_protection_EQ)) {
    StringRef Name = A->getValue();
    if (Name == "full" || Name == "branch") {
      Opts.CFProtectionBranch = 1;
    }
  }

  if (Opts.ObjC) {
    if (Arg *arg = Args.getLastArg(OPT_fobjc_runtime_EQ)) {
      StringRef value = arg->getValue();
      if (Opts.ObjCRuntime.tryParse(value))
        Diags.Report(diag::err_drv_unknown_objc_runtime) << value;
    }

    if (Args.hasArg(OPT_fobjc_gc_only))
      Opts.setGC(LangOptions::GCOnly);
    else if (Args.hasArg(OPT_fobjc_gc))
      Opts.setGC(LangOptions::HybridGC);
    else if (Args.hasArg(OPT_fobjc_arc)) {
      Opts.ObjCAutoRefCount = 1;
      if (!Opts.ObjCRuntime.allowsARC())
        Diags.Report(diag::err_arc_unsupported_on_runtime);
    }

    // ObjCWeakRuntime tracks whether the runtime supports __weak, not
    // whether the feature is actually enabled.  This is predominantly
    // determined by -fobjc-runtime, but we allow it to be overridden
    // from the command line for testing purposes.
    if (Args.hasArg(OPT_fobjc_runtime_has_weak))
      Opts.ObjCWeakRuntime = 1;
    else
      Opts.ObjCWeakRuntime = Opts.ObjCRuntime.allowsWeak();

    // ObjCWeak determines whether __weak is actually enabled.
    // Note that we allow -fno-objc-weak to disable this even in ARC mode.
    if (auto weakArg = Args.getLastArg(OPT_fobjc_weak, OPT_fno_objc_weak)) {
      if (!weakArg->getOption().matches(OPT_fobjc_weak)) {
        assert(!Opts.ObjCWeak);
      } else if (Opts.getGC() != LangOptions::NonGC) {
        Diags.Report(diag::err_objc_weak_with_gc);
      } else if (!Opts.ObjCWeakRuntime) {
        Diags.Report(diag::err_objc_weak_unsupported);
      } else {
        Opts.ObjCWeak = 1;
      }
    } else if (Opts.ObjCAutoRefCount) {
      Opts.ObjCWeak = Opts.ObjCWeakRuntime;
    }

    if (Args.hasArg(OPT_fobjc_subscripting_legacy_runtime))
      Opts.ObjCSubscriptingLegacyRuntime =
        (Opts.ObjCRuntime.getKind() == ObjCRuntime::FragileMacOSX);
  }

  if (Arg *A = Args.getLastArg(options::OPT_fgnuc_version_EQ)) {
    // Check that the version has 1 to 3 components and the minor and patch
    // versions fit in two decimal digits.
    VersionTuple GNUCVer;
    bool Invalid = GNUCVer.tryParse(A->getValue());
    unsigned Major = GNUCVer.getMajor();
    unsigned Minor = GNUCVer.getMinor().getValueOr(0);
    unsigned Patch = GNUCVer.getSubminor().getValueOr(0);
    if (Invalid || GNUCVer.getBuild() || Minor >= 100 || Patch >= 100) {
      Diags.Report(diag::err_drv_invalid_value)
          << A->getAsString(Args) << A->getValue();
    }
    Opts.GNUCVersion = Major * 100 * 100 + Minor * 100 + Patch;
  }

  if (Args.hasArg(OPT_ftrapv)) {
    Opts.setSignedOverflowBehavior(LangOptions::SOB_Trapping);
    // Set the handler, if one is specified.
    Opts.OverflowHandler =
        std::string(Args.getLastArgValue(OPT_ftrapv_handler));
  }
  else if (Args.hasArg(OPT_fwrapv))
    Opts.setSignedOverflowBehavior(LangOptions::SOB_Defined);

  Opts.MSCompatibilityVersion = 0;
  if (const Arg *A = Args.getLastArg(OPT_fms_compatibility_version)) {
    VersionTuple VT;
    if (VT.tryParse(A->getValue()))
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args)
                                                << A->getValue();
    Opts.MSCompatibilityVersion = VT.getMajor() * 10000000 +
                                  VT.getMinor().getValueOr(0) * 100000 +
                                  VT.getSubminor().getValueOr(0);
  }

  // Mimicking gcc's behavior, trigraphs are only enabled if -trigraphs
  // is specified, or -std is set to a conforming mode.
  // Trigraphs are disabled by default in c++1z onwards.
  // For z/OS, trigraphs are enabled by default (without regard to the above).
  Opts.Trigraphs =
      (!Opts.GNUMode && !Opts.MSVCCompat && !Opts.CPlusPlus17) || T.isOSzOS();
  Opts.Trigraphs =
      Args.hasFlag(OPT_ftrigraphs, OPT_fno_trigraphs, Opts.Trigraphs);

  Opts.Blocks = Args.hasArg(OPT_fblocks) || (Opts.OpenCL
    && Opts.OpenCLVersion == 200);

  Opts.ConvergentFunctions = Opts.OpenCL || (Opts.CUDA && Opts.CUDAIsDevice) ||
                             Opts.SYCLIsDevice ||
                             Args.hasArg(OPT_fconvergent_functions);

  Opts.NoBuiltin = Args.hasArg(OPT_fno_builtin) || Opts.Freestanding;
  if (!Opts.NoBuiltin)
    getAllNoBuiltinFuncValues(Args, Opts.NoBuiltinFuncs);
  Opts.LongDoubleSize = Args.hasArg(OPT_mlong_double_128)
                            ? 128
                            : Args.hasArg(OPT_mlong_double_64) ? 64 : 0;
  if (Opts.FastRelaxedMath)
    Opts.setDefaultFPContractMode(LangOptions::FPM_Fast);
  llvm::sort(Opts.ModuleFeatures);

  // -mrtd option
  if (Arg *A = Args.getLastArg(OPT_mrtd)) {
    if (Opts.getDefaultCallingConv() != LangOptions::DCC_None)
      Diags.Report(diag::err_drv_argument_not_allowed_with)
          << A->getSpelling() << "-fdefault-calling-conv";
    else {
      if (T.getArch() != llvm::Triple::x86)
        Diags.Report(diag::err_drv_argument_not_allowed_with)
            << A->getSpelling() << T.getTriple();
      else
        Opts.setDefaultCallingConv(LangOptions::DCC_StdCall);
    }
  }

  // Check if -fopenmp-simd is specified.
  bool IsSimdSpecified =
      Args.hasFlag(options::OPT_fopenmp_simd, options::OPT_fno_openmp_simd,
                   /*Default=*/false);
  bool IsTargetSpecified =
      Opts.OpenMPIsDevice || Args.hasArg(options::OPT_fopenmp_targets_EQ);

  if (Opts.OpenMP || Opts.OpenMPSimd) {
    if (int Version = getLastArgIntValue(
            Args, OPT_fopenmp_version_EQ,
            (IsSimdSpecified || IsTargetSpecified) ? 50 : Opts.OpenMP, Diags))
      Opts.OpenMP = Version;
    // Provide diagnostic when a given target is not expected to be an OpenMP
    // device or host.
    if (!Opts.OpenMPIsDevice) {
      switch (T.getArch()) {
      default:
        break;
      // Add unsupported host targets here:
      case llvm::Triple::nvptx:
      case llvm::Triple::nvptx64:
        Diags.Report(diag::err_drv_omp_host_target_not_supported) << T.str();
        break;
      }
    }
  }

  // Set the flag to prevent the implementation from emitting device exception
  // handling code for those requiring so.
  if ((Opts.OpenMPIsDevice && (T.isNVPTX() || T.isAMDGCN())) ||
      Opts.OpenCLCPlusPlus) {
    Opts.Exceptions = 0;
    Opts.CXXExceptions = 0;
  }
  if (Opts.OpenMPIsDevice && T.isNVPTX()) {
    Opts.OpenMPCUDANumSMs =
        getLastArgIntValue(Args, options::OPT_fopenmp_cuda_number_of_sm_EQ,
                           Opts.OpenMPCUDANumSMs, Diags);
    Opts.OpenMPCUDABlocksPerSM =
        getLastArgIntValue(Args, options::OPT_fopenmp_cuda_blocks_per_sm_EQ,
                           Opts.OpenMPCUDABlocksPerSM, Diags);
    Opts.OpenMPCUDAReductionBufNum = getLastArgIntValue(
        Args, options::OPT_fopenmp_cuda_teams_reduction_recs_num_EQ,
        Opts.OpenMPCUDAReductionBufNum, Diags);
  }

  // Get the OpenMP target triples if any.
  if (Arg *A = Args.getLastArg(options::OPT_fopenmp_targets_EQ)) {
    enum ArchPtrSize { Arch16Bit, Arch32Bit, Arch64Bit };
    auto getArchPtrSize = [](const llvm::Triple &T) {
      if (T.isArch16Bit())
        return Arch16Bit;
      if (T.isArch32Bit())
        return Arch32Bit;
      assert(T.isArch64Bit() && "Expected 64-bit architecture");
      return Arch64Bit;
    };

    for (unsigned i = 0; i < A->getNumValues(); ++i) {
      llvm::Triple TT(A->getValue(i));

      if (TT.getArch() == llvm::Triple::UnknownArch ||
          !(TT.getArch() == llvm::Triple::aarch64 || TT.isPPC() ||
            TT.getArch() == llvm::Triple::nvptx ||
            TT.getArch() == llvm::Triple::nvptx64 ||
            TT.getArch() == llvm::Triple::amdgcn ||
            TT.getArch() == llvm::Triple::x86 ||
            TT.getArch() == llvm::Triple::x86_64))
        Diags.Report(diag::err_drv_invalid_omp_target) << A->getValue(i);
      else if (getArchPtrSize(T) != getArchPtrSize(TT))
        Diags.Report(diag::err_drv_incompatible_omp_arch)
            << A->getValue(i) << T.str();
      else
        Opts.OMPTargetTriples.push_back(TT);
    }
  }

  // Set CUDA mode for OpenMP target NVPTX/AMDGCN if specified in options
  Opts.OpenMPCUDAMode = Opts.OpenMPIsDevice && (T.isNVPTX() || T.isAMDGCN()) &&
                        Args.hasArg(options::OPT_fopenmp_cuda_mode);

  // Set CUDA support for parallel execution of target regions for OpenMP target
  // NVPTX/AMDGCN if specified in options.
  Opts.OpenMPCUDATargetParallel =
      Opts.OpenMPIsDevice && (T.isNVPTX() || T.isAMDGCN()) &&
      Args.hasArg(options::OPT_fopenmp_cuda_parallel_target_regions);

  // Set CUDA mode for OpenMP target NVPTX/AMDGCN if specified in options
  Opts.OpenMPCUDAForceFullRuntime =
      Opts.OpenMPIsDevice && (T.isNVPTX() || T.isAMDGCN()) &&
      Args.hasArg(options::OPT_fopenmp_cuda_force_full_runtime);

  // FIXME: Eliminate this dependency.
  unsigned Opt = getOptimizationLevel(Args, IK, Diags),
       OptSize = getOptimizationLevelSize(Args);
  Opts.Optimize = Opt != 0;
  Opts.OptimizeSize = OptSize != 0;

  // This is the __NO_INLINE__ define, which just depends on things like the
  // optimization level and -fno-inline, not actually whether the backend has
  // inlining enabled.
  Opts.NoInlineDefine = !Opts.Optimize;
  if (Arg *InlineArg = Args.getLastArg(
          options::OPT_finline_functions, options::OPT_finline_hint_functions,
          options::OPT_fno_inline_functions, options::OPT_fno_inline))
    if (InlineArg->getOption().matches(options::OPT_fno_inline))
      Opts.NoInlineDefine = true;

  if (Arg *A = Args.getLastArg(OPT_ffp_contract)) {
    StringRef Val = A->getValue();
    if (Val == "fast")
      Opts.setDefaultFPContractMode(LangOptions::FPM_Fast);
    else if (Val == "on")
      Opts.setDefaultFPContractMode(LangOptions::FPM_On);
    else if (Val == "off")
      Opts.setDefaultFPContractMode(LangOptions::FPM_Off);
    else if (Val == "fast-honor-pragmas")
      Opts.setDefaultFPContractMode(LangOptions::FPM_FastHonorPragmas);
    else
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Val;
  }

  // Parse -fsanitize= arguments.
  parseSanitizerKinds("-fsanitize=", Args.getAllArgValues(OPT_fsanitize_EQ),
                      Diags, Opts.Sanitize);
  std::vector<std::string> systemBlacklists =
      Args.getAllArgValues(OPT_fsanitize_system_blacklist);
  Opts.SanitizerBlacklistFiles.insert(Opts.SanitizerBlacklistFiles.end(),
                                      systemBlacklists.begin(),
                                      systemBlacklists.end());

  if (Arg *A = Args.getLastArg(OPT_fclang_abi_compat_EQ)) {
    Opts.setClangABICompat(LangOptions::ClangABI::Latest);

    StringRef Ver = A->getValue();
    std::pair<StringRef, StringRef> VerParts = Ver.split('.');
    unsigned Major, Minor = 0;

    // Check the version number is valid: either 3.x (0 <= x <= 9) or
    // y or y.0 (4 <= y <= current version).
    if (!VerParts.first.startswith("0") &&
        !VerParts.first.getAsInteger(10, Major) &&
        3 <= Major && Major <= CLANG_VERSION_MAJOR &&
        (Major == 3 ? VerParts.second.size() == 1 &&
                      !VerParts.second.getAsInteger(10, Minor)
                    : VerParts.first.size() == Ver.size() ||
                      VerParts.second == "0")) {
      // Got a valid version number.
      if (Major == 3 && Minor <= 8)
        Opts.setClangABICompat(LangOptions::ClangABI::Ver3_8);
      else if (Major <= 4)
        Opts.setClangABICompat(LangOptions::ClangABI::Ver4);
      else if (Major <= 6)
        Opts.setClangABICompat(LangOptions::ClangABI::Ver6);
      else if (Major <= 7)
        Opts.setClangABICompat(LangOptions::ClangABI::Ver7);
      else if (Major <= 9)
        Opts.setClangABICompat(LangOptions::ClangABI::Ver9);
      else if (Major <= 11)
        Opts.setClangABICompat(LangOptions::ClangABI::Ver11);
    } else if (Ver != "latest") {
      Diags.Report(diag::err_drv_invalid_value)
          << A->getAsString(Args) << A->getValue();
    }
  }

  if (Arg *A = Args.getLastArg(OPT_msign_return_address_EQ)) {
    StringRef SignScope = A->getValue();

    if (SignScope.equals_lower("none"))
      Opts.setSignReturnAddressScope(
          LangOptions::SignReturnAddressScopeKind::None);
    else if (SignScope.equals_lower("all"))
      Opts.setSignReturnAddressScope(
          LangOptions::SignReturnAddressScopeKind::All);
    else if (SignScope.equals_lower("non-leaf"))
      Opts.setSignReturnAddressScope(
          LangOptions::SignReturnAddressScopeKind::NonLeaf);
    else
      Diags.Report(diag::err_drv_invalid_value)
          << A->getAsString(Args) << SignScope;

    if (Arg *A = Args.getLastArg(OPT_msign_return_address_key_EQ)) {
      StringRef SignKey = A->getValue();
      if (!SignScope.empty() && !SignKey.empty()) {
        if (SignKey.equals_lower("a_key"))
          Opts.setSignReturnAddressKey(
              LangOptions::SignReturnAddressKeyKind::AKey);
        else if (SignKey.equals_lower("b_key"))
          Opts.setSignReturnAddressKey(
              LangOptions::SignReturnAddressKeyKind::BKey);
        else
          Diags.Report(diag::err_drv_invalid_value)
              << A->getAsString(Args) << SignKey;
      }
    }
  }
}

static bool isStrictlyPreprocessorAction(frontend::ActionKind Action) {
  switch (Action) {
  case frontend::ASTDeclList:
  case frontend::ASTDump:
  case frontend::ASTPrint:
  case frontend::ASTView:
  case frontend::EmitAssembly:
  case frontend::EmitBC:
  case frontend::EmitHTML:
  case frontend::EmitLLVM:
  case frontend::EmitLLVMOnly:
  case frontend::EmitCodeGenOnly:
  case frontend::EmitObj:
  case frontend::FixIt:
  case frontend::GenerateModule:
  case frontend::GenerateModuleInterface:
  case frontend::GenerateHeaderModule:
  case frontend::GeneratePCH:
  case frontend::GenerateInterfaceStubs:
  case frontend::ParseSyntaxOnly:
  case frontend::ModuleFileInfo:
  case frontend::VerifyPCH:
  case frontend::PluginAction:
  case frontend::RewriteObjC:
  case frontend::RewriteTest:
  case frontend::RunAnalysis:
  case frontend::TemplightDump:
  case frontend::MigrateSource:
    return false;

  case frontend::DumpCompilerOptions:
  case frontend::DumpRawTokens:
  case frontend::DumpTokens:
  case frontend::InitOnly:
  case frontend::PrintPreamble:
  case frontend::PrintPreprocessedInput:
  case frontend::RewriteMacros:
  case frontend::RunPreprocessorOnly:
  case frontend::PrintDependencyDirectivesSourceMinimizerOutput:
    return true;
  }
  llvm_unreachable("invalid frontend action");
}

static void ParsePreprocessorArgs(PreprocessorOptions &Opts, ArgList &Args,
                                  DiagnosticsEngine &Diags,
                                  frontend::ActionKind Action) {
  Opts.PCHWithHdrStop = Args.hasArg(OPT_pch_through_hdrstop_create) ||
                        Args.hasArg(OPT_pch_through_hdrstop_use);

  for (const auto *A : Args.filtered(OPT_error_on_deserialized_pch_decl))
    Opts.DeserializedPCHDeclsToErrorOn.insert(A->getValue());

  for (const auto &A : Args.getAllArgValues(OPT_fmacro_prefix_map_EQ)) {
    auto Split = StringRef(A).split('=');
    Opts.MacroPrefixMap.insert(
        {std::string(Split.first), std::string(Split.second)});
  }

  if (const Arg *A = Args.getLastArg(OPT_preamble_bytes_EQ)) {
    StringRef Value(A->getValue());
    size_t Comma = Value.find(',');
    unsigned Bytes = 0;
    unsigned EndOfLine = 0;

    if (Comma == StringRef::npos ||
        Value.substr(0, Comma).getAsInteger(10, Bytes) ||
        Value.substr(Comma + 1).getAsInteger(10, EndOfLine))
      Diags.Report(diag::err_drv_preamble_format);
    else {
      Opts.PrecompiledPreambleBytes.first = Bytes;
      Opts.PrecompiledPreambleBytes.second = (EndOfLine != 0);
    }
  }

  // Add the __CET__ macro if a CFProtection option is set.
  if (const Arg *A = Args.getLastArg(OPT_fcf_protection_EQ)) {
    StringRef Name = A->getValue();
    if (Name == "branch")
      Opts.addMacroDef("__CET__=1");
    else if (Name == "return")
      Opts.addMacroDef("__CET__=2");
    else if (Name == "full")
      Opts.addMacroDef("__CET__=3");
  }

  // Add macros from the command line.
  for (const auto *A : Args.filtered(OPT_D, OPT_U)) {
    if (A->getOption().matches(OPT_D))
      Opts.addMacroDef(A->getValue());
    else
      Opts.addMacroUndef(A->getValue());
  }

  // Add the ordered list of -includes.
  for (const auto *A : Args.filtered(OPT_include))
    Opts.Includes.emplace_back(A->getValue());

  for (const auto *A : Args.filtered(OPT_chain_include))
    Opts.ChainedIncludes.emplace_back(A->getValue());

  for (const auto *A : Args.filtered(OPT_remap_file)) {
    std::pair<StringRef, StringRef> Split = StringRef(A->getValue()).split(';');

    if (Split.second.empty()) {
      Diags.Report(diag::err_drv_invalid_remap_file) << A->getAsString(Args);
      continue;
    }

    Opts.addRemappedFile(Split.first, Split.second);
  }

  // Always avoid lexing editor placeholders when we're just running the
  // preprocessor as we never want to emit the
  // "editor placeholder in source file" error in PP only mode.
  if (isStrictlyPreprocessorAction(Action))
    Opts.LexEditorPlaceholders = false;
}

static void ParsePreprocessorOutputArgs(PreprocessorOutputOptions &Opts,
                                        ArgList &Args,
                                        frontend::ActionKind Action) {
  if (isStrictlyPreprocessorAction(Action))
    Opts.ShowCPP = !Args.hasArg(OPT_dM);
  else
    Opts.ShowCPP = 0;

  Opts.ShowMacros = Args.hasArg(OPT_dM) || Args.hasArg(OPT_dD);
}

static void ParseTargetArgs(TargetOptions &Opts, ArgList &Args,
                            DiagnosticsEngine &Diags) {
  if (Arg *A = Args.getLastArg(options::OPT_target_sdk_version_EQ)) {
    llvm::VersionTuple Version;
    if (Version.tryParse(A->getValue()))
      Diags.Report(diag::err_drv_invalid_value)
          << A->getAsString(Args) << A->getValue();
    else
      Opts.SDKVersion = Version;
  }
}

bool CompilerInvocation::CreateFromArgs(CompilerInvocation &Res,
                                        ArrayRef<const char *> CommandLineArgs,
                                        DiagnosticsEngine &Diags,
                                        const char *Argv0) {
  bool Success = true;

  // Parse the arguments.
  const OptTable &Opts = getDriverOptTable();
  const unsigned IncludedFlagsBitmask = options::CC1Option;
  unsigned MissingArgIndex, MissingArgCount;
  InputArgList Args = Opts.ParseArgs(CommandLineArgs, MissingArgIndex,
                                     MissingArgCount, IncludedFlagsBitmask);
  LangOptions &LangOpts = *Res.getLangOpts();

  // Check for missing argument error.
  if (MissingArgCount) {
    Diags.Report(diag::err_drv_missing_argument)
        << Args.getArgString(MissingArgIndex) << MissingArgCount;
    Success = false;
  }

  // Issue errors on unknown arguments.
  for (const auto *A : Args.filtered(OPT_UNKNOWN)) {
    auto ArgString = A->getAsString(Args);
    std::string Nearest;
    if (Opts.findNearest(ArgString, Nearest, IncludedFlagsBitmask) > 1)
      Diags.Report(diag::err_drv_unknown_argument) << ArgString;
    else
      Diags.Report(diag::err_drv_unknown_argument_with_suggestion)
          << ArgString << Nearest;
    Success = false;
  }

  Success &= Res.parseSimpleArgs(Args, Diags);

  Success &= ParseAnalyzerArgs(*Res.getAnalyzerOpts(), Args, Diags);
  ParseDependencyOutputArgs(Res.getDependencyOutputOpts(), Args);
  if (!Res.getDependencyOutputOpts().OutputFile.empty() &&
      Res.getDependencyOutputOpts().Targets.empty()) {
    Diags.Report(diag::err_fe_dependency_file_requires_MT);
    Success = false;
  }
  Success &= ParseDiagnosticArgs(Res.getDiagnosticOpts(), Args, &Diags,
                                 /*DefaultDiagColor=*/false);
  // FIXME: We shouldn't have to pass the DashX option around here
  InputKind DashX = ParseFrontendArgs(Res.getFrontendOpts(), Args, Diags,
                                      LangOpts.IsHeaderFile);
  ParseTargetArgs(Res.getTargetOpts(), Args, Diags);
  llvm::Triple T(Res.getTargetOpts().Triple);
  ParseHeaderSearchArgs(Res.getHeaderSearchOpts(), Args, Diags,
                        Res.getFileSystemOpts().WorkingDir);
  if (DashX.getFormat() == InputKind::Precompiled ||
      DashX.getLanguage() == Language::LLVM_IR) {
    // ObjCAAutoRefCount and Sanitize LangOpts are used to setup the
    // PassManager in BackendUtil.cpp. They need to be initializd no matter
    // what the input type is.
    if (Args.hasArg(OPT_fobjc_arc))
      LangOpts.ObjCAutoRefCount = 1;
    // PIClevel and PIELevel are needed during code generation and this should be
    // set regardless of the input type.
    LangOpts.PICLevel = getLastArgIntValue(Args, OPT_pic_level, 0, Diags);
    LangOpts.PIE = Args.hasArg(OPT_pic_is_pie);
    parseSanitizerKinds("-fsanitize=", Args.getAllArgValues(OPT_fsanitize_EQ),
                        Diags, LangOpts.Sanitize);
  } else {
    // Other LangOpts are only initialized when the input is not AST or LLVM IR.
    // FIXME: Should we really be calling this for an Language::Asm input?
    ParseLangArgs(LangOpts, Args, DashX, T, Res.getPreprocessorOpts().Includes,
                  Diags);
    if (Res.getFrontendOpts().ProgramAction == frontend::RewriteObjC)
      LangOpts.ObjCExceptions = 1;
    if (T.isOSDarwin() && DashX.isPreprocessed()) {
      // Supress the darwin-specific 'stdlibcxx-not-found' diagnostic for
      // preprocessed input as we don't expect it to be used with -std=libc++
      // anyway.
      Res.getDiagnosticOpts().Warnings.push_back("no-stdlibcxx-not-found");
    }
  }

  if (LangOpts.CUDA) {
    // During CUDA device-side compilation, the aux triple is the
    // triple used for host compilation.
    if (LangOpts.CUDAIsDevice)
      Res.getTargetOpts().HostTriple = Res.getFrontendOpts().AuxTriple;
  }

  // Set the triple of the host for OpenMP device compile.
  if (LangOpts.OpenMPIsDevice)
    Res.getTargetOpts().HostTriple = Res.getFrontendOpts().AuxTriple;

  Success &= ParseCodeGenArgs(Res.getCodeGenOpts(), Args, DashX, Diags, T,
                              Res.getFrontendOpts().OutputFile, LangOpts);

  // FIXME: Override value name discarding when asan or msan is used because the
  // backend passes depend on the name of the alloca in order to print out
  // names.
  Res.getCodeGenOpts().DiscardValueNames &=
      !LangOpts.Sanitize.has(SanitizerKind::Address) &&
      !LangOpts.Sanitize.has(SanitizerKind::KernelAddress) &&
      !LangOpts.Sanitize.has(SanitizerKind::Memory) &&
      !LangOpts.Sanitize.has(SanitizerKind::KernelMemory);

  ParsePreprocessorArgs(Res.getPreprocessorOpts(), Args, Diags,
                        Res.getFrontendOpts().ProgramAction);
  ParsePreprocessorOutputArgs(Res.getPreprocessorOutputOpts(), Args,
                              Res.getFrontendOpts().ProgramAction);

  // Turn on -Wspir-compat for SPIR target.
  if (T.isSPIR())
    Res.getDiagnosticOpts().Warnings.push_back("spir-compat");

  // If sanitizer is enabled, disable OPT_ffine_grained_bitfield_accesses.
  if (Res.getCodeGenOpts().FineGrainedBitfieldAccesses &&
      !Res.getLangOpts()->Sanitize.empty()) {
    Res.getCodeGenOpts().FineGrainedBitfieldAccesses = false;
    Diags.Report(diag::warn_drv_fine_grained_bitfield_accesses_ignored);
  }

  // Store the command-line for using in the CodeView backend.
  Res.getCodeGenOpts().Argv0 = Argv0;
  Res.getCodeGenOpts().CommandLineArgs = CommandLineArgs;

  FixupInvocation(Res, Diags, Args, DashX);

  return Success;
}

std::string CompilerInvocation::getModuleHash() const {
  // Note: For QoI reasons, the things we use as a hash here should all be
  // dumped via the -module-info flag.
  using llvm::hash_code;
  using llvm::hash_value;
  using llvm::hash_combine;
  using llvm::hash_combine_range;

  // Start the signature with the compiler version.
  // FIXME: We'd rather use something more cryptographically sound than
  // CityHash, but this will do for now.
  hash_code code = hash_value(getClangFullRepositoryVersion());

  // Also include the serialization version, in case LLVM_APPEND_VC_REV is off
  // and getClangFullRepositoryVersion() doesn't include git revision.
  code = hash_combine(code, serialization::VERSION_MAJOR,
                      serialization::VERSION_MINOR);

  // Extend the signature with the language options
#define LANGOPT(Name, Bits, Default, Description) \
   code = hash_combine(code, LangOpts->Name);
#define ENUM_LANGOPT(Name, Type, Bits, Default, Description) \
  code = hash_combine(code, static_cast<unsigned>(LangOpts->get##Name()));
#define BENIGN_LANGOPT(Name, Bits, Default, Description)
#define BENIGN_ENUM_LANGOPT(Name, Type, Bits, Default, Description)
#include "clang/Basic/LangOptions.def"

  for (StringRef Feature : LangOpts->ModuleFeatures)
    code = hash_combine(code, Feature);

  code = hash_combine(code, LangOpts->ObjCRuntime);
  const auto &BCN = LangOpts->CommentOpts.BlockCommandNames;
  code = hash_combine(code, hash_combine_range(BCN.begin(), BCN.end()));

  // Extend the signature with the target options.
  code = hash_combine(code, TargetOpts->Triple, TargetOpts->CPU,
                      TargetOpts->TuneCPU, TargetOpts->ABI);
  for (const auto &FeatureAsWritten : TargetOpts->FeaturesAsWritten)
    code = hash_combine(code, FeatureAsWritten);

  // Extend the signature with preprocessor options.
  const PreprocessorOptions &ppOpts = getPreprocessorOpts();
  const HeaderSearchOptions &hsOpts = getHeaderSearchOpts();
  code = hash_combine(code, ppOpts.UsePredefines, ppOpts.DetailedRecord);

  for (const auto &I : getPreprocessorOpts().Macros) {
    // If we're supposed to ignore this macro for the purposes of modules,
    // don't put it into the hash.
    if (!hsOpts.ModulesIgnoreMacros.empty()) {
      // Check whether we're ignoring this macro.
      StringRef MacroDef = I.first;
      if (hsOpts.ModulesIgnoreMacros.count(
              llvm::CachedHashString(MacroDef.split('=').first)))
        continue;
    }

    code = hash_combine(code, I.first, I.second);
  }

  // Extend the signature with the sysroot and other header search options.
  code = hash_combine(code, hsOpts.Sysroot,
                      hsOpts.ModuleFormat,
                      hsOpts.UseDebugInfo,
                      hsOpts.UseBuiltinIncludes,
                      hsOpts.UseStandardSystemIncludes,
                      hsOpts.UseStandardCXXIncludes,
                      hsOpts.UseLibcxx,
                      hsOpts.ModulesValidateDiagnosticOptions);
  code = hash_combine(code, hsOpts.ResourceDir);

  if (hsOpts.ModulesStrictContextHash) {
    hash_code SHPC = hash_combine_range(hsOpts.SystemHeaderPrefixes.begin(),
                                        hsOpts.SystemHeaderPrefixes.end());
    hash_code UEC = hash_combine_range(hsOpts.UserEntries.begin(),
                                       hsOpts.UserEntries.end());
    code = hash_combine(code, hsOpts.SystemHeaderPrefixes.size(), SHPC,
                        hsOpts.UserEntries.size(), UEC);

    const DiagnosticOptions &diagOpts = getDiagnosticOpts();
    #define DIAGOPT(Name, Bits, Default) \
      code = hash_combine(code, diagOpts.Name);
    #define ENUM_DIAGOPT(Name, Type, Bits, Default) \
      code = hash_combine(code, diagOpts.get##Name());
    #include "clang/Basic/DiagnosticOptions.def"
    #undef DIAGOPT
    #undef ENUM_DIAGOPT
  }

  // Extend the signature with the user build path.
  code = hash_combine(code, hsOpts.ModuleUserBuildPath);

  // Extend the signature with the module file extensions.
  const FrontendOptions &frontendOpts = getFrontendOpts();
  for (const auto &ext : frontendOpts.ModuleFileExtensions) {
    code = ext->hashExtension(code);
  }

  // When compiling with -gmodules, also hash -fdebug-prefix-map as it
  // affects the debug info in the PCM.
  if (getCodeGenOpts().DebugTypeExtRefs)
    for (const auto &KeyValue : getCodeGenOpts().DebugPrefixMap)
      code = hash_combine(code, KeyValue.first, KeyValue.second);

  // Extend the signature with the enabled sanitizers, if at least one is
  // enabled. Sanitizers which cannot affect AST generation aren't hashed.
  SanitizerSet SanHash = LangOpts->Sanitize;
  SanHash.clear(getPPTransparentSanitizers());
  if (!SanHash.empty())
    code = hash_combine(code, SanHash.Mask);

  return llvm::APInt(64, code).toString(36, /*Signed=*/false);
}

void CompilerInvocation::generateCC1CommandLine(
    SmallVectorImpl<const char *> &Args, StringAllocator SA) const {
#define OPTION_WITH_MARSHALLING(                                               \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  GENERATE_OPTION_WITH_MARSHALLING(Args, SA, KIND, FLAGS, SPELLING,            \
                                   ALWAYS_EMIT, this->KEYPATH, DEFAULT_VALUE,  \
                                   IMPLIED_CHECK, IMPLIED_VALUE, DENORMALIZER, \
                                   EXTRACTOR, TABLE_INDEX)

#define DIAG_OPTION_WITH_MARSHALLING OPTION_WITH_MARSHALLING
#define LANG_OPTION_WITH_MARSHALLING OPTION_WITH_MARSHALLING
#define CODEGEN_OPTION_WITH_MARSHALLING OPTION_WITH_MARSHALLING

#include "clang/Driver/Options.inc"

#undef CODEGEN_OPTION_WITH_MARSHALLING
#undef LANG_OPTION_WITH_MARSHALLING
#undef DIAG_OPTION_WITH_MARSHALLING
#undef OPTION_WITH_MARSHALLING

  GenerateHeaderSearchArgs(*HeaderSearchOpts, Args, SA);
  GenerateLangArgs(*LangOpts, Args, SA);
}

IntrusiveRefCntPtr<llvm::vfs::FileSystem>
clang::createVFSFromCompilerInvocation(const CompilerInvocation &CI,
                                       DiagnosticsEngine &Diags) {
  return createVFSFromCompilerInvocation(CI, Diags,
                                         llvm::vfs::getRealFileSystem());
}

IntrusiveRefCntPtr<llvm::vfs::FileSystem>
clang::createVFSFromCompilerInvocation(
    const CompilerInvocation &CI, DiagnosticsEngine &Diags,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS) {
  if (CI.getHeaderSearchOpts().VFSOverlayFiles.empty())
    return BaseFS;

  IntrusiveRefCntPtr<llvm::vfs::FileSystem> Result = BaseFS;
  // earlier vfs files are on the bottom
  for (const auto &File : CI.getHeaderSearchOpts().VFSOverlayFiles) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
        Result->getBufferForFile(File);
    if (!Buffer) {
      Diags.Report(diag::err_missing_vfs_overlay_file) << File;
      continue;
    }

    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS = llvm::vfs::getVFSFromYAML(
        std::move(Buffer.get()), /*DiagHandler*/ nullptr, File,
        /*DiagContext*/ nullptr, Result);
    if (!FS) {
      Diags.Report(diag::err_invalid_vfs_overlay) << File;
      continue;
    }

    Result = FS;
  }
  return Result;
}
