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
#include "clang/Frontend/TextDiagnosticBuffer.h"
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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
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
#include "llvm/Support/HashBuilder.h"
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
#include <fstream>
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
// Helpers.
//===----------------------------------------------------------------------===//

// Parse misexpect tolerance argument value.
// Valid option values are integers in the range [0, 100)
inline Expected<Optional<uint64_t>> parseToleranceOption(StringRef Arg) {
  int64_t Val;
  if (Arg.getAsInteger(10, Val))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Not an integer: %s", Arg.data());
  return Val;
}

//===----------------------------------------------------------------------===//
// Initialization.
//===----------------------------------------------------------------------===//

CompilerInvocationRefBase::CompilerInvocationRefBase()
    : LangOpts(new LangOptions()), TargetOpts(new TargetOptions()),
      DiagnosticOpts(new DiagnosticOptions()),
      HeaderSearchOpts(new HeaderSearchOptions()),
      PreprocessorOpts(new PreprocessorOptions()),
      AnalyzerOpts(new AnalyzerOptions()) {}

CompilerInvocationRefBase::CompilerInvocationRefBase(
    const CompilerInvocationRefBase &X)
    : LangOpts(new LangOptions(*X.getLangOpts())),
      TargetOpts(new TargetOptions(X.getTargetOpts())),
      DiagnosticOpts(new DiagnosticOptions(X.getDiagnosticOpts())),
      HeaderSearchOpts(new HeaderSearchOptions(X.getHeaderSearchOpts())),
      PreprocessorOpts(new PreprocessorOptions(X.getPreprocessorOpts())),
      AnalyzerOpts(new AnalyzerOptions(*X.getAnalyzerOpts())) {}

CompilerInvocationRefBase::CompilerInvocationRefBase(
    CompilerInvocationRefBase &&X) = default;

CompilerInvocationRefBase &
CompilerInvocationRefBase::operator=(CompilerInvocationRefBase X) {
  LangOpts.swap(X.LangOpts);
  TargetOpts.swap(X.TargetOpts);
  DiagnosticOpts.swap(X.DiagnosticOpts);
  HeaderSearchOpts.swap(X.HeaderSearchOpts);
  PreprocessorOpts.swap(X.PreprocessorOpts);
  AnalyzerOpts.swap(X.AnalyzerOpts);
  return *this;
}

CompilerInvocationRefBase &
CompilerInvocationRefBase::operator=(CompilerInvocationRefBase &&X) = default;

CompilerInvocationRefBase::~CompilerInvocationRefBase() = default;

//===----------------------------------------------------------------------===//
// Normalizers
//===----------------------------------------------------------------------===//

#define SIMPLE_ENUM_VALUE_TABLE
#include "clang/Driver/Options.inc"
#undef SIMPLE_ENUM_VALUE_TABLE

static llvm::Optional<bool> normalizeSimpleFlag(OptSpecifier Opt,
                                                unsigned TableIndex,
                                                const ArgList &Args,
                                                DiagnosticsEngine &Diags) {
  if (Args.hasArg(Opt))
    return true;
  return None;
}

static Optional<bool> normalizeSimpleNegativeFlag(OptSpecifier Opt, unsigned,
                                                  const ArgList &Args,
                                                  DiagnosticsEngine &) {
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
                 DiagnosticsEngine &) -> Optional<T> {
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
                                       const ArgList &Args,
                                       DiagnosticsEngine &) -> Optional<bool> {
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
                                  const Twine &Value) {
  switch (OptClass) {
  case Option::SeparateClass:
  case Option::JoinedOrSeparateClass:
  case Option::JoinedAndSeparateClass:
    Args.push_back(Spelling);
    Args.push_back(SA(Value));
    break;
  case Option::JoinedClass:
  case Option::CommaJoinedClass:
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

static llvm::Optional<unsigned> normalizeSimpleEnum(OptSpecifier Opt,
                                                    unsigned TableIndex,
                                                    const ArgList &Args,
                                                    DiagnosticsEngine &Diags) {
  assert(TableIndex < SimpleEnumValueTablesSize);
  const SimpleEnumValueTable &Table = SimpleEnumValueTables[TableIndex];

  auto *Arg = Args.getLastArg(Opt);
  if (!Arg)
    return None;

  StringRef ArgValue = Arg->getValue();
  if (auto MaybeEnumVal = findValueTableByName(Table, ArgValue))
    return MaybeEnumVal->Value;

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
                                             DiagnosticsEngine &Diags) {
  auto *Arg = Args.getLastArg(Opt);
  if (!Arg)
    return None;
  return std::string(Arg->getValue());
}

template <typename IntTy>
static Optional<IntTy> normalizeStringIntegral(OptSpecifier Opt, int,
                                               const ArgList &Args,
                                               DiagnosticsEngine &Diags) {
  auto *Arg = Args.getLastArg(Opt);
  if (!Arg)
    return None;
  IntTy Res;
  if (StringRef(Arg->getValue()).getAsInteger(0, Res)) {
    Diags.Report(diag::err_drv_invalid_int_value)
        << Arg->getAsString(Args) << Arg->getValue();
    return None;
  }
  return Res;
}

static Optional<std::vector<std::string>>
normalizeStringVector(OptSpecifier Opt, int, const ArgList &Args,
                      DiagnosticsEngine &) {
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
                                             DiagnosticsEngine &Diags) {
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
  return ((KeyPath & Value) == Value) ? static_cast<T>(Value) : T();
}

#define PARSE_OPTION_WITH_MARSHALLING(                                         \
    ARGS, DIAGS, ID, FLAGS, PARAM, SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,       \
    IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, MERGER, TABLE_INDEX)             \
  if ((FLAGS)&options::CC1Option) {                                            \
    KEYPATH = MERGER(KEYPATH, DEFAULT_VALUE);                                  \
    if (IMPLIED_CHECK)                                                         \
      KEYPATH = MERGER(KEYPATH, IMPLIED_VALUE);                                \
    if (SHOULD_PARSE)                                                          \
      if (auto MaybeValue = NORMALIZER(OPT_##ID, TABLE_INDEX, ARGS, DIAGS))    \
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

static StringRef GetInputKindName(InputKind IK);

static bool FixupInvocation(CompilerInvocation &Invocation,
                            DiagnosticsEngine &Diags, const ArgList &Args,
                            InputKind IK) {
  unsigned NumErrorsBefore = Diags.getNumErrors();

  LangOptions &LangOpts = *Invocation.getLangOpts();
  CodeGenOptions &CodeGenOpts = Invocation.getCodeGenOpts();
  TargetOptions &TargetOpts = Invocation.getTargetOpts();
  FrontendOptions &FrontendOpts = Invocation.getFrontendOpts();
  CodeGenOpts.XRayInstrumentFunctions = LangOpts.XRayInstrument;
  CodeGenOpts.XRayAlwaysEmitCustomEvents = LangOpts.XRayAlwaysEmitCustomEvents;
  CodeGenOpts.XRayAlwaysEmitTypedEvents = LangOpts.XRayAlwaysEmitTypedEvents;
  CodeGenOpts.DisableFree = FrontendOpts.DisableFree;
  FrontendOpts.GenerateGlobalModuleIndex = FrontendOpts.UseGlobalModuleIndex;
  if (FrontendOpts.ShowStats)
    CodeGenOpts.ClearASTBeforeBackend = false;
  LangOpts.SanitizeCoverage = CodeGenOpts.hasSanitizeCoverage();
  LangOpts.ForceEmitVTables = CodeGenOpts.ForceEmitVTables;
  LangOpts.SpeculativeLoadHardening = CodeGenOpts.SpeculativeLoadHardening;
  LangOpts.CurrentModule = LangOpts.ModuleName;

  llvm::Triple T(TargetOpts.Triple);
  llvm::Triple::ArchType Arch = T.getArch();

  CodeGenOpts.CodeModel = TargetOpts.CodeModel;

  if (LangOpts.getExceptionHandling() !=
          LangOptions::ExceptionHandlingKind::None &&
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

  // Prevent the user from specifying both -fsycl-is-device and -fsycl-is-host.
  if (LangOpts.SYCLIsDevice && LangOpts.SYCLIsHost)
    Diags.Report(diag::err_drv_argument_not_allowed_with) << "-fsycl-is-device"
                                                          << "-fsycl-is-host";

  if (Args.hasArg(OPT_fgnu89_inline) && LangOpts.CPlusPlus)
    Diags.Report(diag::err_drv_argument_not_allowed_with)
        << "-fgnu89-inline" << GetInputKindName(IK);

  if (Args.hasArg(OPT_fgpu_allow_device_init) && !LangOpts.HIP)
    Diags.Report(diag::warn_ignored_hip_only_option)
        << Args.getLastArg(OPT_fgpu_allow_device_init)->getAsString(Args);

  if (Args.hasArg(OPT_gpu_max_threads_per_block_EQ) && !LangOpts.HIP)
    Diags.Report(diag::warn_ignored_hip_only_option)
        << Args.getLastArg(OPT_gpu_max_threads_per_block_EQ)->getAsString(Args);

  // When these options are used, the compiler is allowed to apply
  // optimizations that may affect the final result. For example
  // (x+y)+z is transformed to x+(y+z) but may not give the same
  // final result; it's not value safe.
  // Another example can be to simplify x/x to 1.0 but x could be 0.0, INF
  // or NaN. Final result may then differ. An error is issued when the eval
  // method is set with one of these options.
  if (Args.hasArg(OPT_ffp_eval_method_EQ)) {
    if (LangOpts.ApproxFunc)
      Diags.Report(diag::err_incompatible_fp_eval_method_options) << 0;
    if (LangOpts.AllowFPReassoc)
      Diags.Report(diag::err_incompatible_fp_eval_method_options) << 1;
    if (LangOpts.AllowRecip)
      Diags.Report(diag::err_incompatible_fp_eval_method_options) << 2;
  }

  // -cl-strict-aliasing needs to emit diagnostic in the case where CL > 1.0.
  // This option should be deprecated for CL > 1.0 because
  // this option was added for compatibility with OpenCL 1.0.
  if (Args.getLastArg(OPT_cl_strict_aliasing) &&
      (LangOpts.getOpenCLCompatibleVersion() > 100))
    Diags.Report(diag::warn_option_invalid_ocl_version)
        << LangOpts.getOpenCLVersionString()
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

  return Diags.getNumErrors() == NumErrorsBefore;
}

//===----------------------------------------------------------------------===//
// Deserialization (from args)
//===----------------------------------------------------------------------===//

static unsigned getOptimizationLevel(ArgList &Args, InputKind IK,
                                     DiagnosticsEngine &Diags) {
  unsigned DefaultOpt = llvm::CodeGenOpt::None;
  if ((IK.getLanguage() == Language::OpenCL ||
       IK.getLanguage() == Language::OpenCLCXX) &&
      !Args.hasArg(OPT_cl_opt_disable))
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

static void GenerateArg(SmallVectorImpl<const char *> &Args,
                        llvm::opt::OptSpecifier OptSpecifier,
                        CompilerInvocation::StringAllocator SA) {
  Option Opt = getDriverOptTable().getOption(OptSpecifier);
  denormalizeSimpleFlag(Args, SA(Opt.getPrefix() + Opt.getName()), SA,
                        Option::OptionClass::FlagClass, 0);
}

static void GenerateArg(SmallVectorImpl<const char *> &Args,
                        llvm::opt::OptSpecifier OptSpecifier,
                        const Twine &Value,
                        CompilerInvocation::StringAllocator SA) {
  Option Opt = getDriverOptTable().getOption(OptSpecifier);
  denormalizeString(Args, SA(Opt.getPrefix() + Opt.getName()), SA,
                    Opt.getKind(), 0, Value);
}

// Parse command line arguments into CompilerInvocation.
using ParseFn =
    llvm::function_ref<bool(CompilerInvocation &, ArrayRef<const char *>,
                            DiagnosticsEngine &, const char *)>;

// Generate command line arguments from CompilerInvocation.
using GenerateFn = llvm::function_ref<void(
    CompilerInvocation &, SmallVectorImpl<const char *> &,
    CompilerInvocation::StringAllocator)>;

// May perform round-trip of command line arguments. By default, the round-trip
// is enabled in assert builds. This can be overwritten at run-time via the
// "-round-trip-args" and "-no-round-trip-args" command line flags.
// During round-trip, the command line arguments are parsed into a dummy
// instance of CompilerInvocation which is used to generate the command line
// arguments again. The real CompilerInvocation instance is then created by
// parsing the generated arguments, not the original ones.
static bool RoundTrip(ParseFn Parse, GenerateFn Generate,
                      CompilerInvocation &RealInvocation,
                      CompilerInvocation &DummyInvocation,
                      ArrayRef<const char *> CommandLineArgs,
                      DiagnosticsEngine &Diags, const char *Argv0) {
#ifndef NDEBUG
  bool DoRoundTripDefault = true;
#else
  bool DoRoundTripDefault = false;
#endif

  bool DoRoundTrip = DoRoundTripDefault;
  for (const auto *Arg : CommandLineArgs) {
    if (Arg == StringRef("-round-trip-args"))
      DoRoundTrip = true;
    if (Arg == StringRef("-no-round-trip-args"))
      DoRoundTrip = false;
  }

  // If round-trip was not requested, simply run the parser with the real
  // invocation diagnostics.
  if (!DoRoundTrip)
    return Parse(RealInvocation, CommandLineArgs, Diags, Argv0);

  // Serializes quoted (and potentially escaped) arguments.
  auto SerializeArgs = [](ArrayRef<const char *> Args) {
    std::string Buffer;
    llvm::raw_string_ostream OS(Buffer);
    for (const char *Arg : Args) {
      llvm::sys::printArg(OS, Arg, /*Quote=*/true);
      OS << ' ';
    }
    OS.flush();
    return Buffer;
  };

  // Setup a dummy DiagnosticsEngine.
  DiagnosticsEngine DummyDiags(new DiagnosticIDs(), new DiagnosticOptions());
  DummyDiags.setClient(new TextDiagnosticBuffer());

  // Run the first parse on the original arguments with the dummy invocation and
  // diagnostics.
  if (!Parse(DummyInvocation, CommandLineArgs, DummyDiags, Argv0) ||
      DummyDiags.getNumWarnings() != 0) {
    // If the first parse did not succeed, it must be user mistake (invalid
    // command line arguments). We won't be able to generate arguments that
    // would reproduce the same result. Let's fail again with the real
    // invocation and diagnostics, so all side-effects of parsing are visible.
    unsigned NumWarningsBefore = Diags.getNumWarnings();
    auto Success = Parse(RealInvocation, CommandLineArgs, Diags, Argv0);
    if (!Success || Diags.getNumWarnings() != NumWarningsBefore)
      return Success;

    // Parse with original options and diagnostics succeeded even though it
    // shouldn't have. Something is off.
    Diags.Report(diag::err_cc1_round_trip_fail_then_ok);
    Diags.Report(diag::note_cc1_round_trip_original)
        << SerializeArgs(CommandLineArgs);
    return false;
  }

  // Setup string allocator.
  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver StringPool(Alloc);
  auto SA = [&StringPool](const Twine &Arg) {
    return StringPool.save(Arg).data();
  };

  // Generate arguments from the dummy invocation. If Generate is the
  // inverse of Parse, the newly generated arguments must have the same
  // semantics as the original.
  SmallVector<const char *> GeneratedArgs1;
  Generate(DummyInvocation, GeneratedArgs1, SA);

  // Run the second parse, now on the generated arguments, and with the real
  // invocation and diagnostics. The result is what we will end up using for the
  // rest of compilation, so if Generate is not inverse of Parse, something down
  // the line will break.
  bool Success2 = Parse(RealInvocation, GeneratedArgs1, Diags, Argv0);

  // The first parse on original arguments succeeded, but second parse of
  // generated arguments failed. Something must be wrong with the generator.
  if (!Success2) {
    Diags.Report(diag::err_cc1_round_trip_ok_then_fail);
    Diags.Report(diag::note_cc1_round_trip_generated)
        << 1 << SerializeArgs(GeneratedArgs1);
    return false;
  }

  // Generate arguments again, this time from the options we will end up using
  // for the rest of the compilation.
  SmallVector<const char *> GeneratedArgs2;
  Generate(RealInvocation, GeneratedArgs2, SA);

  // Compares two lists of generated arguments.
  auto Equal = [](const ArrayRef<const char *> A,
                  const ArrayRef<const char *> B) {
    return std::equal(A.begin(), A.end(), B.begin(), B.end(),
                      [](const char *AElem, const char *BElem) {
                        return StringRef(AElem) == StringRef(BElem);
                      });
  };

  // If we generated different arguments from what we assume are two
  // semantically equivalent CompilerInvocations, the Generate function may
  // be non-deterministic.
  if (!Equal(GeneratedArgs1, GeneratedArgs2)) {
    Diags.Report(diag::err_cc1_round_trip_mismatch);
    Diags.Report(diag::note_cc1_round_trip_generated)
        << 1 << SerializeArgs(GeneratedArgs1);
    Diags.Report(diag::note_cc1_round_trip_generated)
        << 2 << SerializeArgs(GeneratedArgs2);
    return false;
  }

  Diags.Report(diag::remark_cc1_round_trip_generated)
      << 1 << SerializeArgs(GeneratedArgs1);
  Diags.Report(diag::remark_cc1_round_trip_generated)
      << 2 << SerializeArgs(GeneratedArgs2);

  return Success2;
}

static void addDiagnosticArgs(ArgList &Args, OptSpecifier Group,
                              OptSpecifier GroupWithValue,
                              std::vector<std::string> &Diagnostics) {
  for (auto *A : Args.filtered(Group)) {
    if (A->getOption().getKind() == Option::FlagClass) {
      // The argument is a pure flag (such as OPT_Wall or OPT_Wdeprecated). Add
      // its name (minus the "W" or "R" at the beginning) to the diagnostics.
      Diagnostics.push_back(
          std::string(A->getOption().getName().drop_front(1)));
    } else if (A->getOption().matches(GroupWithValue)) {
      // This is -Wfoo= or -Rfoo=, where foo is the name of the diagnostic
      // group. Add only the group name to the diagnostics.
      Diagnostics.push_back(
          std::string(A->getOption().getName().drop_front(1).rtrim("=-")));
    } else {
      // Otherwise, add its value (for OPT_W_Joined and similar).
      Diagnostics.push_back(A->getValue());
    }
  }
}

// Parse the Static Analyzer configuration. If \p Diags is set to nullptr,
// it won't verify the input.
static void parseAnalyzerConfigs(AnalyzerOptions &AnOpts,
                                 DiagnosticsEngine *Diags);

static void getAllNoBuiltinFuncValues(ArgList &Args,
                                      std::vector<std::string> &Funcs) {
  std::vector<std::string> Values = Args.getAllArgValues(OPT_fno_builtin_);
  auto BuiltinEnd = llvm::partition(Values, Builtin::Context::isBuiltinFunc);
  Funcs.insert(Funcs.end(), Values.begin(), BuiltinEnd);
}

static void GenerateAnalyzerArgs(AnalyzerOptions &Opts,
                                 SmallVectorImpl<const char *> &Args,
                                 CompilerInvocation::StringAllocator SA) {
  const AnalyzerOptions *AnalyzerOpts = &Opts;

#define ANALYZER_OPTION_WITH_MARSHALLING(                                      \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  GENERATE_OPTION_WITH_MARSHALLING(                                            \
      Args, SA, KIND, FLAGS, SPELLING, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE,    \
      IMPLIED_CHECK, IMPLIED_VALUE, DENORMALIZER, EXTRACTOR, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef ANALYZER_OPTION_WITH_MARSHALLING

  if (Opts.AnalysisConstraintsOpt != RangeConstraintsModel) {
    switch (Opts.AnalysisConstraintsOpt) {
#define ANALYSIS_CONSTRAINTS(NAME, CMDFLAG, DESC, CREATFN)                     \
  case NAME##Model:                                                            \
    GenerateArg(Args, OPT_analyzer_constraints, CMDFLAG, SA);                  \
    break;
#include "clang/StaticAnalyzer/Core/Analyses.def"
    default:
      llvm_unreachable("Tried to generate unknown analysis constraint.");
    }
  }

  if (Opts.AnalysisDiagOpt != PD_HTML) {
    switch (Opts.AnalysisDiagOpt) {
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATFN)                     \
  case PD_##NAME:                                                              \
    GenerateArg(Args, OPT_analyzer_output, CMDFLAG, SA);                       \
    break;
#include "clang/StaticAnalyzer/Core/Analyses.def"
    default:
      llvm_unreachable("Tried to generate unknown analysis diagnostic client.");
    }
  }

  if (Opts.AnalysisPurgeOpt != PurgeStmt) {
    switch (Opts.AnalysisPurgeOpt) {
#define ANALYSIS_PURGE(NAME, CMDFLAG, DESC)                                    \
  case NAME:                                                                   \
    GenerateArg(Args, OPT_analyzer_purge, CMDFLAG, SA);                        \
    break;
#include "clang/StaticAnalyzer/Core/Analyses.def"
    default:
      llvm_unreachable("Tried to generate unknown analysis purge mode.");
    }
  }

  if (Opts.InliningMode != NoRedundancy) {
    switch (Opts.InliningMode) {
#define ANALYSIS_INLINING_MODE(NAME, CMDFLAG, DESC)                            \
  case NAME:                                                                   \
    GenerateArg(Args, OPT_analyzer_inlining_mode, CMDFLAG, SA);                \
    break;
#include "clang/StaticAnalyzer/Core/Analyses.def"
    default:
      llvm_unreachable("Tried to generate unknown analysis inlining mode.");
    }
  }

  for (const auto &CP : Opts.CheckersAndPackages) {
    OptSpecifier Opt =
        CP.second ? OPT_analyzer_checker : OPT_analyzer_disable_checker;
    GenerateArg(Args, Opt, CP.first, SA);
  }

  AnalyzerOptions ConfigOpts;
  parseAnalyzerConfigs(ConfigOpts, nullptr);

  for (const auto &C : Opts.Config) {
    // Don't generate anything that came from parseAnalyzerConfigs. It would be
    // redundant and may not be valid on the command line.
    auto Entry = ConfigOpts.Config.find(C.getKey());
    if (Entry != ConfigOpts.Config.end() && Entry->getValue() == C.getValue())
      continue;

    GenerateArg(Args, OPT_analyzer_config, C.getKey() + "=" + C.getValue(), SA);
  }

  // Nothing to generate for FullCompilerInvocation.
}

static bool ParseAnalyzerArgs(AnalyzerOptions &Opts, ArgList &Args,
                              DiagnosticsEngine &Diags) {
  unsigned NumErrorsBefore = Diags.getNumErrors();

  AnalyzerOptions *AnalyzerOpts = &Opts;

#define ANALYZER_OPTION_WITH_MARSHALLING(                                      \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(                                               \
      Args, Diags, ID, FLAGS, PARAM, SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,     \
      IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef ANALYZER_OPTION_WITH_MARSHALLING

  if (Args.hasArg(OPT_analyzer_store))
    Diags.Report(diag::warn_analyzer_deprecated_option) << "-analyzer-store"
                                                        << "clang-16";
  if (Args.hasArg(OPT_analyzer_opt_analyze_nested_blocks))
    Diags.Report(diag::warn_analyzer_deprecated_option)
        << "-analyzer-opt-analyze-nested-blocks"
        << "clang-16";

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
    } else {
#ifndef LLVM_WITH_Z3
      if (Value == AnalysisConstraints::Z3ConstraintsModel) {
        Diags.Report(diag::err_analyzer_not_built_with_z3);
      }
#endif // LLVM_WITH_Z3
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
        break;
      }
      if (val.contains('=')) {
        Diags.Report(SourceLocation(),
                     diag::err_analyzer_config_multiple_values)
          << configVal;
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

  return Diags.getNumErrors() == NumErrorsBefore;
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
    OptionField = *PossiblyInvalidVal;
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
  // enough to store checker options.

#define ANALYZER_OPTION(TYPE, NAME, CMDFLAG, DESC, DEFAULT_VAL)                \
  initOption(AnOpts.Config, Diags, AnOpts.NAME, CMDFLAG, DEFAULT_VAL);
#define ANALYZER_OPTION_DEPENDS_ON_USER_MODE(...)
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.def"

  assert(AnOpts.UserMode == "shallow" || AnOpts.UserMode == "deep");
  const bool InShallowMode = AnOpts.UserMode == "shallow";

#define ANALYZER_OPTION(...)
#define ANALYZER_OPTION_DEPENDS_ON_USER_MODE(TYPE, NAME, CMDFLAG, DESC,        \
                                             SHALLOW_VAL, DEEP_VAL)            \
  initOption(AnOpts.Config, Diags, AnOpts.NAME, CMDFLAG,                       \
             InShallowMode ? SHALLOW_VAL : DEEP_VAL);
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.def"

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
        bool IsValidName = IsChecker
                               ? llvm::is_contained(Checkers, CheckerOrPackage)
                               : llvm::is_contained(Packages, CheckerOrPackage);

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

/// Generate a remark argument. This is an inverse of `ParseOptimizationRemark`.
static void
GenerateOptimizationRemark(SmallVectorImpl<const char *> &Args,
                           CompilerInvocation::StringAllocator SA,
                           OptSpecifier OptEQ, StringRef Name,
                           const CodeGenOptions::OptRemark &Remark) {
  if (Remark.hasValidPattern()) {
    GenerateArg(Args, OptEQ, Remark.Pattern, SA);
  } else if (Remark.Kind == CodeGenOptions::RK_Enabled) {
    GenerateArg(Args, OPT_R_Joined, Name, SA);
  } else if (Remark.Kind == CodeGenOptions::RK_Disabled) {
    GenerateArg(Args, OPT_R_Joined, StringRef("no-") + Name, SA);
  }
}

/// Parse a remark command line argument. It may be missing, disabled/enabled by
/// '-R[no-]group' or specified with a regular expression by '-Rgroup=regexp'.
/// On top of that, it can be disabled/enabled globally by '-R[no-]everything'.
static CodeGenOptions::OptRemark
ParseOptimizationRemark(DiagnosticsEngine &Diags, ArgList &Args,
                        OptSpecifier OptEQ, StringRef Name) {
  CodeGenOptions::OptRemark Result;

  auto InitializeResultPattern = [&Diags, &Args, &Result](const Arg *A,
                                                          StringRef Pattern) {
    Result.Pattern = Pattern.str();

    std::string RegexError;
    Result.Regex = std::make_shared<llvm::Regex>(Result.Pattern);
    if (!Result.Regex->isValid(RegexError)) {
      Diags.Report(diag::err_drv_optimization_remark_pattern)
          << RegexError << A->getAsString(Args);
      return false;
    }

    return true;
  };

  for (Arg *A : Args) {
    if (A->getOption().matches(OPT_R_Joined)) {
      StringRef Value = A->getValue();

      if (Value == Name)
        Result.Kind = CodeGenOptions::RK_Enabled;
      else if (Value == "everything")
        Result.Kind = CodeGenOptions::RK_EnabledEverything;
      else if (Value.split('-') == std::make_pair(StringRef("no"), Name))
        Result.Kind = CodeGenOptions::RK_Disabled;
      else if (Value == "no-everything")
        Result.Kind = CodeGenOptions::RK_DisabledEverything;
      else
        continue;

      if (Result.Kind == CodeGenOptions::RK_Disabled ||
          Result.Kind == CodeGenOptions::RK_DisabledEverything) {
        Result.Pattern = "";
        Result.Regex = nullptr;
      } else {
        InitializeResultPattern(A, ".*");
      }
    } else if (A->getOption().matches(OptEQ)) {
      Result.Kind = CodeGenOptions::RK_WithPattern;
      if (!InitializeResultPattern(A, A->getValue()))
        return CodeGenOptions::OptRemark();
    }
  }

  return Result;
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

static SmallVector<StringRef, 4> serializeSanitizerKinds(SanitizerSet S) {
  SmallVector<StringRef, 4> Values;
  serializeSanitizerSet(S, Values);
  return Values;
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

static std::string serializeXRayInstrumentationBundle(const XRayInstrSet &S) {
  llvm::SmallVector<StringRef, 2> BundleParts;
  serializeXRayInstrValue(S, BundleParts);
  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  llvm::interleave(BundleParts, OS, [&OS](StringRef Part) { OS << Part; }, ",");
  return Buffer;
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

void CompilerInvocation::GenerateCodeGenArgs(
    const CodeGenOptions &Opts, SmallVectorImpl<const char *> &Args,
    StringAllocator SA, const llvm::Triple &T, const std::string &OutputFile,
    const LangOptions *LangOpts) {
  const CodeGenOptions &CodeGenOpts = Opts;

  if (Opts.OptimizationLevel == 0)
    GenerateArg(Args, OPT_O0, SA);
  else
    GenerateArg(Args, OPT_O, Twine(Opts.OptimizationLevel), SA);

#define CODEGEN_OPTION_WITH_MARSHALLING(                                       \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  GENERATE_OPTION_WITH_MARSHALLING(                                            \
      Args, SA, KIND, FLAGS, SPELLING, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE,    \
      IMPLIED_CHECK, IMPLIED_VALUE, DENORMALIZER, EXTRACTOR, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef CODEGEN_OPTION_WITH_MARSHALLING

  if (Opts.OptimizationLevel > 0) {
    if (Opts.Inlining == CodeGenOptions::NormalInlining)
      GenerateArg(Args, OPT_finline_functions, SA);
    else if (Opts.Inlining == CodeGenOptions::OnlyHintInlining)
      GenerateArg(Args, OPT_finline_hint_functions, SA);
    else if (Opts.Inlining == CodeGenOptions::OnlyAlwaysInlining)
      GenerateArg(Args, OPT_fno_inline, SA);
  }

  if (Opts.DirectAccessExternalData && LangOpts->PICLevel != 0)
    GenerateArg(Args, OPT_fdirect_access_external_data, SA);
  else if (!Opts.DirectAccessExternalData && LangOpts->PICLevel == 0)
    GenerateArg(Args, OPT_fno_direct_access_external_data, SA);

  Optional<StringRef> DebugInfoVal;
  switch (Opts.DebugInfo) {
  case codegenoptions::DebugLineTablesOnly:
    DebugInfoVal = "line-tables-only";
    break;
  case codegenoptions::DebugDirectivesOnly:
    DebugInfoVal = "line-directives-only";
    break;
  case codegenoptions::DebugInfoConstructor:
    DebugInfoVal = "constructor";
    break;
  case codegenoptions::LimitedDebugInfo:
    DebugInfoVal = "limited";
    break;
  case codegenoptions::FullDebugInfo:
    DebugInfoVal = "standalone";
    break;
  case codegenoptions::UnusedTypeInfo:
    DebugInfoVal = "unused-types";
    break;
  case codegenoptions::NoDebugInfo: // default value
    DebugInfoVal = None;
    break;
  case codegenoptions::LocTrackingOnly: // implied value
    DebugInfoVal = None;
    break;
  }
  if (DebugInfoVal)
    GenerateArg(Args, OPT_debug_info_kind_EQ, *DebugInfoVal, SA);

  for (const auto &Prefix : Opts.DebugPrefixMap)
    GenerateArg(Args, OPT_fdebug_prefix_map_EQ,
                Prefix.first + "=" + Prefix.second, SA);

  for (const auto &Prefix : Opts.CoveragePrefixMap)
    GenerateArg(Args, OPT_fcoverage_prefix_map_EQ,
                Prefix.first + "=" + Prefix.second, SA);

  if (Opts.NewStructPathTBAA)
    GenerateArg(Args, OPT_new_struct_path_tbaa, SA);

  if (Opts.OptimizeSize == 1)
    GenerateArg(Args, OPT_O, "s", SA);
  else if (Opts.OptimizeSize == 2)
    GenerateArg(Args, OPT_O, "z", SA);

  // SimplifyLibCalls is set only in the absence of -fno-builtin and
  // -ffreestanding. We'll consider that when generating them.

  // NoBuiltinFuncs are generated by LangOptions.

  if (Opts.UnrollLoops && Opts.OptimizationLevel <= 1)
    GenerateArg(Args, OPT_funroll_loops, SA);
  else if (!Opts.UnrollLoops && Opts.OptimizationLevel > 1)
    GenerateArg(Args, OPT_fno_unroll_loops, SA);

  if (!Opts.BinutilsVersion.empty())
    GenerateArg(Args, OPT_fbinutils_version_EQ, Opts.BinutilsVersion, SA);

  if (Opts.DebugNameTable ==
      static_cast<unsigned>(llvm::DICompileUnit::DebugNameTableKind::GNU))
    GenerateArg(Args, OPT_ggnu_pubnames, SA);
  else if (Opts.DebugNameTable ==
           static_cast<unsigned>(
               llvm::DICompileUnit::DebugNameTableKind::Default))
    GenerateArg(Args, OPT_gpubnames, SA);

  auto TNK = Opts.getDebugSimpleTemplateNames();
  if (TNK != codegenoptions::DebugTemplateNamesKind::Full) {
    if (TNK == codegenoptions::DebugTemplateNamesKind::Simple)
      GenerateArg(Args, OPT_gsimple_template_names_EQ, "simple", SA);
    else if (TNK == codegenoptions::DebugTemplateNamesKind::Mangled)
      GenerateArg(Args, OPT_gsimple_template_names_EQ, "mangled", SA);
  }
  // ProfileInstrumentUsePath is marshalled automatically, no need to generate
  // it or PGOUseInstrumentor.

  if (Opts.TimePasses) {
    if (Opts.TimePassesPerRun)
      GenerateArg(Args, OPT_ftime_report_EQ, "per-pass-run", SA);
    else
      GenerateArg(Args, OPT_ftime_report, SA);
  }

  if (Opts.PrepareForLTO && !Opts.PrepareForThinLTO)
    GenerateArg(Args, OPT_flto_EQ, "full", SA);

  if (Opts.PrepareForThinLTO)
    GenerateArg(Args, OPT_flto_EQ, "thin", SA);

  if (!Opts.ThinLTOIndexFile.empty())
    GenerateArg(Args, OPT_fthinlto_index_EQ, Opts.ThinLTOIndexFile, SA);

  if (Opts.SaveTempsFilePrefix == OutputFile)
    GenerateArg(Args, OPT_save_temps_EQ, "obj", SA);

  StringRef MemProfileBasename("memprof.profraw");
  if (!Opts.MemoryProfileOutput.empty()) {
    if (Opts.MemoryProfileOutput == MemProfileBasename) {
      GenerateArg(Args, OPT_fmemory_profile, SA);
    } else {
      size_t ArgLength =
          Opts.MemoryProfileOutput.size() - MemProfileBasename.size();
      GenerateArg(Args, OPT_fmemory_profile_EQ,
                  Opts.MemoryProfileOutput.substr(0, ArgLength), SA);
    }
  }

  if (memcmp(Opts.CoverageVersion, "408*", 4) != 0)
    GenerateArg(Args, OPT_coverage_version_EQ,
                StringRef(Opts.CoverageVersion, 4), SA);

  // TODO: Check if we need to generate arguments stored in CmdArgs. (Namely
  //  '-fembed_bitcode', which does not map to any CompilerInvocation field and
  //  won't be generated.)

  if (Opts.XRayInstrumentationBundle.Mask != XRayInstrKind::All) {
    std::string InstrBundle =
        serializeXRayInstrumentationBundle(Opts.XRayInstrumentationBundle);
    if (!InstrBundle.empty())
      GenerateArg(Args, OPT_fxray_instrumentation_bundle, InstrBundle, SA);
  }

  if (Opts.CFProtectionReturn && Opts.CFProtectionBranch)
    GenerateArg(Args, OPT_fcf_protection_EQ, "full", SA);
  else if (Opts.CFProtectionReturn)
    GenerateArg(Args, OPT_fcf_protection_EQ, "return", SA);
  else if (Opts.CFProtectionBranch)
    GenerateArg(Args, OPT_fcf_protection_EQ, "branch", SA);

  if (Opts.IBTSeal)
    GenerateArg(Args, OPT_mibt_seal, SA);

  for (const auto &F : Opts.LinkBitcodeFiles) {
    bool Builtint = F.LinkFlags == llvm::Linker::Flags::LinkOnlyNeeded &&
                    F.PropagateAttrs && F.Internalize;
    GenerateArg(Args,
                Builtint ? OPT_mlink_builtin_bitcode : OPT_mlink_bitcode_file,
                F.Filename, SA);
  }

  // TODO: Consider removing marshalling annotations from f[no_]emulated_tls.
  //  That would make it easy to generate the option only **once** if it was
  //  explicitly set to non-default value.
  if (Opts.ExplicitEmulatedTLS) {
    GenerateArg(
        Args, Opts.EmulatedTLS ? OPT_femulated_tls : OPT_fno_emulated_tls, SA);
  }

  if (Opts.FPDenormalMode != llvm::DenormalMode::getIEEE())
    GenerateArg(Args, OPT_fdenormal_fp_math_EQ, Opts.FPDenormalMode.str(), SA);

  if ((Opts.FPDenormalMode != Opts.FP32DenormalMode) ||
      (Opts.FP32DenormalMode != llvm::DenormalMode::getIEEE()))
    GenerateArg(Args, OPT_fdenormal_fp_math_f32_EQ, Opts.FP32DenormalMode.str(),
                SA);

  if (Opts.StructReturnConvention == CodeGenOptions::SRCK_OnStack) {
    OptSpecifier Opt =
        T.isPPC32() ? OPT_maix_struct_return : OPT_fpcc_struct_return;
    GenerateArg(Args, Opt, SA);
  } else if (Opts.StructReturnConvention == CodeGenOptions::SRCK_InRegs) {
    OptSpecifier Opt =
        T.isPPC32() ? OPT_msvr4_struct_return : OPT_freg_struct_return;
    GenerateArg(Args, Opt, SA);
  }

  if (Opts.EnableAIXExtendedAltivecABI)
    GenerateArg(Args, OPT_mabi_EQ_vec_extabi, SA);

  if (!Opts.OptRecordPasses.empty())
    GenerateArg(Args, OPT_opt_record_passes, Opts.OptRecordPasses, SA);

  if (!Opts.OptRecordFormat.empty())
    GenerateArg(Args, OPT_opt_record_format, Opts.OptRecordFormat, SA);

  GenerateOptimizationRemark(Args, SA, OPT_Rpass_EQ, "pass",
                             Opts.OptimizationRemark);

  GenerateOptimizationRemark(Args, SA, OPT_Rpass_missed_EQ, "pass-missed",
                             Opts.OptimizationRemarkMissed);

  GenerateOptimizationRemark(Args, SA, OPT_Rpass_analysis_EQ, "pass-analysis",
                             Opts.OptimizationRemarkAnalysis);

  GenerateArg(Args, OPT_fdiagnostics_hotness_threshold_EQ,
              Opts.DiagnosticsHotnessThreshold
                  ? Twine(*Opts.DiagnosticsHotnessThreshold)
                  : "auto",
              SA);

  GenerateArg(Args, OPT_fdiagnostics_misexpect_tolerance_EQ,
              Twine(*Opts.DiagnosticsMisExpectTolerance), SA);

  for (StringRef Sanitizer : serializeSanitizerKinds(Opts.SanitizeRecover))
    GenerateArg(Args, OPT_fsanitize_recover_EQ, Sanitizer, SA);

  for (StringRef Sanitizer : serializeSanitizerKinds(Opts.SanitizeTrap))
    GenerateArg(Args, OPT_fsanitize_trap_EQ, Sanitizer, SA);

  if (!Opts.EmitVersionIdentMetadata)
    GenerateArg(Args, OPT_Qn, SA);

  switch (Opts.FiniteLoops) {
  case CodeGenOptions::FiniteLoopsKind::Language:
    break;
  case CodeGenOptions::FiniteLoopsKind::Always:
    GenerateArg(Args, OPT_ffinite_loops, SA);
    break;
  case CodeGenOptions::FiniteLoopsKind::Never:
    GenerateArg(Args, OPT_fno_finite_loops, SA);
    break;
  }
}

bool CompilerInvocation::ParseCodeGenArgs(CodeGenOptions &Opts, ArgList &Args,
                                          InputKind IK,
                                          DiagnosticsEngine &Diags,
                                          const llvm::Triple &T,
                                          const std::string &OutputFile,
                                          const LangOptions &LangOptsRef) {
  unsigned NumErrorsBefore = Diags.getNumErrors();

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
  PARSE_OPTION_WITH_MARSHALLING(                                               \
      Args, Diags, ID, FLAGS, PARAM, SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,     \
      IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef CODEGEN_OPTION_WITH_MARSHALLING

  // At O0 we want to fully disable inlining outside of cases marked with
  // 'alwaysinline' that are required for correctness.
  if (Opts.OptimizationLevel == 0) {
    Opts.setInlining(CodeGenOptions::OnlyAlwaysInlining);
  } else if (const Arg *A = Args.getLastArg(options::OPT_finline_functions,
                                            options::OPT_finline_hint_functions,
                                            options::OPT_fno_inline_functions,
                                            options::OPT_fno_inline)) {
    // Explicit inlining flags can disable some or all inlining even at
    // optimization levels above zero.
    if (A->getOption().matches(options::OPT_finline_functions))
      Opts.setInlining(CodeGenOptions::NormalInlining);
    else if (A->getOption().matches(options::OPT_finline_hint_functions))
      Opts.setInlining(CodeGenOptions::OnlyHintInlining);
    else
      Opts.setInlining(CodeGenOptions::OnlyAlwaysInlining);
  } else {
    Opts.setInlining(CodeGenOptions::NormalInlining);
  }

  // PIC defaults to -fno-direct-access-external-data while non-PIC defaults to
  // -fdirect-access-external-data.
  Opts.DirectAccessExternalData =
      Args.hasArg(OPT_fdirect_access_external_data) ||
      (!Args.hasArg(OPT_fno_direct_access_external_data) &&
       LangOpts->PICLevel == 0);

  if (Arg *A = Args.getLastArg(OPT_debug_info_kind_EQ)) {
    unsigned Val =
        llvm::StringSwitch<unsigned>(A->getValue())
            .Case("line-tables-only", codegenoptions::DebugLineTablesOnly)
            .Case("line-directives-only", codegenoptions::DebugDirectivesOnly)
            .Case("constructor", codegenoptions::DebugInfoConstructor)
            .Case("limited", codegenoptions::LimitedDebugInfo)
            .Case("standalone", codegenoptions::FullDebugInfo)
            .Case("unused-types", codegenoptions::UnusedTypeInfo)
            .Default(~0U);
    if (Val == ~0U)
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args)
                                                << A->getValue();
    else
      Opts.setDebugInfo(static_cast<codegenoptions::DebugInfoKind>(Val));
  }

  // If -fuse-ctor-homing is set and limited debug info is already on, then use
  // constructor homing, and vice versa for -fno-use-ctor-homing.
  if (const Arg *A =
          Args.getLastArg(OPT_fuse_ctor_homing, OPT_fno_use_ctor_homing)) {
    if (A->getOption().matches(OPT_fuse_ctor_homing) &&
        Opts.getDebugInfo() == codegenoptions::LimitedDebugInfo)
      Opts.setDebugInfo(codegenoptions::DebugInfoConstructor);
    if (A->getOption().matches(OPT_fno_use_ctor_homing) &&
        Opts.getDebugInfo() == codegenoptions::DebugInfoConstructor)
      Opts.setDebugInfo(codegenoptions::LimitedDebugInfo);
  }

  for (const auto &Arg : Args.getAllArgValues(OPT_fdebug_prefix_map_EQ)) {
    auto Split = StringRef(Arg).split('=');
    Opts.DebugPrefixMap.insert(
        {std::string(Split.first), std::string(Split.second)});
  }

  for (const auto &Arg : Args.getAllArgValues(OPT_fcoverage_prefix_map_EQ)) {
    auto Split = StringRef(Arg).split('=');
    Opts.CoveragePrefixMap.insert(
        {std::string(Split.first), std::string(Split.second)});
  }

  const llvm::Triple::ArchType DebugEntryValueArchs[] = {
      llvm::Triple::x86, llvm::Triple::x86_64, llvm::Triple::aarch64,
      llvm::Triple::arm, llvm::Triple::armeb, llvm::Triple::mips,
      llvm::Triple::mipsel, llvm::Triple::mips64, llvm::Triple::mips64el};

  if (Opts.OptimizationLevel > 0 && Opts.hasReducedDebugInfo() &&
      llvm::is_contained(DebugEntryValueArchs, T.getArch()))
    Opts.EmitCallSiteInfo = true;

  if (!Opts.EnableDIPreservationVerify && Opts.DIBugsReportFilePath.size()) {
    Diags.Report(diag::warn_ignoring_verify_debuginfo_preserve_export)
        << Opts.DIBugsReportFilePath;
    Opts.DIBugsReportFilePath = "";
  }

  Opts.NewStructPathTBAA = !Args.hasArg(OPT_no_struct_path_tbaa) &&
                           Args.hasArg(OPT_new_struct_path_tbaa);
  Opts.OptimizeSize = getOptimizationLevelSize(Args);
  Opts.SimplifyLibCalls = !LangOpts->NoBuiltin;
  if (Opts.SimplifyLibCalls)
    Opts.NoBuiltinFuncs = LangOpts->NoBuiltinFuncs;
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
  if (const Arg *A = Args.getLastArg(OPT_gsimple_template_names_EQ)) {
    StringRef Value = A->getValue();
    if (Value != "simple" && Value != "mangled")
      Diags.Report(diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << A->getValue();
    Opts.setDebugSimpleTemplateNames(
        StringRef(A->getValue()) == "simple"
            ? codegenoptions::DebugTemplateNamesKind::Simple
            : codegenoptions::DebugTemplateNamesKind::Mangled);
  }

  if (!Opts.ProfileInstrumentUsePath.empty())
    setPGOUseInstrumentor(Opts, Opts.ProfileInstrumentUsePath);

  if (const Arg *A = Args.getLastArg(OPT_ftime_report, OPT_ftime_report_EQ)) {
    Opts.TimePasses = true;

    // -ftime-report= is only for new pass manager.
    if (A->getOption().getID() == OPT_ftime_report_EQ) {
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

  Opts.PrepareForLTO = false;
  Opts.PrepareForThinLTO = false;
  if (Arg *A = Args.getLastArg(OPT_flto_EQ)) {
    Opts.PrepareForLTO = true;
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

  memcpy(Opts.CoverageVersion, "408*", 4);
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
    else if (Name != "none")
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Name;
  }

  if (Opts.PrepareForLTO && Args.hasArg(OPT_mibt_seal))
    Opts.IBTSeal = 1;

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

  if (Arg *A = Args.getLastArg(OPT_ftlsmodel_EQ)) {
    if (T.isOSAIX()) {
      StringRef Name = A->getValue();
      if (Name != "global-dynamic")
        Diags.Report(diag::err_aix_unsupported_tls_model) << Name;
    }
  }

  if (Arg *A = Args.getLastArg(OPT_fdenormal_fp_math_EQ)) {
    StringRef Val = A->getValue();
    Opts.FPDenormalMode = llvm::parseDenormalFPAttribute(Val);
    Opts.FP32DenormalMode = Opts.FPDenormalMode;
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

  if (Arg *A =
          Args.getLastArg(OPT_mabi_EQ_vec_default, OPT_mabi_EQ_vec_extabi)) {
    if (!T.isOSAIX())
      Diags.Report(diag::err_drv_unsupported_opt_for_target)
          << A->getSpelling() << T.str();

    const Option &O = A->getOption();
    Opts.EnableAIXExtendedAltivecABI = O.matches(OPT_mabi_EQ_vec_extabi);
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

  Opts.OptimizationRemark =
      ParseOptimizationRemark(Diags, Args, OPT_Rpass_EQ, "pass");

  Opts.OptimizationRemarkMissed =
      ParseOptimizationRemark(Diags, Args, OPT_Rpass_missed_EQ, "pass-missed");

  Opts.OptimizationRemarkAnalysis = ParseOptimizationRemark(
      Diags, Args, OPT_Rpass_analysis_EQ, "pass-analysis");

  NeedLocTracking |= Opts.OptimizationRemark.hasValidPattern() ||
                     Opts.OptimizationRemarkMissed.hasValidPattern() ||
                     Opts.OptimizationRemarkAnalysis.hasValidPattern();

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
      if ((!Opts.DiagnosticsHotnessThreshold ||
           Opts.DiagnosticsHotnessThreshold.getValue() > 0) &&
          !UsingProfile)
        Diags.Report(diag::warn_drv_diagnostics_hotness_requires_pgo)
            << "-fdiagnostics-hotness-threshold=";
    }
  }

  if (auto *arg =
          Args.getLastArg(options::OPT_fdiagnostics_misexpect_tolerance_EQ)) {
    auto ResultOrErr = parseToleranceOption(arg->getValue());

    if (!ResultOrErr) {
      Diags.Report(diag::err_drv_invalid_diagnotics_misexpect_tolerance)
          << "-fdiagnostics-misexpect-tolerance=";
    } else {
      Opts.DiagnosticsMisExpectTolerance = *ResultOrErr;
      if ((!Opts.DiagnosticsMisExpectTolerance ||
           Opts.DiagnosticsMisExpectTolerance.getValue() > 0) &&
          !UsingProfile)
        Diags.Report(diag::warn_drv_diagnostics_misexpect_requires_pgo)
            << "-fdiagnostics-misexpect-tolerance=";
    }
  }

  // If the user requested to use a sample profile for PGO, then the
  // backend will need to track source location information so the profile
  // can be incorporated into the IR.
  if (UsingSampleProfile)
    NeedLocTracking = true;

  if (!Opts.StackUsageOutput.empty())
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

  if (Args.hasArg(options::OPT_ffinite_loops))
    Opts.FiniteLoops = CodeGenOptions::FiniteLoopsKind::Always;
  else if (Args.hasArg(options::OPT_fno_finite_loops))
    Opts.FiniteLoops = CodeGenOptions::FiniteLoopsKind::Never;

  Opts.EmitIEEENaNCompliantInsts = Args.hasFlag(
      options::OPT_mamdgpu_ieee, options::OPT_mno_amdgpu_ieee, true);
  if (!Opts.EmitIEEENaNCompliantInsts && !LangOptsRef.NoHonorNaNs)
    Diags.Report(diag::err_drv_amdgpu_ieee_without_no_honor_nans);

  return Diags.getNumErrors() == NumErrorsBefore;
}

static void
GenerateDependencyOutputArgs(const DependencyOutputOptions &Opts,
                             SmallVectorImpl<const char *> &Args,
                             CompilerInvocation::StringAllocator SA) {
  const DependencyOutputOptions &DependencyOutputOpts = Opts;
#define DEPENDENCY_OUTPUT_OPTION_WITH_MARSHALLING(                             \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  GENERATE_OPTION_WITH_MARSHALLING(                                            \
      Args, SA, KIND, FLAGS, SPELLING, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE,    \
      IMPLIED_CHECK, IMPLIED_VALUE, DENORMALIZER, EXTRACTOR, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef DEPENDENCY_OUTPUT_OPTION_WITH_MARSHALLING

  if (Opts.ShowIncludesDest != ShowIncludesDestination::None)
    GenerateArg(Args, OPT_show_includes, SA);

  for (const auto &Dep : Opts.ExtraDeps) {
    switch (Dep.second) {
    case EDK_SanitizeIgnorelist:
      // Sanitizer ignorelist arguments are generated from LanguageOptions.
      continue;
    case EDK_ModuleFile:
      // Module file arguments are generated from FrontendOptions and
      // HeaderSearchOptions.
      continue;
    case EDK_ProfileList:
      // Profile list arguments are generated from LanguageOptions via the
      // marshalling infrastructure.
      continue;
    case EDK_DepFileEntry:
      GenerateArg(Args, OPT_fdepfile_entry, Dep.first, SA);
      break;
    }
  }
}

static bool ParseDependencyOutputArgs(DependencyOutputOptions &Opts,
                                      ArgList &Args, DiagnosticsEngine &Diags,
                                      frontend::ActionKind Action,
                                      bool ShowLineMarkers) {
  unsigned NumErrorsBefore = Diags.getNumErrors();

  DependencyOutputOptions &DependencyOutputOpts = Opts;
#define DEPENDENCY_OUTPUT_OPTION_WITH_MARSHALLING(                             \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(                                               \
      Args, Diags, ID, FLAGS, PARAM, SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,     \
      IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef DEPENDENCY_OUTPUT_OPTION_WITH_MARSHALLING

  if (Args.hasArg(OPT_show_includes)) {
    // Writing both /showIncludes and preprocessor output to stdout
    // would produce interleaved output, so use stderr for /showIncludes.
    // This behaves the same as cl.exe, when /E, /EP or /P are passed.
    if (Action == frontend::PrintPreprocessedInput || !ShowLineMarkers)
      Opts.ShowIncludesDest = ShowIncludesDestination::Stderr;
    else
      Opts.ShowIncludesDest = ShowIncludesDestination::Stdout;
  } else {
    Opts.ShowIncludesDest = ShowIncludesDestination::None;
  }

  // Add sanitizer ignorelists as extra dependencies.
  // They won't be discovered by the regular preprocessor, so
  // we let make / ninja to know about this implicit dependency.
  if (!Args.hasArg(OPT_fno_sanitize_ignorelist)) {
    for (const auto *A : Args.filtered(OPT_fsanitize_ignorelist_EQ)) {
      StringRef Val = A->getValue();
      if (!Val.contains('='))
        Opts.ExtraDeps.emplace_back(std::string(Val), EDK_SanitizeIgnorelist);
    }
    if (Opts.IncludeSystemHeaders) {
      for (const auto *A : Args.filtered(OPT_fsanitize_system_ignorelist_EQ)) {
        StringRef Val = A->getValue();
        if (!Val.contains('='))
          Opts.ExtraDeps.emplace_back(std::string(Val), EDK_SanitizeIgnorelist);
      }
    }
  }

  // -fprofile-list= dependencies.
  for (const auto &Filename : Args.getAllArgValues(OPT_fprofile_list_EQ))
    Opts.ExtraDeps.emplace_back(Filename, EDK_ProfileList);

  // Propagate the extra dependencies.
  for (const auto *A : Args.filtered(OPT_fdepfile_entry))
    Opts.ExtraDeps.emplace_back(A->getValue(), EDK_DepFileEntry);

  // Only the -fmodule-file=<file> form.
  for (const auto *A : Args.filtered(OPT_fmodule_file)) {
    StringRef Val = A->getValue();
    if (!Val.contains('='))
      Opts.ExtraDeps.emplace_back(std::string(Val), EDK_ModuleFile);
  }

  return Diags.getNumErrors() == NumErrorsBefore;
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
    if (O.matches(options::OPT_fcolor_diagnostics)) {
      ShowColors = Colors_On;
    } else if (O.matches(options::OPT_fno_color_diagnostics)) {
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

static void GenerateFileSystemArgs(const FileSystemOptions &Opts,
                                   SmallVectorImpl<const char *> &Args,
                                   CompilerInvocation::StringAllocator SA) {
  const FileSystemOptions &FileSystemOpts = Opts;

#define FILE_SYSTEM_OPTION_WITH_MARSHALLING(                                   \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  GENERATE_OPTION_WITH_MARSHALLING(                                            \
      Args, SA, KIND, FLAGS, SPELLING, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE,    \
      IMPLIED_CHECK, IMPLIED_VALUE, DENORMALIZER, EXTRACTOR, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef FILE_SYSTEM_OPTION_WITH_MARSHALLING
}

static bool ParseFileSystemArgs(FileSystemOptions &Opts, const ArgList &Args,
                                DiagnosticsEngine &Diags) {
  unsigned NumErrorsBefore = Diags.getNumErrors();

  FileSystemOptions &FileSystemOpts = Opts;

#define FILE_SYSTEM_OPTION_WITH_MARSHALLING(                                   \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(                                               \
      Args, Diags, ID, FLAGS, PARAM, SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,     \
      IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef FILE_SYSTEM_OPTION_WITH_MARSHALLING

  return Diags.getNumErrors() == NumErrorsBefore;
}

static void GenerateMigratorArgs(const MigratorOptions &Opts,
                                 SmallVectorImpl<const char *> &Args,
                                 CompilerInvocation::StringAllocator SA) {
  const MigratorOptions &MigratorOpts = Opts;
#define MIGRATOR_OPTION_WITH_MARSHALLING(                                      \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  GENERATE_OPTION_WITH_MARSHALLING(                                            \
      Args, SA, KIND, FLAGS, SPELLING, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE,    \
      IMPLIED_CHECK, IMPLIED_VALUE, DENORMALIZER, EXTRACTOR, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef MIGRATOR_OPTION_WITH_MARSHALLING
}

static bool ParseMigratorArgs(MigratorOptions &Opts, const ArgList &Args,
                              DiagnosticsEngine &Diags) {
  unsigned NumErrorsBefore = Diags.getNumErrors();

  MigratorOptions &MigratorOpts = Opts;

#define MIGRATOR_OPTION_WITH_MARSHALLING(                                      \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(                                               \
      Args, Diags, ID, FLAGS, PARAM, SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,     \
      IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef MIGRATOR_OPTION_WITH_MARSHALLING

  return Diags.getNumErrors() == NumErrorsBefore;
}

void CompilerInvocation::GenerateDiagnosticArgs(
    const DiagnosticOptions &Opts, SmallVectorImpl<const char *> &Args,
    StringAllocator SA, bool DefaultDiagColor) {
  const DiagnosticOptions *DiagnosticOpts = &Opts;
#define DIAG_OPTION_WITH_MARSHALLING(                                          \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  GENERATE_OPTION_WITH_MARSHALLING(                                            \
      Args, SA, KIND, FLAGS, SPELLING, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE,    \
      IMPLIED_CHECK, IMPLIED_VALUE, DENORMALIZER, EXTRACTOR, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef DIAG_OPTION_WITH_MARSHALLING

  if (!Opts.DiagnosticSerializationFile.empty())
    GenerateArg(Args, OPT_diagnostic_serialized_file,
                Opts.DiagnosticSerializationFile, SA);

  if (Opts.ShowColors)
    GenerateArg(Args, OPT_fcolor_diagnostics, SA);

  if (Opts.VerifyDiagnostics &&
      llvm::is_contained(Opts.VerifyPrefixes, "expected"))
    GenerateArg(Args, OPT_verify, SA);

  for (const auto &Prefix : Opts.VerifyPrefixes)
    if (Prefix != "expected")
      GenerateArg(Args, OPT_verify_EQ, Prefix, SA);

  DiagnosticLevelMask VIU = Opts.getVerifyIgnoreUnexpected();
  if (VIU == DiagnosticLevelMask::None) {
    // This is the default, don't generate anything.
  } else if (VIU == DiagnosticLevelMask::All) {
    GenerateArg(Args, OPT_verify_ignore_unexpected, SA);
  } else {
    if (static_cast<unsigned>(VIU & DiagnosticLevelMask::Note) != 0)
      GenerateArg(Args, OPT_verify_ignore_unexpected_EQ, "note", SA);
    if (static_cast<unsigned>(VIU & DiagnosticLevelMask::Remark) != 0)
      GenerateArg(Args, OPT_verify_ignore_unexpected_EQ, "remark", SA);
    if (static_cast<unsigned>(VIU & DiagnosticLevelMask::Warning) != 0)
      GenerateArg(Args, OPT_verify_ignore_unexpected_EQ, "warning", SA);
    if (static_cast<unsigned>(VIU & DiagnosticLevelMask::Error) != 0)
      GenerateArg(Args, OPT_verify_ignore_unexpected_EQ, "error", SA);
  }

  for (const auto &Warning : Opts.Warnings) {
    // This option is automatically generated from UndefPrefixes.
    if (Warning == "undef-prefix")
      continue;
    Args.push_back(SA(StringRef("-W") + Warning));
  }

  for (const auto &Remark : Opts.Remarks) {
    // These arguments are generated from OptimizationRemark fields of
    // CodeGenOptions.
    StringRef IgnoredRemarks[] = {"pass",          "no-pass",
                                  "pass-analysis", "no-pass-analysis",
                                  "pass-missed",   "no-pass-missed"};
    if (llvm::is_contained(IgnoredRemarks, Remark))
      continue;

    Args.push_back(SA(StringRef("-R") + Remark));
  }
}

std::unique_ptr<DiagnosticOptions>
clang::CreateAndPopulateDiagOpts(ArrayRef<const char *> Argv) {
  auto DiagOpts = std::make_unique<DiagnosticOptions>();
  unsigned MissingArgIndex, MissingArgCount;
  InputArgList Args = getDriverOptTable().ParseArgs(
      Argv.slice(1), MissingArgIndex, MissingArgCount);
  // We ignore MissingArgCount and the return value of ParseDiagnosticArgs.
  // Any errors that would be diagnosed here will also be diagnosed later,
  // when the DiagnosticsEngine actually exists.
  (void)ParseDiagnosticArgs(*DiagOpts, Args);
  return DiagOpts;
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

  unsigned NumErrorsBefore = Diags->getNumErrors();

  // The key paths of diagnostic options defined in Options.td start with
  // "DiagnosticOpts->". Let's provide the expected variable name and type.
  DiagnosticOptions *DiagnosticOpts = &Opts;

#define DIAG_OPTION_WITH_MARSHALLING(                                          \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(                                               \
      Args, *Diags, ID, FLAGS, PARAM, SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,    \
      IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef DIAG_OPTION_WITH_MARSHALLING

  llvm::sys::Process::UseANSIEscapeCodes(Opts.UseANSIEscapeCodes);

  if (Arg *A =
          Args.getLastArg(OPT_diagnostic_serialized_file, OPT__serialize_diags))
    Opts.DiagnosticSerializationFile = A->getValue();
  Opts.ShowColors = parseShowColorsArgs(Args, DefaultDiagColor);

  Opts.VerifyDiagnostics = Args.hasArg(OPT_verify) || Args.hasArg(OPT_verify_EQ);
  Opts.VerifyPrefixes = Args.getAllArgValues(OPT_verify_EQ);
  if (Args.hasArg(OPT_verify))
    Opts.VerifyPrefixes.push_back("expected");
  // Keep VerifyPrefixes in its original order for the sake of diagnostics, and
  // then sort it to prepare for fast lookup using std::binary_search.
  if (!checkVerifyPrefixes(Opts.VerifyPrefixes, *Diags))
    Opts.VerifyDiagnostics = false;
  else
    llvm::sort(Opts.VerifyPrefixes);
  DiagnosticLevelMask DiagMask = DiagnosticLevelMask::None;
  parseDiagnosticLevelMask(
      "-verify-ignore-unexpected=",
      Args.getAllArgValues(OPT_verify_ignore_unexpected_EQ), *Diags, DiagMask);
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

  return Diags->getNumErrors() == NumErrorsBefore;
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

/// Return a table that associates command line option specifiers with the
/// frontend action. Note: The pair {frontend::PluginAction, OPT_plugin} is
/// intentionally missing, as this case is handled separately from other
/// frontend options.
static const auto &getFrontendActionTable() {
  static const std::pair<frontend::ActionKind, unsigned> Table[] = {
      {frontend::ASTDeclList, OPT_ast_list},

      {frontend::ASTDump, OPT_ast_dump_all_EQ},
      {frontend::ASTDump, OPT_ast_dump_all},
      {frontend::ASTDump, OPT_ast_dump_EQ},
      {frontend::ASTDump, OPT_ast_dump},
      {frontend::ASTDump, OPT_ast_dump_lookups},
      {frontend::ASTDump, OPT_ast_dump_decl_types},

      {frontend::ASTPrint, OPT_ast_print},
      {frontend::ASTView, OPT_ast_view},
      {frontend::DumpCompilerOptions, OPT_compiler_options_dump},
      {frontend::DumpRawTokens, OPT_dump_raw_tokens},
      {frontend::DumpTokens, OPT_dump_tokens},
      {frontend::EmitAssembly, OPT_S},
      {frontend::EmitBC, OPT_emit_llvm_bc},
      {frontend::EmitHTML, OPT_emit_html},
      {frontend::EmitLLVM, OPT_emit_llvm},
      {frontend::EmitLLVMOnly, OPT_emit_llvm_only},
      {frontend::EmitCodeGenOnly, OPT_emit_codegen_only},
      {frontend::EmitObj, OPT_emit_obj},
      {frontend::ExtractAPI, OPT_extract_api},

      {frontend::FixIt, OPT_fixit_EQ},
      {frontend::FixIt, OPT_fixit},

      {frontend::GenerateModule, OPT_emit_module},
      {frontend::GenerateModuleInterface, OPT_emit_module_interface},
      {frontend::GenerateHeaderModule, OPT_emit_header_module},
      {frontend::GenerateHeaderUnit, OPT_emit_header_unit},
      {frontend::GeneratePCH, OPT_emit_pch},
      {frontend::GenerateInterfaceStubs, OPT_emit_interface_stubs},
      {frontend::InitOnly, OPT_init_only},
      {frontend::ParseSyntaxOnly, OPT_fsyntax_only},
      {frontend::ModuleFileInfo, OPT_module_file_info},
      {frontend::VerifyPCH, OPT_verify_pch},
      {frontend::PrintPreamble, OPT_print_preamble},
      {frontend::PrintPreprocessedInput, OPT_E},
      {frontend::TemplightDump, OPT_templight_dump},
      {frontend::RewriteMacros, OPT_rewrite_macros},
      {frontend::RewriteObjC, OPT_rewrite_objc},
      {frontend::RewriteTest, OPT_rewrite_test},
      {frontend::RunAnalysis, OPT_analyze},
      {frontend::MigrateSource, OPT_migrate},
      {frontend::RunPreprocessorOnly, OPT_Eonly},
      {frontend::PrintDependencyDirectivesSourceMinimizerOutput,
       OPT_print_dependency_directives_minimized_source},
  };

  return Table;
}

/// Maps command line option to frontend action.
static Optional<frontend::ActionKind> getFrontendAction(OptSpecifier &Opt) {
  for (const auto &ActionOpt : getFrontendActionTable())
    if (ActionOpt.second == Opt.getID())
      return ActionOpt.first;

  return None;
}

/// Maps frontend action to command line option.
static Optional<OptSpecifier>
getProgramActionOpt(frontend::ActionKind ProgramAction) {
  for (const auto &ActionOpt : getFrontendActionTable())
    if (ActionOpt.first == ProgramAction)
      return OptSpecifier(ActionOpt.second);

  return None;
}

static void GenerateFrontendArgs(const FrontendOptions &Opts,
                                 SmallVectorImpl<const char *> &Args,
                                 CompilerInvocation::StringAllocator SA,
                                 bool IsHeader) {
  const FrontendOptions &FrontendOpts = Opts;
#define FRONTEND_OPTION_WITH_MARSHALLING(                                      \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  GENERATE_OPTION_WITH_MARSHALLING(                                            \
      Args, SA, KIND, FLAGS, SPELLING, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE,    \
      IMPLIED_CHECK, IMPLIED_VALUE, DENORMALIZER, EXTRACTOR, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef FRONTEND_OPTION_WITH_MARSHALLING

  Optional<OptSpecifier> ProgramActionOpt =
      getProgramActionOpt(Opts.ProgramAction);

  // Generating a simple flag covers most frontend actions.
  std::function<void()> GenerateProgramAction = [&]() {
    GenerateArg(Args, *ProgramActionOpt, SA);
  };

  if (!ProgramActionOpt) {
    // PluginAction is the only program action handled separately.
    assert(Opts.ProgramAction == frontend::PluginAction &&
           "Frontend action without option.");
    GenerateProgramAction = [&]() {
      GenerateArg(Args, OPT_plugin, Opts.ActionName, SA);
    };
  }

  // FIXME: Simplify the complex 'AST dump' command line.
  if (Opts.ProgramAction == frontend::ASTDump) {
    GenerateProgramAction = [&]() {
      // ASTDumpLookups, ASTDumpDeclTypes and ASTDumpFilter are generated via
      // marshalling infrastructure.

      if (Opts.ASTDumpFormat != ADOF_Default) {
        StringRef Format;
        switch (Opts.ASTDumpFormat) {
        case ADOF_Default:
          llvm_unreachable("Default AST dump format.");
        case ADOF_JSON:
          Format = "json";
          break;
        }

        if (Opts.ASTDumpAll)
          GenerateArg(Args, OPT_ast_dump_all_EQ, Format, SA);
        if (Opts.ASTDumpDecls)
          GenerateArg(Args, OPT_ast_dump_EQ, Format, SA);
      } else {
        if (Opts.ASTDumpAll)
          GenerateArg(Args, OPT_ast_dump_all, SA);
        if (Opts.ASTDumpDecls)
          GenerateArg(Args, OPT_ast_dump, SA);
      }
    };
  }

  if (Opts.ProgramAction == frontend::FixIt && !Opts.FixItSuffix.empty()) {
    GenerateProgramAction = [&]() {
      GenerateArg(Args, OPT_fixit_EQ, Opts.FixItSuffix, SA);
    };
  }

  GenerateProgramAction();

  for (const auto &PluginArgs : Opts.PluginArgs) {
    Option Opt = getDriverOptTable().getOption(OPT_plugin_arg);
    const char *Spelling =
        SA(Opt.getPrefix() + Opt.getName() + PluginArgs.first);
    for (const auto &PluginArg : PluginArgs.second)
      denormalizeString(Args, Spelling, SA, Opt.getKind(), 0, PluginArg);
  }

  for (const auto &Ext : Opts.ModuleFileExtensions)
    if (auto *TestExt = dyn_cast_or_null<TestModuleFileExtension>(Ext.get()))
      GenerateArg(Args, OPT_ftest_module_file_extension_EQ, TestExt->str(), SA);

  if (!Opts.CodeCompletionAt.FileName.empty())
    GenerateArg(Args, OPT_code_completion_at, Opts.CodeCompletionAt.ToString(),
                SA);

  for (const auto &Plugin : Opts.Plugins)
    GenerateArg(Args, OPT_load, Plugin, SA);

  // ASTDumpDecls and ASTDumpAll already handled with ProgramAction.

  for (const auto &ModuleFile : Opts.ModuleFiles)
    GenerateArg(Args, OPT_fmodule_file, ModuleFile, SA);

  if (Opts.AuxTargetCPU)
    GenerateArg(Args, OPT_aux_target_cpu, *Opts.AuxTargetCPU, SA);

  if (Opts.AuxTargetFeatures)
    for (const auto &Feature : *Opts.AuxTargetFeatures)
      GenerateArg(Args, OPT_aux_target_feature, Feature, SA);

  {
    StringRef Preprocessed = Opts.DashX.isPreprocessed() ? "-cpp-output" : "";
    StringRef ModuleMap =
        Opts.DashX.getFormat() == InputKind::ModuleMap ? "-module-map" : "";
    StringRef HeaderUnit = "";
    switch (Opts.DashX.getHeaderUnitKind()) {
    case InputKind::HeaderUnit_None:
      break;
    case InputKind::HeaderUnit_User:
      HeaderUnit = "-user";
      break;
    case InputKind::HeaderUnit_System:
      HeaderUnit = "-system";
      break;
    case InputKind::HeaderUnit_Abs:
      HeaderUnit = "-header-unit";
      break;
    }
    StringRef Header = IsHeader ? "-header" : "";

    StringRef Lang;
    switch (Opts.DashX.getLanguage()) {
    case Language::C:
      Lang = "c";
      break;
    case Language::OpenCL:
      Lang = "cl";
      break;
    case Language::OpenCLCXX:
      Lang = "clcpp";
      break;
    case Language::CUDA:
      Lang = "cuda";
      break;
    case Language::HIP:
      Lang = "hip";
      break;
    case Language::CXX:
      Lang = "c++";
      break;
    case Language::ObjC:
      Lang = "objective-c";
      break;
    case Language::ObjCXX:
      Lang = "objective-c++";
      break;
    case Language::RenderScript:
      Lang = "renderscript";
      break;
    case Language::Asm:
      Lang = "assembler-with-cpp";
      break;
    case Language::Unknown:
      assert(Opts.DashX.getFormat() == InputKind::Precompiled &&
             "Generating -x argument for unknown language (not precompiled).");
      Lang = "ast";
      break;
    case Language::LLVM_IR:
      Lang = "ir";
      break;
    case Language::HLSL:
      Lang = "hlsl";
      break;
    }

    GenerateArg(Args, OPT_x,
                Lang + HeaderUnit + Header + ModuleMap + Preprocessed, SA);
  }

  // OPT_INPUT has a unique class, generate it directly.
  for (const auto &Input : Opts.Inputs)
    Args.push_back(SA(Input.getFile()));
}

static bool ParseFrontendArgs(FrontendOptions &Opts, ArgList &Args,
                              DiagnosticsEngine &Diags, bool &IsHeaderFile) {
  unsigned NumErrorsBefore = Diags.getNumErrors();

  FrontendOptions &FrontendOpts = Opts;

#define FRONTEND_OPTION_WITH_MARSHALLING(                                      \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(                                               \
      Args, Diags, ID, FLAGS, PARAM, SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,     \
      IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef FRONTEND_OPTION_WITH_MARSHALLING

  Opts.ProgramAction = frontend::ParseSyntaxOnly;
  if (const Arg *A = Args.getLastArg(OPT_Action_Group)) {
    OptSpecifier Opt = OptSpecifier(A->getOption().getID());
    Optional<frontend::ActionKind> ProgramAction = getFrontendAction(Opt);
    assert(ProgramAction && "Option specifier not in Action_Group.");

    if (ProgramAction == frontend::ASTDump &&
        (Opt == OPT_ast_dump_all_EQ || Opt == OPT_ast_dump_EQ)) {
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
    }

    if (ProgramAction == frontend::FixIt && Opt == OPT_fixit_EQ)
      Opts.FixItSuffix = A->getValue();

    if (ProgramAction == frontend::GenerateInterfaceStubs) {
      StringRef ArgStr =
          Args.hasArg(OPT_interface_stub_version_EQ)
              ? Args.getLastArgValue(OPT_interface_stub_version_EQ)
              : "ifs-v1";
      if (ArgStr == "experimental-yaml-elf-v1" ||
          ArgStr == "experimental-ifs-v1" || ArgStr == "experimental-ifs-v2" ||
          ArgStr == "experimental-tapi-elf-v1") {
        std::string ErrorMessage =
            "Invalid interface stub format: " + ArgStr.str() +
            " is deprecated.";
        Diags.Report(diag::err_drv_invalid_value)
            << "Must specify a valid interface stub format type, ie: "
               "-interface-stub-version=ifs-v1"
            << ErrorMessage;
        ProgramAction = frontend::ParseSyntaxOnly;
      } else if (!ArgStr.startswith("ifs-")) {
        std::string ErrorMessage =
            "Invalid interface stub format: " + ArgStr.str() + ".";
        Diags.Report(diag::err_drv_invalid_value)
            << "Must specify a valid interface stub format type, ie: "
               "-interface-stub-version=ifs-v1"
            << ErrorMessage;
        ProgramAction = frontend::ParseSyntaxOnly;
      }
    }

    Opts.ProgramAction = *ProgramAction;
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
    if (!Val.contains('='))
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

    // Parse suffixes:
    // '<lang>(-[{header-unit,user,system}-]header|[-module-map][-cpp-output])'.
    // FIXME: Supporting '<lang>-header-cpp-output' would be useful.
    bool Preprocessed = XValue.consume_back("-cpp-output");
    bool ModuleMap = XValue.consume_back("-module-map");
    // Detect and consume the header indicator.
    bool IsHeader =
        XValue != "precompiled-header" && XValue.consume_back("-header");

    // If we have c++-{user,system}-header, that indicates a header unit input
    // likewise, if the user put -fmodule-header together with a header with an
    // absolute path (header-unit-header).
    InputKind::HeaderUnitKind HUK = InputKind::HeaderUnit_None;
    if (IsHeader || Preprocessed) {
      if (XValue.consume_back("-header-unit"))
        HUK = InputKind::HeaderUnit_Abs;
      else if (XValue.consume_back("-system"))
        HUK = InputKind::HeaderUnit_System;
      else if (XValue.consume_back("-user"))
        HUK = InputKind::HeaderUnit_User;
    }

    // The value set by this processing is an un-preprocessed source which is
    // not intended to be a module map or header unit.
    IsHeaderFile = IsHeader && !Preprocessed && !ModuleMap &&
                   HUK == InputKind::HeaderUnit_None;

    // Principal languages.
    DashX = llvm::StringSwitch<InputKind>(XValue)
                .Case("c", Language::C)
                .Case("cl", Language::OpenCL)
                .Case("clcpp", Language::OpenCLCXX)
                .Case("cuda", Language::CUDA)
                .Case("hip", Language::HIP)
                .Case("c++", Language::CXX)
                .Case("objective-c", Language::ObjC)
                .Case("objective-c++", Language::ObjCXX)
                .Case("renderscript", Language::RenderScript)
                .Case("hlsl", Language::HLSL)
                .Default(Language::Unknown);

    // "objc[++]-cpp-output" is an acceptable synonym for
    // "objective-c[++]-cpp-output".
    if (DashX.isUnknown() && Preprocessed && !IsHeaderFile && !ModuleMap &&
        HUK == InputKind::HeaderUnit_None)
      DashX = llvm::StringSwitch<InputKind>(XValue)
                  .Case("objc", Language::ObjC)
                  .Case("objc++", Language::ObjCXX)
                  .Default(Language::Unknown);

    // Some special cases cannot be combined with suffixes.
    if (DashX.isUnknown() && !Preprocessed && !IsHeaderFile && !ModuleMap &&
        HUK == InputKind::HeaderUnit_None)
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
    // A regular header is considered mutually exclusive with a header unit.
    if (HUK != InputKind::HeaderUnit_None) {
      DashX = DashX.withHeaderUnit(HUK);
      IsHeaderFile = true;
    } else if (IsHeaderFile)
      DashX = DashX.getHeader();
    if (ModuleMap)
      DashX = DashX.withFormat(InputKind::ModuleMap);
  }

  // '-' is the default input if none is given.
  std::vector<std::string> Inputs = Args.getAllArgValues(OPT_INPUT);
  Opts.Inputs.clear();
  if (Inputs.empty())
    Inputs.push_back("-");

  if (DashX.getHeaderUnitKind() != InputKind::HeaderUnit_None &&
      Inputs.size() > 1)
    Diags.Report(diag::err_drv_header_unit_extra_inputs) << Inputs[1];

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

  Opts.DashX = DashX;

  return Diags.getNumErrors() == NumErrorsBefore;
}

std::string CompilerInvocation::GetResourcesPath(const char *Argv0,
                                                 void *MainAddr) {
  std::string ClangExecutable =
      llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
  return Driver::GetResourcesPath(ClangExecutable, CLANG_RESOURCE_DIR);
}

static void GenerateHeaderSearchArgs(HeaderSearchOptions &Opts,
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

  if (Opts.UseLibcxx)
    GenerateArg(Args, OPT_stdlib_EQ, "libc++", SA);

  if (!Opts.ModuleCachePath.empty())
    GenerateArg(Args, OPT_fmodules_cache_path, Opts.ModuleCachePath, SA);

  for (const auto &File : Opts.PrebuiltModuleFiles)
    GenerateArg(Args, OPT_fmodule_file, File.first + "=" + File.second, SA);

  for (const auto &Path : Opts.PrebuiltModulePaths)
    GenerateArg(Args, OPT_fprebuilt_module_path, Path, SA);

  for (const auto &Macro : Opts.ModulesIgnoreMacros)
    GenerateArg(Args, OPT_fmodules_ignore_macro, Macro.val(), SA);

  auto Matches = [](const HeaderSearchOptions::Entry &Entry,
                    llvm::ArrayRef<frontend::IncludeDirGroup> Groups,
                    llvm::Optional<bool> IsFramework,
                    llvm::Optional<bool> IgnoreSysRoot) {
    return llvm::is_contained(Groups, Entry.Group) &&
           (!IsFramework || (Entry.IsFramework == *IsFramework)) &&
           (!IgnoreSysRoot || (Entry.IgnoreSysRoot == *IgnoreSysRoot));
  };

  auto It = Opts.UserEntries.begin();
  auto End = Opts.UserEntries.end();

  // Add -I..., -F..., and -index-header-map options in order.
  for (; It < End &&
         Matches(*It, {frontend::IndexHeaderMap, frontend::Angled}, None, true);
       ++It) {
    OptSpecifier Opt = [It, Matches]() {
      if (Matches(*It, frontend::IndexHeaderMap, true, true))
        return OPT_F;
      if (Matches(*It, frontend::IndexHeaderMap, false, true))
        return OPT_I;
      if (Matches(*It, frontend::Angled, true, true))
        return OPT_F;
      if (Matches(*It, frontend::Angled, false, true))
        return OPT_I;
      llvm_unreachable("Unexpected HeaderSearchOptions::Entry.");
    }();

    if (It->Group == frontend::IndexHeaderMap)
      GenerateArg(Args, OPT_index_header_map, SA);
    GenerateArg(Args, Opt, It->Path, SA);
  };

  // Note: some paths that came from "[-iprefix=xx] -iwithprefixbefore=yy" may
  // have already been generated as "-I[xx]yy". If that's the case, their
  // position on command line was such that this has no semantic impact on
  // include paths.
  for (; It < End &&
         Matches(*It, {frontend::After, frontend::Angled}, false, true);
       ++It) {
    OptSpecifier Opt =
        It->Group == frontend::After ? OPT_iwithprefix : OPT_iwithprefixbefore;
    GenerateArg(Args, Opt, It->Path, SA);
  }

  // Note: Some paths that came from "-idirafter=xxyy" may have already been
  // generated as "-iwithprefix=xxyy". If that's the case, their position on
  // command line was such that this has no semantic impact on include paths.
  for (; It < End && Matches(*It, {frontend::After}, false, true); ++It)
    GenerateArg(Args, OPT_idirafter, It->Path, SA);
  for (; It < End && Matches(*It, {frontend::Quoted}, false, true); ++It)
    GenerateArg(Args, OPT_iquote, It->Path, SA);
  for (; It < End && Matches(*It, {frontend::System}, false, None); ++It)
    GenerateArg(Args, It->IgnoreSysRoot ? OPT_isystem : OPT_iwithsysroot,
                It->Path, SA);
  for (; It < End && Matches(*It, {frontend::System}, true, true); ++It)
    GenerateArg(Args, OPT_iframework, It->Path, SA);
  for (; It < End && Matches(*It, {frontend::System}, true, false); ++It)
    GenerateArg(Args, OPT_iframeworkwithsysroot, It->Path, SA);

  // Add the paths for the various language specific isystem flags.
  for (; It < End && Matches(*It, {frontend::CSystem}, false, true); ++It)
    GenerateArg(Args, OPT_c_isystem, It->Path, SA);
  for (; It < End && Matches(*It, {frontend::CXXSystem}, false, true); ++It)
    GenerateArg(Args, OPT_cxx_isystem, It->Path, SA);
  for (; It < End && Matches(*It, {frontend::ObjCSystem}, false, true); ++It)
    GenerateArg(Args, OPT_objc_isystem, It->Path, SA);
  for (; It < End && Matches(*It, {frontend::ObjCXXSystem}, false, true); ++It)
    GenerateArg(Args, OPT_objcxx_isystem, It->Path, SA);

  // Add the internal paths from a driver that detects standard include paths.
  // Note: Some paths that came from "-internal-isystem" arguments may have
  // already been generated as "-isystem". If that's the case, their position on
  // command line was such that this has no semantic impact on include paths.
  for (; It < End &&
         Matches(*It, {frontend::System, frontend::ExternCSystem}, false, true);
       ++It) {
    OptSpecifier Opt = It->Group == frontend::System
                           ? OPT_internal_isystem
                           : OPT_internal_externc_isystem;
    GenerateArg(Args, Opt, It->Path, SA);
  }

  assert(It == End && "Unhandled HeaderSearchOption::Entry.");

  // Add the path prefixes which are implicitly treated as being system headers.
  for (const auto &P : Opts.SystemHeaderPrefixes) {
    OptSpecifier Opt = P.IsSystemHeader ? OPT_system_header_prefix
                                        : OPT_no_system_header_prefix;
    GenerateArg(Args, Opt, P.Prefix, SA);
  }

  for (const std::string &F : Opts.VFSOverlayFiles)
    GenerateArg(Args, OPT_ivfsoverlay, F, SA);
}

static bool ParseHeaderSearchArgs(HeaderSearchOptions &Opts, ArgList &Args,
                                  DiagnosticsEngine &Diags,
                                  const std::string &WorkingDir) {
  unsigned NumErrorsBefore = Diags.getNumErrors();

  HeaderSearchOptions *HeaderSearchOpts = &Opts;

#define HEADER_SEARCH_OPTION_WITH_MARSHALLING(                                 \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(                                               \
      Args, Diags, ID, FLAGS, PARAM, SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,     \
      IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, MERGER, TABLE_INDEX)
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
    if (Val.contains('=')) {
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

  return Diags.getNumErrors() == NumErrorsBefore;
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
    return S.getLanguage() == Language::OpenCL ||
           S.getLanguage() == Language::OpenCLCXX;

  case Language::OpenCLCXX:
    return S.getLanguage() == Language::OpenCLCXX;

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

  case Language::HLSL:
    return S.getLanguage() == Language::HLSL;
  }

  llvm_unreachable("unexpected input language");
}

/// Get language name for given input kind.
static StringRef GetInputKindName(InputKind IK) {
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
  case Language::OpenCLCXX:
    return "C++ for OpenCL";
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

  case Language::HLSL:
    return "HLSL";

  case Language::Unknown:
    break;
  }
  llvm_unreachable("unknown input language");
}

void CompilerInvocation::GenerateLangArgs(const LangOptions &Opts,
                                          SmallVectorImpl<const char *> &Args,
                                          StringAllocator SA,
                                          const llvm::Triple &T, InputKind IK) {
  if (IK.getFormat() == InputKind::Precompiled ||
      IK.getLanguage() == Language::LLVM_IR) {
    if (Opts.ObjCAutoRefCount)
      GenerateArg(Args, OPT_fobjc_arc, SA);
    if (Opts.PICLevel != 0)
      GenerateArg(Args, OPT_pic_level, Twine(Opts.PICLevel), SA);
    if (Opts.PIE)
      GenerateArg(Args, OPT_pic_is_pie, SA);
    for (StringRef Sanitizer : serializeSanitizerKinds(Opts.Sanitize))
      GenerateArg(Args, OPT_fsanitize_EQ, Sanitizer, SA);

    return;
  }

  OptSpecifier StdOpt;
  switch (Opts.LangStd) {
  case LangStandard::lang_opencl10:
  case LangStandard::lang_opencl11:
  case LangStandard::lang_opencl12:
  case LangStandard::lang_opencl20:
  case LangStandard::lang_opencl30:
  case LangStandard::lang_openclcpp10:
  case LangStandard::lang_openclcpp2021:
    StdOpt = OPT_cl_std_EQ;
    break;
  default:
    StdOpt = OPT_std_EQ;
    break;
  }

  auto LangStandard = LangStandard::getLangStandardForKind(Opts.LangStd);
  GenerateArg(Args, StdOpt, LangStandard.getName(), SA);

  if (Opts.IncludeDefaultHeader)
    GenerateArg(Args, OPT_finclude_default_header, SA);
  if (Opts.DeclareOpenCLBuiltins)
    GenerateArg(Args, OPT_fdeclare_opencl_builtins, SA);

  const LangOptions *LangOpts = &Opts;

#define LANG_OPTION_WITH_MARSHALLING(                                          \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  GENERATE_OPTION_WITH_MARSHALLING(                                            \
      Args, SA, KIND, FLAGS, SPELLING, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE,    \
      IMPLIED_CHECK, IMPLIED_VALUE, DENORMALIZER, EXTRACTOR, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef LANG_OPTION_WITH_MARSHALLING

  // The '-fcf-protection=' option is generated by CodeGenOpts generator.

  if (Opts.ObjC) {
    GenerateArg(Args, OPT_fobjc_runtime_EQ, Opts.ObjCRuntime.getAsString(), SA);

    if (Opts.GC == LangOptions::GCOnly)
      GenerateArg(Args, OPT_fobjc_gc_only, SA);
    else if (Opts.GC == LangOptions::HybridGC)
      GenerateArg(Args, OPT_fobjc_gc, SA);
    else if (Opts.ObjCAutoRefCount == 1)
      GenerateArg(Args, OPT_fobjc_arc, SA);

    if (Opts.ObjCWeakRuntime)
      GenerateArg(Args, OPT_fobjc_runtime_has_weak, SA);

    if (Opts.ObjCWeak)
      GenerateArg(Args, OPT_fobjc_weak, SA);

    if (Opts.ObjCSubscriptingLegacyRuntime)
      GenerateArg(Args, OPT_fobjc_subscripting_legacy_runtime, SA);
  }

  if (Opts.GNUCVersion != 0) {
    unsigned Major = Opts.GNUCVersion / 100 / 100;
    unsigned Minor = (Opts.GNUCVersion / 100) % 100;
    unsigned Patch = Opts.GNUCVersion % 100;
    GenerateArg(Args, OPT_fgnuc_version_EQ,
                Twine(Major) + "." + Twine(Minor) + "." + Twine(Patch), SA);
  }

  if (Opts.IgnoreXCOFFVisibility)
    GenerateArg(Args, OPT_mignore_xcoff_visibility, SA);

  if (Opts.SignedOverflowBehavior == LangOptions::SOB_Trapping) {
    GenerateArg(Args, OPT_ftrapv, SA);
    GenerateArg(Args, OPT_ftrapv_handler, Opts.OverflowHandler, SA);
  } else if (Opts.SignedOverflowBehavior == LangOptions::SOB_Defined) {
    GenerateArg(Args, OPT_fwrapv, SA);
  }

  if (Opts.MSCompatibilityVersion != 0) {
    unsigned Major = Opts.MSCompatibilityVersion / 10000000;
    unsigned Minor = (Opts.MSCompatibilityVersion / 100000) % 100;
    unsigned Subminor = Opts.MSCompatibilityVersion % 100000;
    GenerateArg(Args, OPT_fms_compatibility_version,
                Twine(Major) + "." + Twine(Minor) + "." + Twine(Subminor), SA);
  }

  if ((!Opts.GNUMode && !Opts.MSVCCompat && !Opts.CPlusPlus17) || T.isOSzOS()) {
    if (!Opts.Trigraphs)
      GenerateArg(Args, OPT_fno_trigraphs, SA);
  } else {
    if (Opts.Trigraphs)
      GenerateArg(Args, OPT_ftrigraphs, SA);
  }

  if (Opts.Blocks && !(Opts.OpenCL && Opts.OpenCLVersion == 200))
    GenerateArg(Args, OPT_fblocks, SA);

  if (Opts.ConvergentFunctions &&
      !(Opts.OpenCL || (Opts.CUDA && Opts.CUDAIsDevice) || Opts.SYCLIsDevice))
    GenerateArg(Args, OPT_fconvergent_functions, SA);

  if (Opts.NoBuiltin && !Opts.Freestanding)
    GenerateArg(Args, OPT_fno_builtin, SA);

  if (!Opts.NoBuiltin)
    for (const auto &Func : Opts.NoBuiltinFuncs)
      GenerateArg(Args, OPT_fno_builtin_, Func, SA);

  if (Opts.LongDoubleSize == 128)
    GenerateArg(Args, OPT_mlong_double_128, SA);
  else if (Opts.LongDoubleSize == 64)
    GenerateArg(Args, OPT_mlong_double_64, SA);
  else if (Opts.LongDoubleSize == 80)
    GenerateArg(Args, OPT_mlong_double_80, SA);

  // Not generating '-mrtd', it's just an alias for '-fdefault-calling-conv='.

  // OpenMP was requested via '-fopenmp', not implied by '-fopenmp-simd' or
  // '-fopenmp-targets='.
  if (Opts.OpenMP && !Opts.OpenMPSimd) {
    GenerateArg(Args, OPT_fopenmp, SA);

    if (Opts.OpenMP != 50)
      GenerateArg(Args, OPT_fopenmp_version_EQ, Twine(Opts.OpenMP), SA);

    if (!Opts.OpenMPUseTLS)
      GenerateArg(Args, OPT_fnoopenmp_use_tls, SA);

    if (Opts.OpenMPIsDevice)
      GenerateArg(Args, OPT_fopenmp_is_device, SA);

    if (Opts.OpenMPIRBuilder)
      GenerateArg(Args, OPT_fopenmp_enable_irbuilder, SA);
  }

  if (Opts.OpenMPSimd) {
    GenerateArg(Args, OPT_fopenmp_simd, SA);

    if (Opts.OpenMP != 50)
      GenerateArg(Args, OPT_fopenmp_version_EQ, Twine(Opts.OpenMP), SA);
  }

  if (Opts.OpenMPThreadSubscription)
    GenerateArg(Args, OPT_fopenmp_assume_threads_oversubscription, SA);

  if (Opts.OpenMPTeamSubscription)
    GenerateArg(Args, OPT_fopenmp_assume_teams_oversubscription, SA);

  if (Opts.OpenMPTargetDebug != 0)
    GenerateArg(Args, OPT_fopenmp_target_debug_EQ,
                Twine(Opts.OpenMPTargetDebug), SA);

  if (Opts.OpenMPCUDANumSMs != 0)
    GenerateArg(Args, OPT_fopenmp_cuda_number_of_sm_EQ,
                Twine(Opts.OpenMPCUDANumSMs), SA);

  if (Opts.OpenMPCUDABlocksPerSM != 0)
    GenerateArg(Args, OPT_fopenmp_cuda_blocks_per_sm_EQ,
                Twine(Opts.OpenMPCUDABlocksPerSM), SA);

  if (Opts.OpenMPCUDAReductionBufNum != 1024)
    GenerateArg(Args, OPT_fopenmp_cuda_teams_reduction_recs_num_EQ,
                Twine(Opts.OpenMPCUDAReductionBufNum), SA);

  if (!Opts.OMPTargetTriples.empty()) {
    std::string Targets;
    llvm::raw_string_ostream OS(Targets);
    llvm::interleave(
        Opts.OMPTargetTriples, OS,
        [&OS](const llvm::Triple &T) { OS << T.str(); }, ",");
    GenerateArg(Args, OPT_fopenmp_targets_EQ, OS.str(), SA);
  }

  if (!Opts.OMPHostIRFile.empty())
    GenerateArg(Args, OPT_fopenmp_host_ir_file_path, Opts.OMPHostIRFile, SA);

  if (Opts.OpenMPCUDAMode)
    GenerateArg(Args, OPT_fopenmp_cuda_mode, SA);

  if (Opts.OpenMPCUDAForceFullRuntime)
    GenerateArg(Args, OPT_fopenmp_cuda_force_full_runtime, SA);

  // The arguments used to set Optimize, OptimizeSize and NoInlineDefine are
  // generated from CodeGenOptions.

  if (Opts.DefaultFPContractMode == LangOptions::FPM_Fast)
    GenerateArg(Args, OPT_ffp_contract, "fast", SA);
  else if (Opts.DefaultFPContractMode == LangOptions::FPM_On)
    GenerateArg(Args, OPT_ffp_contract, "on", SA);
  else if (Opts.DefaultFPContractMode == LangOptions::FPM_Off)
    GenerateArg(Args, OPT_ffp_contract, "off", SA);
  else if (Opts.DefaultFPContractMode == LangOptions::FPM_FastHonorPragmas)
    GenerateArg(Args, OPT_ffp_contract, "fast-honor-pragmas", SA);

  for (StringRef Sanitizer : serializeSanitizerKinds(Opts.Sanitize))
    GenerateArg(Args, OPT_fsanitize_EQ, Sanitizer, SA);

  // Conflating '-fsanitize-system-ignorelist' and '-fsanitize-ignorelist'.
  for (const std::string &F : Opts.NoSanitizeFiles)
    GenerateArg(Args, OPT_fsanitize_ignorelist_EQ, F, SA);

  if (Opts.getClangABICompat() == LangOptions::ClangABI::Ver3_8)
    GenerateArg(Args, OPT_fclang_abi_compat_EQ, "3.8", SA);
  else if (Opts.getClangABICompat() == LangOptions::ClangABI::Ver4)
    GenerateArg(Args, OPT_fclang_abi_compat_EQ, "4.0", SA);
  else if (Opts.getClangABICompat() == LangOptions::ClangABI::Ver6)
    GenerateArg(Args, OPT_fclang_abi_compat_EQ, "6.0", SA);
  else if (Opts.getClangABICompat() == LangOptions::ClangABI::Ver7)
    GenerateArg(Args, OPT_fclang_abi_compat_EQ, "7.0", SA);
  else if (Opts.getClangABICompat() == LangOptions::ClangABI::Ver9)
    GenerateArg(Args, OPT_fclang_abi_compat_EQ, "9.0", SA);
  else if (Opts.getClangABICompat() == LangOptions::ClangABI::Ver11)
    GenerateArg(Args, OPT_fclang_abi_compat_EQ, "11.0", SA);
  else if (Opts.getClangABICompat() == LangOptions::ClangABI::Ver12)
    GenerateArg(Args, OPT_fclang_abi_compat_EQ, "12.0", SA);
  else if (Opts.getClangABICompat() == LangOptions::ClangABI::Ver14)
    GenerateArg(Args, OPT_fclang_abi_compat_EQ, "14.0", SA);

  if (Opts.getSignReturnAddressScope() ==
      LangOptions::SignReturnAddressScopeKind::All)
    GenerateArg(Args, OPT_msign_return_address_EQ, "all", SA);
  else if (Opts.getSignReturnAddressScope() ==
           LangOptions::SignReturnAddressScopeKind::NonLeaf)
    GenerateArg(Args, OPT_msign_return_address_EQ, "non-leaf", SA);

  if (Opts.getSignReturnAddressKey() ==
      LangOptions::SignReturnAddressKeyKind::BKey)
    GenerateArg(Args, OPT_msign_return_address_key_EQ, "b_key", SA);

  if (Opts.CXXABI)
    GenerateArg(Args, OPT_fcxx_abi_EQ, TargetCXXABI::getSpelling(*Opts.CXXABI),
                SA);

  if (Opts.RelativeCXXABIVTables)
    GenerateArg(Args, OPT_fexperimental_relative_cxx_abi_vtables, SA);
  else
    GenerateArg(Args, OPT_fno_experimental_relative_cxx_abi_vtables, SA);

  if (Opts.UseTargetPathSeparator)
    GenerateArg(Args, OPT_ffile_reproducible, SA);
  else
    GenerateArg(Args, OPT_fno_file_reproducible, SA);

  for (const auto &MP : Opts.MacroPrefixMap)
    GenerateArg(Args, OPT_fmacro_prefix_map_EQ, MP.first + "=" + MP.second, SA);

  if (!Opts.RandstructSeed.empty())
    GenerateArg(Args, OPT_frandomize_layout_seed_EQ, Opts.RandstructSeed, SA);
}

bool CompilerInvocation::ParseLangArgs(LangOptions &Opts, ArgList &Args,
                                       InputKind IK, const llvm::Triple &T,
                                       std::vector<std::string> &Includes,
                                       DiagnosticsEngine &Diags) {
  unsigned NumErrorsBefore = Diags.getNumErrors();

  if (IK.getFormat() == InputKind::Precompiled ||
      IK.getLanguage() == Language::LLVM_IR) {
    // ObjCAAutoRefCount and Sanitize LangOpts are used to setup the
    // PassManager in BackendUtil.cpp. They need to be initialized no matter
    // what the input type is.
    if (Args.hasArg(OPT_fobjc_arc))
      Opts.ObjCAutoRefCount = 1;
    // PICLevel and PIELevel are needed during code generation and this should
    // be set regardless of the input type.
    Opts.PICLevel = getLastArgIntValue(Args, OPT_pic_level, 0, Diags);
    Opts.PIE = Args.hasArg(OPT_pic_is_pie);
    parseSanitizerKinds("-fsanitize=", Args.getAllArgValues(OPT_fsanitize_EQ),
                        Diags, Opts.Sanitize);

    return Diags.getNumErrors() == NumErrorsBefore;
  }

  // Other LangOpts are only initialized when the input is not AST or LLVM IR.
  // FIXME: Should we really be parsing this for an Language::Asm input?

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
        .Cases("clc++", "CLC++", LangStandard::lang_openclcpp10)
        .Cases("clc++1.0", "CLC++1.0", LangStandard::lang_openclcpp10)
        .Cases("clc++2021", "CLC++2021", LangStandard::lang_openclcpp2021)
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

  LangOptions::setLangDefaults(Opts, IK.getLanguage(), T, Includes, LangStd);

  // The key paths of codegen options defined in Options.td start with
  // "LangOpts->". Let's provide the expected variable name and type.
  LangOptions *LangOpts = &Opts;

#define LANG_OPTION_WITH_MARSHALLING(                                          \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(                                               \
      Args, Diags, ID, FLAGS, PARAM, SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,     \
      IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef LANG_OPTION_WITH_MARSHALLING

  if (const Arg *A = Args.getLastArg(OPT_fcf_protection_EQ)) {
    StringRef Name = A->getValue();
    if (Name == "full" || Name == "branch") {
      Opts.CFProtectionBranch = 1;
    }
  }

  if ((Args.hasArg(OPT_fsycl_is_device) || Args.hasArg(OPT_fsycl_is_host)) &&
      !Args.hasArg(OPT_sycl_std_EQ)) {
    // If the user supplied -fsycl-is-device or -fsycl-is-host, but failed to
    // provide -sycl-std=, we want to default it to whatever the default SYCL
    // version is. I could not find a way to express this with the options
    // tablegen because we still want this value to be SYCL_None when the user
    // is not in device or host mode.
    Opts.setSYCLVersion(LangOptions::SYCL_Default);
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
    unsigned Minor = GNUCVer.getMinor().value_or(0);
    unsigned Patch = GNUCVer.getSubminor().value_or(0);
    if (Invalid || GNUCVer.getBuild() || Minor >= 100 || Patch >= 100) {
      Diags.Report(diag::err_drv_invalid_value)
          << A->getAsString(Args) << A->getValue();
    }
    Opts.GNUCVersion = Major * 100 * 100 + Minor * 100 + Patch;
  }

  if (T.isOSAIX() && (Args.hasArg(OPT_mignore_xcoff_visibility)))
    Opts.IgnoreXCOFFVisibility = 1;

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
                                  VT.getMinor().value_or(0) * 100000 +
                                  VT.getSubminor().value_or(0);
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
  if (Arg *A = Args.getLastArg(options::OPT_LongDouble_Group)) {
    if (A->getOption().matches(options::OPT_mlong_double_64))
      Opts.LongDoubleSize = 64;
    else if (A->getOption().matches(options::OPT_mlong_double_80))
      Opts.LongDoubleSize = 80;
    else if (A->getOption().matches(options::OPT_mlong_double_128))
      Opts.LongDoubleSize = 128;
    else
      Opts.LongDoubleSize = 0;
  }
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

  // Check if -fopenmp is specified and set default version to 5.0.
  Opts.OpenMP = Args.hasArg(OPT_fopenmp) ? 50 : 0;
  // Check if -fopenmp-simd is specified.
  bool IsSimdSpecified =
      Args.hasFlag(options::OPT_fopenmp_simd, options::OPT_fno_openmp_simd,
                   /*Default=*/false);
  Opts.OpenMPSimd = !Opts.OpenMP && IsSimdSpecified;
  Opts.OpenMPUseTLS =
      Opts.OpenMP && !Args.hasArg(options::OPT_fnoopenmp_use_tls);
  Opts.OpenMPIsDevice =
      Opts.OpenMP && Args.hasArg(options::OPT_fopenmp_is_device);
  Opts.OpenMPIRBuilder =
      Opts.OpenMP && Args.hasArg(options::OPT_fopenmp_enable_irbuilder);
  bool IsTargetSpecified =
      Opts.OpenMPIsDevice || Args.hasArg(options::OPT_fopenmp_targets_EQ);

  Opts.ConvergentFunctions = Opts.ConvergentFunctions || Opts.OpenMPIsDevice;

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

  // Set the value of the debugging flag used in the new offloading device RTL.
  // Set either by a specific value or to a default if not specified.
  if (Opts.OpenMPIsDevice && (Args.hasArg(OPT_fopenmp_target_debug) ||
                              Args.hasArg(OPT_fopenmp_target_debug_EQ))) {
    Opts.OpenMPTargetDebug = getLastArgIntValue(
        Args, OPT_fopenmp_target_debug_EQ, Opts.OpenMPTargetDebug, Diags);
    if (!Opts.OpenMPTargetDebug && Args.hasArg(OPT_fopenmp_target_debug))
      Opts.OpenMPTargetDebug = 1;
  }

  if (Opts.OpenMPIsDevice) {
    if (Args.hasArg(OPT_fopenmp_assume_teams_oversubscription))
      Opts.OpenMPTeamSubscription = true;
    if (Args.hasArg(OPT_fopenmp_assume_threads_oversubscription))
      Opts.OpenMPThreadSubscription = true;
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

  // Get OpenMP host file path if any and report if a non existent file is
  // found
  if (Arg *A = Args.getLastArg(options::OPT_fopenmp_host_ir_file_path)) {
    Opts.OMPHostIRFile = A->getValue();
    if (!llvm::sys::fs::exists(Opts.OMPHostIRFile))
      Diags.Report(diag::err_drv_omp_host_ir_file_not_found)
          << Opts.OMPHostIRFile;
  }

  // Set CUDA mode for OpenMP target NVPTX/AMDGCN if specified in options
  Opts.OpenMPCUDAMode = Opts.OpenMPIsDevice && (T.isNVPTX() || T.isAMDGCN()) &&
                        Args.hasArg(options::OPT_fopenmp_cuda_mode);

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
  Opts.NoSanitizeFiles = Args.getAllArgValues(OPT_fsanitize_ignorelist_EQ);
  std::vector<std::string> systemIgnorelists =
      Args.getAllArgValues(OPT_fsanitize_system_ignorelist_EQ);
  Opts.NoSanitizeFiles.insert(Opts.NoSanitizeFiles.end(),
                              systemIgnorelists.begin(),
                              systemIgnorelists.end());

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
      else if (Major <= 12)
        Opts.setClangABICompat(LangOptions::ClangABI::Ver12);
      else if (Major <= 14)
        Opts.setClangABICompat(LangOptions::ClangABI::Ver14);
    } else if (Ver != "latest") {
      Diags.Report(diag::err_drv_invalid_value)
          << A->getAsString(Args) << A->getValue();
    }
  }

  if (Arg *A = Args.getLastArg(OPT_msign_return_address_EQ)) {
    StringRef SignScope = A->getValue();

    if (SignScope.equals_insensitive("none"))
      Opts.setSignReturnAddressScope(
          LangOptions::SignReturnAddressScopeKind::None);
    else if (SignScope.equals_insensitive("all"))
      Opts.setSignReturnAddressScope(
          LangOptions::SignReturnAddressScopeKind::All);
    else if (SignScope.equals_insensitive("non-leaf"))
      Opts.setSignReturnAddressScope(
          LangOptions::SignReturnAddressScopeKind::NonLeaf);
    else
      Diags.Report(diag::err_drv_invalid_value)
          << A->getAsString(Args) << SignScope;

    if (Arg *A = Args.getLastArg(OPT_msign_return_address_key_EQ)) {
      StringRef SignKey = A->getValue();
      if (!SignScope.empty() && !SignKey.empty()) {
        if (SignKey.equals_insensitive("a_key"))
          Opts.setSignReturnAddressKey(
              LangOptions::SignReturnAddressKeyKind::AKey);
        else if (SignKey.equals_insensitive("b_key"))
          Opts.setSignReturnAddressKey(
              LangOptions::SignReturnAddressKeyKind::BKey);
        else
          Diags.Report(diag::err_drv_invalid_value)
              << A->getAsString(Args) << SignKey;
      }
    }
  }

  // The value can be empty, which indicates the system default should be used.
  StringRef CXXABI = Args.getLastArgValue(OPT_fcxx_abi_EQ);
  if (!CXXABI.empty()) {
    if (!TargetCXXABI::isABI(CXXABI)) {
      Diags.Report(diag::err_invalid_cxx_abi) << CXXABI;
    } else {
      auto Kind = TargetCXXABI::getKind(CXXABI);
      if (!TargetCXXABI::isSupportedCXXABI(T, Kind))
        Diags.Report(diag::err_unsupported_cxx_abi) << CXXABI << T.str();
      else
        Opts.CXXABI = Kind;
    }
  }

  Opts.RelativeCXXABIVTables =
      Args.hasFlag(options::OPT_fexperimental_relative_cxx_abi_vtables,
                   options::OPT_fno_experimental_relative_cxx_abi_vtables,
                   TargetCXXABI::usesRelativeVTables(T));

  for (const auto &A : Args.getAllArgValues(OPT_fmacro_prefix_map_EQ)) {
    auto Split = StringRef(A).split('=');
    Opts.MacroPrefixMap.insert(
        {std::string(Split.first), std::string(Split.second)});
  }

  Opts.UseTargetPathSeparator =
      !Args.getLastArg(OPT_fno_file_reproducible) &&
      (Args.getLastArg(OPT_ffile_compilation_dir_EQ) ||
       Args.getLastArg(OPT_fmacro_prefix_map_EQ) ||
       Args.getLastArg(OPT_ffile_reproducible));

  // Error if -mvscale-min is unbounded.
  if (Arg *A = Args.getLastArg(options::OPT_mvscale_min_EQ)) {
    unsigned VScaleMin;
    if (StringRef(A->getValue()).getAsInteger(10, VScaleMin) || VScaleMin == 0)
      Diags.Report(diag::err_cc1_unbounded_vscale_min);
  }

  if (const Arg *A = Args.getLastArg(OPT_frandomize_layout_seed_file_EQ)) {
    std::ifstream SeedFile(A->getValue(0));

    if (!SeedFile.is_open())
      Diags.Report(diag::err_drv_cannot_open_randomize_layout_seed_file)
          << A->getValue(0);

    std::getline(SeedFile, Opts.RandstructSeed);
  }

  if (const Arg *A = Args.getLastArg(OPT_frandomize_layout_seed_EQ))
    Opts.RandstructSeed = A->getValue(0);

  return Diags.getNumErrors() == NumErrorsBefore;
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
  case frontend::ExtractAPI:
  case frontend::FixIt:
  case frontend::GenerateModule:
  case frontend::GenerateModuleInterface:
  case frontend::GenerateHeaderModule:
  case frontend::GenerateHeaderUnit:
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

static void GeneratePreprocessorArgs(PreprocessorOptions &Opts,
                                     SmallVectorImpl<const char *> &Args,
                                     CompilerInvocation::StringAllocator SA,
                                     const LangOptions &LangOpts,
                                     const FrontendOptions &FrontendOpts,
                                     const CodeGenOptions &CodeGenOpts) {
  PreprocessorOptions *PreprocessorOpts = &Opts;

#define PREPROCESSOR_OPTION_WITH_MARSHALLING(                                  \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  GENERATE_OPTION_WITH_MARSHALLING(                                            \
      Args, SA, KIND, FLAGS, SPELLING, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE,    \
      IMPLIED_CHECK, IMPLIED_VALUE, DENORMALIZER, EXTRACTOR, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef PREPROCESSOR_OPTION_WITH_MARSHALLING

  if (Opts.PCHWithHdrStop && !Opts.PCHWithHdrStopCreate)
    GenerateArg(Args, OPT_pch_through_hdrstop_use, SA);

  for (const auto &D : Opts.DeserializedPCHDeclsToErrorOn)
    GenerateArg(Args, OPT_error_on_deserialized_pch_decl, D, SA);

  if (Opts.PrecompiledPreambleBytes != std::make_pair(0u, false))
    GenerateArg(Args, OPT_preamble_bytes_EQ,
                Twine(Opts.PrecompiledPreambleBytes.first) + "," +
                    (Opts.PrecompiledPreambleBytes.second ? "1" : "0"),
                SA);

  for (const auto &M : Opts.Macros) {
    // Don't generate __CET__ macro definitions. They are implied by the
    // -fcf-protection option that is generated elsewhere.
    if (M.first == "__CET__=1" && !M.second &&
        !CodeGenOpts.CFProtectionReturn && CodeGenOpts.CFProtectionBranch)
      continue;
    if (M.first == "__CET__=2" && !M.second && CodeGenOpts.CFProtectionReturn &&
        !CodeGenOpts.CFProtectionBranch)
      continue;
    if (M.first == "__CET__=3" && !M.second && CodeGenOpts.CFProtectionReturn &&
        CodeGenOpts.CFProtectionBranch)
      continue;

    GenerateArg(Args, M.second ? OPT_U : OPT_D, M.first, SA);
  }

  for (const auto &I : Opts.Includes) {
    // Don't generate OpenCL includes. They are implied by other flags that are
    // generated elsewhere.
    if (LangOpts.OpenCL && LangOpts.IncludeDefaultHeader &&
        ((LangOpts.DeclareOpenCLBuiltins && I == "opencl-c-base.h") ||
         I == "opencl-c.h"))
      continue;
    // Don't generate HLSL includes. They are implied by other flags that are
    // generated elsewhere.
    if (LangOpts.HLSL && I == "hlsl.h")
      continue;

    GenerateArg(Args, OPT_include, I, SA);
  }

  for (const auto &CI : Opts.ChainedIncludes)
    GenerateArg(Args, OPT_chain_include, CI, SA);

  for (const auto &RF : Opts.RemappedFiles)
    GenerateArg(Args, OPT_remap_file, RF.first + ";" + RF.second, SA);

  // Don't handle LexEditorPlaceholders. It is implied by the action that is
  // generated elsewhere.
}

static bool ParsePreprocessorArgs(PreprocessorOptions &Opts, ArgList &Args,
                                  DiagnosticsEngine &Diags,
                                  frontend::ActionKind Action,
                                  const FrontendOptions &FrontendOpts) {
  unsigned NumErrorsBefore = Diags.getNumErrors();

  PreprocessorOptions *PreprocessorOpts = &Opts;

#define PREPROCESSOR_OPTION_WITH_MARSHALLING(                                  \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(                                               \
      Args, Diags, ID, FLAGS, PARAM, SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,     \
      IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef PREPROCESSOR_OPTION_WITH_MARSHALLING

  Opts.PCHWithHdrStop = Args.hasArg(OPT_pch_through_hdrstop_create) ||
                        Args.hasArg(OPT_pch_through_hdrstop_use);

  for (const auto *A : Args.filtered(OPT_error_on_deserialized_pch_decl))
    Opts.DeserializedPCHDeclsToErrorOn.insert(A->getValue());

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

  return Diags.getNumErrors() == NumErrorsBefore;
}

static void GeneratePreprocessorOutputArgs(
    const PreprocessorOutputOptions &Opts, SmallVectorImpl<const char *> &Args,
    CompilerInvocation::StringAllocator SA, frontend::ActionKind Action) {
  const PreprocessorOutputOptions &PreprocessorOutputOpts = Opts;

#define PREPROCESSOR_OUTPUT_OPTION_WITH_MARSHALLING(                           \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  GENERATE_OPTION_WITH_MARSHALLING(                                            \
      Args, SA, KIND, FLAGS, SPELLING, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE,    \
      IMPLIED_CHECK, IMPLIED_VALUE, DENORMALIZER, EXTRACTOR, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef PREPROCESSOR_OUTPUT_OPTION_WITH_MARSHALLING

  bool Generate_dM = isStrictlyPreprocessorAction(Action) && !Opts.ShowCPP;
  if (Generate_dM)
    GenerateArg(Args, OPT_dM, SA);
  if (!Generate_dM && Opts.ShowMacros)
    GenerateArg(Args, OPT_dD, SA);
  if (Opts.DirectivesOnly)
    GenerateArg(Args, OPT_fdirectives_only, SA);
}

static bool ParsePreprocessorOutputArgs(PreprocessorOutputOptions &Opts,
                                        ArgList &Args, DiagnosticsEngine &Diags,
                                        frontend::ActionKind Action) {
  unsigned NumErrorsBefore = Diags.getNumErrors();

  PreprocessorOutputOptions &PreprocessorOutputOpts = Opts;

#define PREPROCESSOR_OUTPUT_OPTION_WITH_MARSHALLING(                           \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(                                               \
      Args, Diags, ID, FLAGS, PARAM, SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,     \
      IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef PREPROCESSOR_OUTPUT_OPTION_WITH_MARSHALLING

  Opts.ShowCPP = isStrictlyPreprocessorAction(Action) && !Args.hasArg(OPT_dM);
  Opts.ShowMacros = Args.hasArg(OPT_dM) || Args.hasArg(OPT_dD);
  Opts.DirectivesOnly = Args.hasArg(OPT_fdirectives_only);

  return Diags.getNumErrors() == NumErrorsBefore;
}

static void GenerateTargetArgs(const TargetOptions &Opts,
                               SmallVectorImpl<const char *> &Args,
                               CompilerInvocation::StringAllocator SA) {
  const TargetOptions *TargetOpts = &Opts;
#define TARGET_OPTION_WITH_MARSHALLING(                                        \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  GENERATE_OPTION_WITH_MARSHALLING(                                            \
      Args, SA, KIND, FLAGS, SPELLING, ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE,    \
      IMPLIED_CHECK, IMPLIED_VALUE, DENORMALIZER, EXTRACTOR, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef TARGET_OPTION_WITH_MARSHALLING

  if (!Opts.SDKVersion.empty())
    GenerateArg(Args, OPT_target_sdk_version_EQ, Opts.SDKVersion.getAsString(),
                SA);
  if (!Opts.DarwinTargetVariantSDKVersion.empty())
    GenerateArg(Args, OPT_darwin_target_variant_sdk_version_EQ,
                Opts.DarwinTargetVariantSDKVersion.getAsString(), SA);
}

static bool ParseTargetArgs(TargetOptions &Opts, ArgList &Args,
                            DiagnosticsEngine &Diags) {
  unsigned NumErrorsBefore = Diags.getNumErrors();

  TargetOptions *TargetOpts = &Opts;

#define TARGET_OPTION_WITH_MARSHALLING(                                        \
    PREFIX_TYPE, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,        \
    HELPTEXT, METAVAR, VALUES, SPELLING, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH,   \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    MERGER, EXTRACTOR, TABLE_INDEX)                                            \
  PARSE_OPTION_WITH_MARSHALLING(                                               \
      Args, Diags, ID, FLAGS, PARAM, SHOULD_PARSE, KEYPATH, DEFAULT_VALUE,     \
      IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, MERGER, TABLE_INDEX)
#include "clang/Driver/Options.inc"
#undef TARGET_OPTION_WITH_MARSHALLING

  if (Arg *A = Args.getLastArg(options::OPT_target_sdk_version_EQ)) {
    llvm::VersionTuple Version;
    if (Version.tryParse(A->getValue()))
      Diags.Report(diag::err_drv_invalid_value)
          << A->getAsString(Args) << A->getValue();
    else
      Opts.SDKVersion = Version;
  }
  if (Arg *A =
          Args.getLastArg(options::OPT_darwin_target_variant_sdk_version_EQ)) {
    llvm::VersionTuple Version;
    if (Version.tryParse(A->getValue()))
      Diags.Report(diag::err_drv_invalid_value)
          << A->getAsString(Args) << A->getValue();
    else
      Opts.DarwinTargetVariantSDKVersion = Version;
  }

  return Diags.getNumErrors() == NumErrorsBefore;
}

bool CompilerInvocation::CreateFromArgsImpl(
    CompilerInvocation &Res, ArrayRef<const char *> CommandLineArgs,
    DiagnosticsEngine &Diags, const char *Argv0) {
  unsigned NumErrorsBefore = Diags.getNumErrors();

  // Parse the arguments.
  const OptTable &Opts = getDriverOptTable();
  const unsigned IncludedFlagsBitmask = options::CC1Option;
  unsigned MissingArgIndex, MissingArgCount;
  InputArgList Args = Opts.ParseArgs(CommandLineArgs, MissingArgIndex,
                                     MissingArgCount, IncludedFlagsBitmask);
  LangOptions &LangOpts = *Res.getLangOpts();

  // Check for missing argument error.
  if (MissingArgCount)
    Diags.Report(diag::err_drv_missing_argument)
        << Args.getArgString(MissingArgIndex) << MissingArgCount;

  // Issue errors on unknown arguments.
  for (const auto *A : Args.filtered(OPT_UNKNOWN)) {
    auto ArgString = A->getAsString(Args);
    std::string Nearest;
    if (Opts.findNearest(ArgString, Nearest, IncludedFlagsBitmask) > 1)
      Diags.Report(diag::err_drv_unknown_argument) << ArgString;
    else
      Diags.Report(diag::err_drv_unknown_argument_with_suggestion)
          << ArgString << Nearest;
  }

  ParseFileSystemArgs(Res.getFileSystemOpts(), Args, Diags);
  ParseMigratorArgs(Res.getMigratorOpts(), Args, Diags);
  ParseAnalyzerArgs(*Res.getAnalyzerOpts(), Args, Diags);
  ParseDiagnosticArgs(Res.getDiagnosticOpts(), Args, &Diags,
                      /*DefaultDiagColor=*/false);
  ParseFrontendArgs(Res.getFrontendOpts(), Args, Diags, LangOpts.IsHeaderFile);
  // FIXME: We shouldn't have to pass the DashX option around here
  InputKind DashX = Res.getFrontendOpts().DashX;
  ParseTargetArgs(Res.getTargetOpts(), Args, Diags);
  llvm::Triple T(Res.getTargetOpts().Triple);
  ParseHeaderSearchArgs(Res.getHeaderSearchOpts(), Args, Diags,
                        Res.getFileSystemOpts().WorkingDir);

  ParseLangArgs(LangOpts, Args, DashX, T, Res.getPreprocessorOpts().Includes,
                Diags);
  if (Res.getFrontendOpts().ProgramAction == frontend::RewriteObjC)
    LangOpts.ObjCExceptions = 1;

  for (auto Warning : Res.getDiagnosticOpts().Warnings) {
    if (Warning == "misexpect" &&
        !Diags.isIgnored(diag::warn_profile_data_misexpect, SourceLocation())) {
      Res.getCodeGenOpts().MisExpect = true;
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

  ParseCodeGenArgs(Res.getCodeGenOpts(), Args, DashX, Diags, T,
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
                        Res.getFrontendOpts().ProgramAction,
                        Res.getFrontendOpts());
  ParsePreprocessorOutputArgs(Res.getPreprocessorOutputOpts(), Args, Diags,
                              Res.getFrontendOpts().ProgramAction);

  ParseDependencyOutputArgs(Res.getDependencyOutputOpts(), Args, Diags,
                            Res.getFrontendOpts().ProgramAction,
                            Res.getPreprocessorOutputOpts().ShowLineMarkers);
  if (!Res.getDependencyOutputOpts().OutputFile.empty() &&
      Res.getDependencyOutputOpts().Targets.empty())
    Diags.Report(diag::err_fe_dependency_file_requires_MT);

  // If sanitizer is enabled, disable OPT_ffine_grained_bitfield_accesses.
  if (Res.getCodeGenOpts().FineGrainedBitfieldAccesses &&
      !Res.getLangOpts()->Sanitize.empty()) {
    Res.getCodeGenOpts().FineGrainedBitfieldAccesses = false;
    Diags.Report(diag::warn_drv_fine_grained_bitfield_accesses_ignored);
  }

  // Store the command-line for using in the CodeView backend.
  Res.getCodeGenOpts().Argv0 = Argv0;
  append_range(Res.getCodeGenOpts().CommandLineArgs, CommandLineArgs);

  FixupInvocation(Res, Diags, Args, DashX);

  return Diags.getNumErrors() == NumErrorsBefore;
}

bool CompilerInvocation::CreateFromArgs(CompilerInvocation &Invocation,
                                        ArrayRef<const char *> CommandLineArgs,
                                        DiagnosticsEngine &Diags,
                                        const char *Argv0) {
  CompilerInvocation DummyInvocation;

  return RoundTrip(
      [](CompilerInvocation &Invocation, ArrayRef<const char *> CommandLineArgs,
         DiagnosticsEngine &Diags, const char *Argv0) {
        return CreateFromArgsImpl(Invocation, CommandLineArgs, Diags, Argv0);
      },
      [](CompilerInvocation &Invocation, SmallVectorImpl<const char *> &Args,
         StringAllocator SA) { Invocation.generateCC1CommandLine(Args, SA); },
      Invocation, DummyInvocation, CommandLineArgs, Diags, Argv0);
}

std::string CompilerInvocation::getModuleHash() const {
  // FIXME: Consider using SHA1 instead of MD5.
  llvm::HashBuilder<llvm::MD5, llvm::support::endianness::native> HBuilder;

  // Note: For QoI reasons, the things we use as a hash here should all be
  // dumped via the -module-info flag.

  // Start the signature with the compiler version.
  HBuilder.add(getClangFullRepositoryVersion());

  // Also include the serialization version, in case LLVM_APPEND_VC_REV is off
  // and getClangFullRepositoryVersion() doesn't include git revision.
  HBuilder.add(serialization::VERSION_MAJOR, serialization::VERSION_MINOR);

  // Extend the signature with the language options
#define LANGOPT(Name, Bits, Default, Description) HBuilder.add(LangOpts->Name);
#define ENUM_LANGOPT(Name, Type, Bits, Default, Description)                   \
  HBuilder.add(static_cast<unsigned>(LangOpts->get##Name()));
#define BENIGN_LANGOPT(Name, Bits, Default, Description)
#define BENIGN_ENUM_LANGOPT(Name, Type, Bits, Default, Description)
#include "clang/Basic/LangOptions.def"

  HBuilder.addRange(LangOpts->ModuleFeatures);

  HBuilder.add(LangOpts->ObjCRuntime);
  HBuilder.addRange(LangOpts->CommentOpts.BlockCommandNames);

  // Extend the signature with the target options.
  HBuilder.add(TargetOpts->Triple, TargetOpts->CPU, TargetOpts->TuneCPU,
               TargetOpts->ABI);
  HBuilder.addRange(TargetOpts->FeaturesAsWritten);

  // Extend the signature with preprocessor options.
  const PreprocessorOptions &ppOpts = getPreprocessorOpts();
  HBuilder.add(ppOpts.UsePredefines, ppOpts.DetailedRecord);

  const HeaderSearchOptions &hsOpts = getHeaderSearchOpts();
  for (const auto &Macro : getPreprocessorOpts().Macros) {
    // If we're supposed to ignore this macro for the purposes of modules,
    // don't put it into the hash.
    if (!hsOpts.ModulesIgnoreMacros.empty()) {
      // Check whether we're ignoring this macro.
      StringRef MacroDef = Macro.first;
      if (hsOpts.ModulesIgnoreMacros.count(
              llvm::CachedHashString(MacroDef.split('=').first)))
        continue;
    }

    HBuilder.add(Macro);
  }

  // Extend the signature with the sysroot and other header search options.
  HBuilder.add(hsOpts.Sysroot, hsOpts.ModuleFormat, hsOpts.UseDebugInfo,
               hsOpts.UseBuiltinIncludes, hsOpts.UseStandardSystemIncludes,
               hsOpts.UseStandardCXXIncludes, hsOpts.UseLibcxx,
               hsOpts.ModulesValidateDiagnosticOptions);
  HBuilder.add(hsOpts.ResourceDir);

  if (hsOpts.ModulesStrictContextHash) {
    HBuilder.addRange(hsOpts.SystemHeaderPrefixes);
    HBuilder.addRange(hsOpts.UserEntries);

    const DiagnosticOptions &diagOpts = getDiagnosticOpts();
#define DIAGOPT(Name, Bits, Default) HBuilder.add(diagOpts.Name);
#define ENUM_DIAGOPT(Name, Type, Bits, Default)                                \
  HBuilder.add(diagOpts.get##Name());
#include "clang/Basic/DiagnosticOptions.def"
#undef DIAGOPT
#undef ENUM_DIAGOPT
  }

  // Extend the signature with the user build path.
  HBuilder.add(hsOpts.ModuleUserBuildPath);

  // Extend the signature with the module file extensions.
  for (const auto &ext : getFrontendOpts().ModuleFileExtensions)
    ext->hashExtension(HBuilder);

  // When compiling with -gmodules, also hash -fdebug-prefix-map as it
  // affects the debug info in the PCM.
  if (getCodeGenOpts().DebugTypeExtRefs)
    HBuilder.addRange(getCodeGenOpts().DebugPrefixMap);

  // Extend the signature with the enabled sanitizers, if at least one is
  // enabled. Sanitizers which cannot affect AST generation aren't hashed.
  SanitizerSet SanHash = LangOpts->Sanitize;
  SanHash.clear(getPPTransparentSanitizers());
  if (!SanHash.empty())
    HBuilder.add(SanHash.Mask);

  llvm::MD5::MD5Result Result;
  HBuilder.getHasher().final(Result);
  uint64_t Hash = Result.high() ^ Result.low();
  return toString(llvm::APInt(64, Hash), 36, /*Signed=*/false);
}

void CompilerInvocation::generateCC1CommandLine(
    SmallVectorImpl<const char *> &Args, StringAllocator SA) const {
  llvm::Triple T(TargetOpts->Triple);

  GenerateFileSystemArgs(FileSystemOpts, Args, SA);
  GenerateMigratorArgs(MigratorOpts, Args, SA);
  GenerateAnalyzerArgs(*AnalyzerOpts, Args, SA);
  GenerateDiagnosticArgs(*DiagnosticOpts, Args, SA, false);
  GenerateFrontendArgs(FrontendOpts, Args, SA, LangOpts->IsHeaderFile);
  GenerateTargetArgs(*TargetOpts, Args, SA);
  GenerateHeaderSearchArgs(*HeaderSearchOpts, Args, SA);
  GenerateLangArgs(*LangOpts, Args, SA, T, FrontendOpts.DashX);
  GenerateCodeGenArgs(CodeGenOpts, Args, SA, T, FrontendOpts.OutputFile,
                      &*LangOpts);
  GeneratePreprocessorArgs(*PreprocessorOpts, Args, SA, *LangOpts, FrontendOpts,
                           CodeGenOpts);
  GeneratePreprocessorOutputArgs(PreprocessorOutputOpts, Args, SA,
                                 FrontendOpts.ProgramAction);
  GenerateDependencyOutputArgs(DependencyOutputOpts, Args, SA);
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
