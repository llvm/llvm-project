//===-- include/flang/Support/Fortran-features.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SUPPORT_FORTRAN_FEATURES_H_
#define FORTRAN_SUPPORT_FORTRAN_FEATURES_H_

#include "Fortran.h"
#include "flang/Common/enum-set.h"
#include "clang/Basic/DiagnosticOptions.h"
#include <string_view>
#include <vector>

namespace Fortran::common {

// Non-conforming extensions & legacies
ENUM_CLASS(LanguageFeature, BackslashEscapes, OldDebugLines,
    FixedFormContinuationWithColumn1Ampersand, LogicalAbbreviations,
    XOROperator, PunctuationInNames, OptionalFreeFormSpace, BOZExtensions,
    EmptyStatement, AlternativeNE, ExecutionPartNamelist, DECStructures,
    DoubleComplex, Byte, StarKind, ExponentMatchingKindParam, QuadPrecision,
    SlashInitialization, TripletInArrayConstructor, MissingColons,
    SignedComplexLiteral, OldStyleParameter, ComplexConstructor, PercentLOC,
    SignedMultOperand, FileName, Carriagecontrol, Convert, Dispose,
    IOListLeadingComma, AbbreviatedEditDescriptor, ProgramParentheses,
    PercentRefAndVal, OmitFunctionDummies, CrayPointer, Hollerith, ArithmeticIF,
    Assign, AssignedGOTO, Pause, OpenACC, OpenMP, CUDA, CruftAfterAmpersand,
    ClassicCComments, AdditionalFormats, BigIntLiterals, RealDoControls,
    EquivalenceNumericWithCharacter, EquivalenceNonDefaultNumeric,
    EquivalenceSameNonSequence, AdditionalIntrinsics, AnonymousParents,
    OldLabelDoEndStatements, LogicalIntegerAssignment, EmptySourceFile,
    ProgramReturn, ImplicitNoneTypeNever, ImplicitNoneTypeAlways,
    ImplicitNoneExternal, ForwardRefImplicitNone, OpenAccessAppend,
    BOZAsDefaultInteger, DistinguishableSpecifics, DefaultSave,
    PointerInSeqType, NonCharacterFormat, SaveMainProgram,
    SaveBigMainProgramVariables, DistinctArrayConstructorLengths, PPCVector,
    RelaxedIntentInChecking, ForwardRefImplicitNoneData,
    NullActualForAllocatable, ActualIntegerConvertedToSmallerKind,
    HollerithOrCharacterAsBOZ, BindingAsProcedure, StatementFunctionExtensions,
    UseGenericIntrinsicWhenSpecificDoesntMatch, DataStmtExtensions,
    RedundantContiguous, RedundantAttribute, InitBlankCommon,
    EmptyBindCDerivedType, MiscSourceExtensions, AllocateToOtherLength,
    LongNames, IntrinsicAsSpecific, BenignNameClash, BenignRedundancy,
    NullMoldAllocatableComponentValue, NopassScalarBase, MiscUseExtensions,
    ImpliedDoIndexScope, DistinctCommonSizes, OddIndexVariableRestrictions,
    IndistinguishableSpecifics, SubroutineAndFunctionSpecifics,
    EmptySequenceType, NonSequenceCrayPointee, BranchIntoConstruct,
    BadBranchTarget, HollerithPolymorphic, ListDirectedSize,
    NonBindCInteroperability, CudaManaged, CudaUnified,
    PolymorphicActualAllocatableOrPointerToMonomorphicDummy, RelaxedPureDummy,
    UndefinableAsynchronousOrVolatileActual, AutomaticInMainProgram, PrintCptr,
    SavedLocalInSpecExpr, PrintNamelist, AssumedRankPassedToNonAssumedRank,
    IgnoreIrrelevantAttributes, Unsigned, ContiguousOkForSeqAssociation,
    ForwardRefExplicitTypeDummy, InaccessibleDeferredOverride,
    CudaWarpMatchFunction, DoConcurrentOffload, TransferBOZ, Coarray,
    PointerPassObject, MultipleIdenticalDATA,
    DefaultStructConstructorNullPointer, AssumedRankIoItem,
    MultipleProgramUnitsOnSameLine, AllocatedForAssociated,
    OpenMPThreadprivateEquivalence, RelaxedCLoc)

// Portability and suspicious usage warnings
// When adding a new UsageWarning, add it to commonWarnings if it should be
//   enabled by -pedantic. Or if it is specific to a Fortran standard, add it
//   to the applicable f**Warnings.
ENUM_CLASS(UsageWarning, Portability, PointerToUndefinable,
    NonTargetPassedToTarget, PointerToPossibleNoncontiguous,
    ShortCharacterActual, ShortArrayActual, ImplicitInterfaceActual,
    PolymorphicTransferArg, PointerComponentTransferArg, TransferSizePresence,
    F202XAllocatableBreakingChange, OptionalMustBePresent, CommonBlockPadding,
    LogicalVsCBool, BindCCharLength, ProcDummyArgShapes, ExternalNameConflict,
    FoldingException, FoldingAvoidsRuntimeCrash, FoldingValueChecks,
    FoldingFailure, FoldingLimit, Interoperability, CharacterInteroperability,
    Bounds, Preprocessing, Scanning, OpenAccUsage, ProcPointerCompatibility,
    VoidMold, KnownBadImplicitInterface, EmptyCase, CaseOverflow, CUDAUsage,
    IgnoreTKRUsage, ExternalInterfaceMismatch, DefinedOperatorArgs, Final,
    ZeroDoStep, UnusedForallIndex, OpenMPUsage, DataLength, IgnoredDirective,
    HomonymousSpecific, HomonymousResult, IgnoredIntrinsicFunctionType,
    PreviousScalarUse, RedeclaredInaccessibleComponent, ImplicitShared,
    IndexVarRedefinition, IncompatibleImplicitInterfaces,
    VectorSubscriptFinalization, UndefinedFunctionResult, UselessIomsg,
    MismatchingDummyProcedure, SubscriptedEmptyArray, UnsignedLiteralTruncation,
    CompatibleDeclarationsFromDistinctModules, ConstantIsContiguous,
    NullActualForDefaultIntentAllocatable, UseAssociationIntoSameNameSubprogram,
    HostAssociatedIntentOutInSpecExpr, NonVolatilePointerToVolatile,
    RealConstantWidening, VolatileOrAsynchronousTemporary, UnusedVariable,
    UsedUndefinedVariable, BadValueInDeadCode, AssumedTypeSizeDummy,
    MisplacedIgnoreTKR, NamelistParameter, ImpureFinalInPure,
    IgnoredNoReallocateLHS, CLoc, SystemClockNotIntrinsic,
    SystemClockArgsOnlyInteger, SystemClockIntArgsOnlyDefault,
    SystemClockIntArgsSameKind, SystemClockMinSize)

using LanguageFeatures = EnumSet<LanguageFeature, LanguageFeature_enumSize>;
using UsageWarnings = EnumSet<UsageWarning, UsageWarning_enumSize>;
using LanguageFeatureOrWarning = std::variant<LanguageFeature, UsageWarning>;
using LanguageControlFlag =
    std::pair<LanguageFeatureOrWarning, /*shouldEnable=*/bool>;

// Usage Warnings that are not limited to specific Fortran standards
static inline UsageWarnings commonWarnings = {UsageWarning::Portability,
    UsageWarning::PointerToUndefinable, UsageWarning::NonTargetPassedToTarget,
    UsageWarning::PointerToPossibleNoncontiguous,
    UsageWarning::ShortCharacterActual, UsageWarning::ShortArrayActual,
    UsageWarning::ImplicitInterfaceActual, UsageWarning::PolymorphicTransferArg,
    UsageWarning::PointerComponentTransferArg,
    UsageWarning::TransferSizePresence,
    UsageWarning::F202XAllocatableBreakingChange,
    UsageWarning::OptionalMustBePresent, UsageWarning::CommonBlockPadding,
    UsageWarning::LogicalVsCBool, UsageWarning::BindCCharLength,
    UsageWarning::ProcDummyArgShapes, UsageWarning::ExternalNameConflict,
    UsageWarning::FoldingException, UsageWarning::FoldingAvoidsRuntimeCrash,
    UsageWarning::FoldingValueChecks, UsageWarning::FoldingFailure,
    UsageWarning::FoldingLimit, UsageWarning::Interoperability,
    UsageWarning::CharacterInteroperability, UsageWarning::Bounds,
    UsageWarning::Preprocessing, UsageWarning::Scanning,
    UsageWarning::OpenAccUsage, UsageWarning::ProcPointerCompatibility,
    UsageWarning::VoidMold, UsageWarning::KnownBadImplicitInterface,
    UsageWarning::EmptyCase, UsageWarning::CaseOverflow,
    UsageWarning::CUDAUsage, UsageWarning::IgnoreTKRUsage,
    UsageWarning::ExternalInterfaceMismatch, UsageWarning::DefinedOperatorArgs,
    UsageWarning::Final, UsageWarning::ZeroDoStep,
    UsageWarning::UnusedForallIndex, UsageWarning::OpenMPUsage,
    UsageWarning::DataLength, UsageWarning::IgnoredDirective,
    UsageWarning::HomonymousSpecific, UsageWarning::HomonymousResult,
    UsageWarning::IgnoredIntrinsicFunctionType, UsageWarning::PreviousScalarUse,
    UsageWarning::RedeclaredInaccessibleComponent, UsageWarning::ImplicitShared,
    UsageWarning::IndexVarRedefinition,
    UsageWarning::IncompatibleImplicitInterfaces,
    UsageWarning::VectorSubscriptFinalization,
    UsageWarning::UndefinedFunctionResult, UsageWarning::UselessIomsg,
    UsageWarning::MismatchingDummyProcedure,
    UsageWarning::SubscriptedEmptyArray,
    UsageWarning::UnsignedLiteralTruncation,
    UsageWarning::CompatibleDeclarationsFromDistinctModules,
    UsageWarning::ConstantIsContiguous,
    UsageWarning::NullActualForDefaultIntentAllocatable,
    UsageWarning::UseAssociationIntoSameNameSubprogram,
    UsageWarning::HostAssociatedIntentOutInSpecExpr,
    UsageWarning::NonVolatilePointerToVolatile,
    UsageWarning::RealConstantWidening,
    UsageWarning::VolatileOrAsynchronousTemporary, UsageWarning::UnusedVariable,
    UsageWarning::UsedUndefinedVariable, UsageWarning::BadValueInDeadCode,
    UsageWarning::AssumedTypeSizeDummy, UsageWarning::MisplacedIgnoreTKR,
    UsageWarning::NamelistParameter, UsageWarning::ImpureFinalInPure,
    UsageWarning::IgnoredNoReallocateLHS, UsageWarning::CLoc};

// Usage Warnings for f77
static inline UsageWarnings f77Warnings = {
    UsageWarning::SystemClockNotIntrinsic};

// Usage Warnings for f90
static inline UsageWarnings f90Warnings = {
    UsageWarning::SystemClockArgsOnlyInteger,
    UsageWarning::SystemClockIntArgsOnlyDefault};

// Usage Warnings for f95
static inline UsageWarnings f95Warnings = {
    UsageWarning::SystemClockArgsOnlyInteger,
    UsageWarning::SystemClockIntArgsOnlyDefault};

// Usage Warnings for f2003
static inline UsageWarnings f2003Warnings = {};

// Usage Warnings for f2008
static inline UsageWarnings f2008Warnings = {};

// Usage Warnings for f2018
static inline UsageWarnings f2018Warnings = {};

// Usage Warnings for f2023
static inline UsageWarnings f2023Warnings = {
    UsageWarning::SystemClockIntArgsSameKind, UsageWarning::SystemClockMinSize};

// Usage Warnings for f202Y
static inline UsageWarnings f202YWarnings = {
    UsageWarning::SystemClockIntArgsSameKind, UsageWarning::SystemClockMinSize};

class LanguageFeatureControl {
public:
  LanguageFeatureControl();
  LanguageFeatureControl(const LanguageFeatureControl &) = default;

  void Enable(LanguageFeature f, bool yes = true) { disable_.set(f, !yes); }
  void EnableWarning(LanguageFeature f, bool yes = true) {
    warnLanguage_.set(f, yes);
  }
  void EnableWarning(UsageWarning w, bool yes = true) {
    warnUsage_.set(w, yes);
  }
  void EnableWarning(LanguageFeatureOrWarning flag, bool yes = true) {
    if (std::holds_alternative<LanguageFeature>(flag)) {
      EnableWarning(std::get<LanguageFeature>(flag), yes);
    } else {
      EnableWarning(std::get<UsageWarning>(flag), yes);
    }
  }
  void WarnOnAllNonstandard(bool yes = true);
  void WarnOnAllUsage(clang::FlangPedanticVersionTy version =
                          clang::FlangPedanticVersionTy::f2018);
  void DisableAllNonstandardWarnings() { warnLanguage_.clear(); }
  void DisableAllUsageWarnings() { warnUsage_.clear(); }
  void DisableAllWarnings() {
    DisableAllNonstandardWarnings();
    DisableAllUsageWarnings();
  }
  bool IsEnabled(LanguageFeature f) const { return !disable_.test(f); }
  bool ShouldWarn(LanguageFeature f) const { return warnLanguage_.test(f); }
  bool ShouldWarn(UsageWarning w) const { return warnUsage_.test(w); }
  // Cli options
  // Find a warning by its Cli spelling, i.e. '[no-]warning-name'.
  std::optional<LanguageControlFlag> FindWarning(std::string_view input);
  // Take a string from the Cli and apply it to the LanguageFeatureControl.
  // Return true if the option was recognized (and hence applied).
  bool EnableWarning(std::string_view input);
  // The add and replace functions are not currently used but are provided
  // to allow a flexible many-to-one mapping from Cli spellings to enum values.
  // Taking a string by value because the functions own this string after the
  // call.
  void AddAlternativeCliSpelling(LanguageFeature f, std::string input) {
    cliOptions_.insert({input, {f}});
  }
  void AddAlternativeCliSpelling(UsageWarning w, std::string input) {
    cliOptions_.insert({input, {w}});
  }
  void AddDeprecatedCliSpelling(LanguageFeature f,
      const std::string &deprecated, const std::string &canonical) {
    cliOptions_.insert({deprecated, {f}});
    deprecatedCliOptions_.insert({deprecated, canonical});
  }
  void AddDeprecatedCliSpelling(UsageWarning w, const std::string &deprecated,
      const std::string &canonical) {
    cliOptions_.insert({deprecated, {w}});
    deprecatedCliOptions_.insert({deprecated, canonical});
  }
  // Returns the canonical spelling if the input is a deprecated spelling.
  std::optional<std::string_view> CheckDeprecatedSpelling(
      std::string_view input) const;
  void ReplaceCliCanonicalSpelling(LanguageFeature f, std::string input);
  void ReplaceCliCanonicalSpelling(UsageWarning w, std::string input);
  std::string_view getDefaultCliSpelling(LanguageFeature f) const {
    return languageFeatureCliCanonicalSpelling_[EnumToInt(f)];
  };
  std::string_view getDefaultCliSpelling(UsageWarning w) const {
    return usageWarningCliCanonicalSpelling_[EnumToInt(w)];
  };
  // Return all spellings of operators names, depending on features enabled
  std::vector<const char *> GetNames(LogicalOperator) const;
  std::vector<const char *> GetNames(RelationalOperator) const;

private:
  // Map from Cli syntax of language features and usage warnings to their enum
  // values.
  std::unordered_map<std::string, LanguageFeatureOrWarning> cliOptions_;
  // Map from deprecated Cli spellings to their canonical replacements.
  std::unordered_map<std::string, std::string> deprecatedCliOptions_;
  // These two arrays map the enum values to their cannonical Cli spellings.
  // Since each of the CanonicalSpelling is a string in the domain of the map
  // above we just use a view of the string instead of another copy.
  std::array<std::string, LanguageFeature_enumSize>
      languageFeatureCliCanonicalSpelling_;
  std::array<std::string, UsageWarning_enumSize>
      usageWarningCliCanonicalSpelling_;
  LanguageFeatures disable_;
  LanguageFeatures warnLanguage_;
  UsageWarnings warnUsage_;
};
} // namespace Fortran::common
#endif // FORTRAN_SUPPORT_FORTRAN_FEATURES_H_
