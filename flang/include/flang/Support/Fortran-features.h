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
    IgnoreIrrelevantAttributes, Unsigned, AmbiguousStructureConstructor,
    ContiguousOkForSeqAssociation, ForwardRefExplicitTypeDummy,
    InaccessibleDeferredOverride, CudaWarpMatchFunction, DoConcurrentOffload,
    TransferBOZ)

// Portability and suspicious usage warnings
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
    CompatibleDeclarationsFromDistinctModules,
    NullActualForDefaultIntentAllocatable, UseAssociationIntoSameNameSubprogram,
    HostAssociatedIntentOutInSpecExpr, NonVolatilePointerToVolatile,
    RealConstantWidening)

using LanguageFeatures = EnumSet<LanguageFeature, LanguageFeature_enumSize>;
using UsageWarnings = EnumSet<UsageWarning, UsageWarning_enumSize>;
using LanguageFeatureOrWarning = std::variant<LanguageFeature, UsageWarning>;
using LanguageControlFlag =
    std::pair<LanguageFeatureOrWarning, /*shouldEnable=*/bool>;

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
  bool IsWarnOnAllNonstandard() const { return warnAllLanguage_; }
  void WarnOnAllUsage(bool yes = true);
  bool IsWarnOnAllUsage() const { return warnAllUsage_; }
  void DisableAllNonstandardWarnings() {
    warnAllLanguage_ = false;
    warnLanguage_.clear();
  }
  void DisableAllUsageWarnings() {
    warnAllUsage_ = false;
    warnUsage_.clear();
  }
  void DisableAllWarnings() {
    disableAllWarnings_ = true;
    DisableAllNonstandardWarnings();
    DisableAllUsageWarnings();
  }
  bool AreWarningsDisabled() const { return disableAllWarnings_; }
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
  // These two arrays map the enum values to their cannonical Cli spellings.
  // Since each of the CanonicalSpelling is a string in the domain of the map
  // above we just use a view of the string instead of another copy.
  std::array<std::string, LanguageFeature_enumSize>
      languageFeatureCliCanonicalSpelling_;
  std::array<std::string, UsageWarning_enumSize>
      usageWarningCliCanonicalSpelling_;
  LanguageFeatures disable_;
  LanguageFeatures warnLanguage_;
  bool warnAllLanguage_{false};
  UsageWarnings warnUsage_;
  bool warnAllUsage_{false};
  bool disableAllWarnings_{false};
};
} // namespace Fortran::common
#endif // FORTRAN_SUPPORT_FORTRAN_FEATURES_H_
