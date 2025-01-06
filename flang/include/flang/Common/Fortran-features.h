//===-- include/flang/Common/Fortran-features.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_FORTRAN_FEATURES_H_
#define FORTRAN_COMMON_FORTRAN_FEATURES_H_

#include "flang/Common/Fortran.h"
#include "flang/Common/enum-set.h"
#include "flang/Common/idioms.h"
#include <optional>
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
    ForwardRefImplicitNone, OpenAccessAppend, BOZAsDefaultInteger,
    DistinguishableSpecifics, DefaultSave, PointerInSeqType, NonCharacterFormat,
    SaveMainProgram, SaveBigMainProgramVariables,
    DistinctArrayConstructorLengths, PPCVector, RelaxedIntentInChecking,
    ForwardRefImplicitNoneData, NullActualForAllocatable,
    ActualIntegerConvertedToSmallerKind, HollerithOrCharacterAsBOZ,
    BindingAsProcedure, StatementFunctionExtensions,
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
    IgnoreIrrelevantAttributes, Unsigned)

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
    IndexVarRedefinition, IncompatibleImplicitInterfaces, BadTypeForTarget,
    VectorSubscriptFinalization, UndefinedFunctionResult, UselessIomsg,
    MismatchingDummyProcedure, SubscriptedEmptyArray, UnsignedLiteralTruncation)

using LanguageFeatures = EnumSet<LanguageFeature, LanguageFeature_enumSize>;
using UsageWarnings = EnumSet<UsageWarning, UsageWarning_enumSize>;

std::optional<LanguageFeature> FindLanguageFeature(const char *);
std::optional<UsageWarning> FindUsageWarning(const char *);

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
  void WarnOnAllNonstandard(bool yes = true) { warnAllLanguage_ = yes; }
  void WarnOnAllUsage(bool yes = true) { warnAllUsage_ = yes; }
  void DisableAllNonstandardWarnings() {
    warnAllLanguage_ = false;
    warnLanguage_.clear();
  }
  void DisableAllUsageWarnings() {
    warnAllUsage_ = false;
    warnUsage_.clear();
  }

  bool IsEnabled(LanguageFeature f) const { return !disable_.test(f); }
  bool ShouldWarn(LanguageFeature f) const {
    return (warnAllLanguage_ && f != LanguageFeature::OpenMP &&
               f != LanguageFeature::OpenACC && f != LanguageFeature::CUDA) ||
        warnLanguage_.test(f);
  }
  bool ShouldWarn(UsageWarning w) const {
    return warnAllUsage_ || warnUsage_.test(w);
  }
  // Return all spellings of operators names, depending on features enabled
  std::vector<const char *> GetNames(LogicalOperator) const;
  std::vector<const char *> GetNames(RelationalOperator) const;

private:
  LanguageFeatures disable_;
  LanguageFeatures warnLanguage_;
  bool warnAllLanguage_{false};
  UsageWarnings warnUsage_;
  bool warnAllUsage_{false};
};
} // namespace Fortran::common
#endif // FORTRAN_COMMON_FORTRAN_FEATURES_H_
