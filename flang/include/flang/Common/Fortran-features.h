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
    SignedPrimary, FileName, Carriagecontrol, Convert, Dispose,
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
    RedundantContiguous, InitBlankCommon, EmptyBindCDerivedType,
    MiscSourceExtensions, AllocateToOtherLength, LongNames, IntrinsicAsSpecific,
    BenignNameClash, BenignRedundancy, NullMoldAllocatableComponentValue,
    NopassScalarBase, MiscUseExtensions, ImpliedDoIndexScope,
    DistinctCommonSizes, OddIndexVariableRestrictions,
    IndistinguishableSpecifics)

// Portability and suspicious usage warnings for conforming code
ENUM_CLASS(UsageWarning, Portability, PointerToUndefinable,
    NonTargetPassedToTarget, PointerToPossibleNoncontiguous,
    ShortCharacterActual, ExprPassedToVolatile, ImplicitInterfaceActual,
    PolymorphicTransferArg, PointerComponentTransferArg, TransferSizePresence,
    F202XAllocatableBreakingChange, DimMustBePresent, CommonBlockPadding,
    LogicalVsCBool, BindCCharLength, ProcDummyArgShapes, ExternalNameConflict)

using LanguageFeatures = EnumSet<LanguageFeature, LanguageFeature_enumSize>;
using UsageWarnings = EnumSet<UsageWarning, UsageWarning_enumSize>;

class LanguageFeatureControl {
public:
  LanguageFeatureControl() {
    // These features must be explicitly enabled by command line options.
    disable_.set(LanguageFeature::OldDebugLines);
    disable_.set(LanguageFeature::OpenACC);
    disable_.set(LanguageFeature::OpenMP);
    disable_.set(LanguageFeature::CUDA); // !@cuf
    disable_.set(LanguageFeature::ImplicitNoneTypeNever);
    disable_.set(LanguageFeature::ImplicitNoneTypeAlways);
    disable_.set(LanguageFeature::DefaultSave);
    disable_.set(LanguageFeature::SaveMainProgram);
    // These features, if enabled, conflict with valid standard usage,
    // so there are disabled here by default.
    disable_.set(LanguageFeature::BackslashEscapes);
    disable_.set(LanguageFeature::LogicalAbbreviations);
    disable_.set(LanguageFeature::XOROperator);
    disable_.set(LanguageFeature::OldStyleParameter);
  }
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
  bool IsEnabled(LanguageFeature f) const { return !disable_.test(f); }
  bool ShouldWarn(LanguageFeature f) const {
    return (warnAllLanguage_ && f != LanguageFeature::OpenMP &&
               f != LanguageFeature::OpenACC) ||
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
