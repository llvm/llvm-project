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

namespace Fortran::common {

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
    Assign, AssignedGOTO, Pause, OpenACC, OpenMP, CruftAfterAmpersand,
    ClassicCComments, AdditionalFormats, BigIntLiterals, RealDoControls,
    EquivalenceNumericWithCharacter, EquivalenceNonDefaultNumeric,
    EquivalenceSameNonSequence, AdditionalIntrinsics, AnonymousParents,
    OldLabelDoEndStatements, LogicalIntegerAssignment, EmptySourceFile,
    ProgramReturn, ImplicitNoneTypeNever, ImplicitNoneTypeAlways,
    ForwardRefDummyImplicitNone, OpenAccessAppend, BOZAsDefaultInteger,
    DistinguishableSpecifics, DefaultSave, PointerInSeqType, NonCharacterFormat,
    SaveMainProgram, SaveBigMainProgramVariables)

using LanguageFeatures = EnumSet<LanguageFeature, LanguageFeature_enumSize>;

class LanguageFeatureControl {
public:
  LanguageFeatureControl() {
    // These features must be explicitly enabled by command line options.
    disable_.set(LanguageFeature::OldDebugLines);
    disable_.set(LanguageFeature::OpenACC);
    disable_.set(LanguageFeature::OpenMP);
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
  void EnableWarning(LanguageFeature f, bool yes = true) { warn_.set(f, yes); }
  void WarnOnAllNonstandard(bool yes = true) { warnAll_ = yes; }
  bool IsEnabled(LanguageFeature f) const { return !disable_.test(f); }
  bool ShouldWarn(LanguageFeature f) const {
    return (warnAll_ && f != LanguageFeature::OpenMP &&
               f != LanguageFeature::OpenACC) ||
        warn_.test(f);
  }
  // Return all spellings of operators names, depending on features enabled
  std::vector<const char *> GetNames(LogicalOperator) const;
  std::vector<const char *> GetNames(RelationalOperator) const;

private:
  LanguageFeatures disable_;
  LanguageFeatures warn_;
  bool warnAll_{false};
};
} // namespace Fortran::common
#endif // FORTRAN_COMMON_FORTRAN_FEATURES_H_
