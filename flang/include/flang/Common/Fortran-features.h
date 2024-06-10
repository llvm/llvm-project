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
    BadBranchTarget, ConvertedArgument, HollerithPolymorphic, ListDirectedSize,
    NonBindCInteroperability, CudaManaged, CudaUnified,
    PolymorphicActualAllocatableOrPointerToMonomorphicDummy, RelaxedPureDummy,
    UndefinableAsynchronousOrVolatileActual)

// Portability and suspicious usage warnings
ENUM_CLASS(UsageWarning, Portability, PointerToUndefinable,
    NonTargetPassedToTarget, PointerToPossibleNoncontiguous,
    ShortCharacterActual, ShortArrayActual, ImplicitInterfaceActual,
    PolymorphicTransferArg, PointerComponentTransferArg, TransferSizePresence,
    F202XAllocatableBreakingChange, OptionalMustBePresent, CommonBlockPadding,
    LogicalVsCBool, BindCCharLength, ProcDummyArgShapes, ExternalNameConflict,
    FoldingException, FoldingAvoidsRuntimeCrash, FoldingValueChecks,
    FoldingFailure, FoldingLimit, Interoperability, Bounds, Preprocessing,
    Scanning, OpenAccUsage, ProcPointerCompatibility, VoidMold,
    KnownBadImplicitInterface, EmptyCase, CaseOverflow, CUDAUsage,
    IgnoreTKRUsage, ExternalInterfaceMismatch, DefinedOperatorArgs, Final,
    ZeroDoStep, UnusedForallIndex, OpenMPUsage, ModuleFile, DataLength,
    IgnoredDirective, HomonymousSpecific, HomonymousResult,
    IgnoredIntrinsicFunctionType, PreviousScalarUse,
    RedeclaredInaccessibleComponent, ImplicitShared, IndexVarRedefinition,
    IncompatibleImplicitInterfaces, BadTypeForTarget)

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
    disable_.set(LanguageFeature::CudaManaged);
    disable_.set(LanguageFeature::CudaUnified);
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
    // These warnings are enabled by default, but only because they used
    // to be unconditional.  TODO: prune this list
    warnLanguage_.set(LanguageFeature::ExponentMatchingKindParam);
    warnLanguage_.set(LanguageFeature::RedundantAttribute);
    warnLanguage_.set(LanguageFeature::SubroutineAndFunctionSpecifics);
    warnLanguage_.set(LanguageFeature::EmptySequenceType);
    warnLanguage_.set(LanguageFeature::NonSequenceCrayPointee);
    warnLanguage_.set(LanguageFeature::BranchIntoConstruct);
    warnLanguage_.set(LanguageFeature::BadBranchTarget);
    warnLanguage_.set(LanguageFeature::ConvertedArgument);
    warnLanguage_.set(LanguageFeature::HollerithPolymorphic);
    warnLanguage_.set(LanguageFeature::ListDirectedSize);
    warnUsage_.set(UsageWarning::ShortArrayActual);
    warnUsage_.set(UsageWarning::FoldingException);
    warnUsage_.set(UsageWarning::FoldingAvoidsRuntimeCrash);
    warnUsage_.set(UsageWarning::FoldingValueChecks);
    warnUsage_.set(UsageWarning::FoldingFailure);
    warnUsage_.set(UsageWarning::FoldingLimit);
    warnUsage_.set(UsageWarning::Interoperability);
    warnUsage_.set(UsageWarning::Bounds);
    warnUsage_.set(UsageWarning::Preprocessing);
    warnUsage_.set(UsageWarning::Scanning);
    warnUsage_.set(UsageWarning::OpenAccUsage);
    warnUsage_.set(UsageWarning::ProcPointerCompatibility);
    warnUsage_.set(UsageWarning::VoidMold);
    warnUsage_.set(UsageWarning::KnownBadImplicitInterface);
    warnUsage_.set(UsageWarning::EmptyCase);
    warnUsage_.set(UsageWarning::CaseOverflow);
    warnUsage_.set(UsageWarning::CUDAUsage);
    warnUsage_.set(UsageWarning::IgnoreTKRUsage);
    warnUsage_.set(UsageWarning::ExternalInterfaceMismatch);
    warnUsage_.set(UsageWarning::DefinedOperatorArgs);
    warnUsage_.set(UsageWarning::Final);
    warnUsage_.set(UsageWarning::ZeroDoStep);
    warnUsage_.set(UsageWarning::UnusedForallIndex);
    warnUsage_.set(UsageWarning::OpenMPUsage);
    warnUsage_.set(UsageWarning::ModuleFile);
    warnUsage_.set(UsageWarning::DataLength);
    warnUsage_.set(UsageWarning::IgnoredDirective);
    warnUsage_.set(UsageWarning::HomonymousSpecific);
    warnUsage_.set(UsageWarning::HomonymousResult);
    warnUsage_.set(UsageWarning::IgnoredIntrinsicFunctionType);
    warnUsage_.set(UsageWarning::PreviousScalarUse);
    warnUsage_.set(UsageWarning::RedeclaredInaccessibleComponent);
    warnUsage_.set(UsageWarning::ImplicitShared);
    warnUsage_.set(UsageWarning::IndexVarRedefinition);
    warnUsage_.set(UsageWarning::IncompatibleImplicitInterfaces);
    warnUsage_.set(UsageWarning::BadTypeForTarget);
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
