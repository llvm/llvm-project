/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Attribute text node dumper                                                 *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Attr.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

  void VisitAMDGPUNumSGPRAttr(const AMDGPUNumSGPRAttr *A) {
    const auto *SA = cast<AMDGPUNumSGPRAttr>(A); (void)SA;
    OS << " " << SA->getNumSGPR();
  }
  void VisitAMDGPUNumVGPRAttr(const AMDGPUNumVGPRAttr *A) {
    const auto *SA = cast<AMDGPUNumVGPRAttr>(A); (void)SA;
    OS << " " << SA->getNumVGPR();
  }
  void VisitARMInterruptAttr(const ARMInterruptAttr *A) {
    const auto *SA = cast<ARMInterruptAttr>(A); (void)SA;
    switch(SA->getInterrupt()) {
    case ARMInterruptAttr::InterruptType::IRQ:
      OS << " IRQ";
      break;
    case ARMInterruptAttr::InterruptType::FIQ:
      OS << " FIQ";
      break;
    case ARMInterruptAttr::InterruptType::SWI:
      OS << " SWI";
      break;
    case ARMInterruptAttr::InterruptType::ABORT:
      OS << " ABORT";
      break;
    case ARMInterruptAttr::InterruptType::UNDEF:
      OS << " UNDEF";
      break;
    case ARMInterruptAttr::InterruptType::Generic:
      OS << " Generic";
      break;
    }
  }
  void VisitARMInterruptSaveFPAttr(const ARMInterruptSaveFPAttr *A) {
    const auto *SA = cast<ARMInterruptSaveFPAttr>(A); (void)SA;
    switch(SA->getInterrupt()) {
    case ARMInterruptSaveFPAttr::InterruptType::IRQ:
      OS << " IRQ";
      break;
    case ARMInterruptSaveFPAttr::InterruptType::FIQ:
      OS << " FIQ";
      break;
    case ARMInterruptSaveFPAttr::InterruptType::SWI:
      OS << " SWI";
      break;
    case ARMInterruptSaveFPAttr::InterruptType::ABORT:
      OS << " ABORT";
      break;
    case ARMInterruptSaveFPAttr::InterruptType::UNDEF:
      OS << " UNDEF";
      break;
    case ARMInterruptSaveFPAttr::InterruptType::Generic:
      OS << " Generic";
      break;
    }
  }
  void VisitAbiTagAttr(const AbiTagAttr *A) {
    const auto *SA = cast<AbiTagAttr>(A); (void)SA;
    for (const auto &Val : SA->tags())
      OS << " " << Val;
  }
  void VisitAcquireCapabilityAttr(const AcquireCapabilityAttr *A) {
    const auto *SA = cast<AcquireCapabilityAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
  }
  void VisitAcquireHandleAttr(const AcquireHandleAttr *A) {
    const auto *SA = cast<AcquireHandleAttr>(A); (void)SA;
    OS << " \"" << SA->getHandleType() << "\"";
  }
  void VisitAddressSpaceAttr(const AddressSpaceAttr *A) {
    const auto *SA = cast<AddressSpaceAttr>(A); (void)SA;
    OS << " " << SA->getAddressSpace();
  }
  void VisitAliasAttr(const AliasAttr *A) {
    const auto *SA = cast<AliasAttr>(A); (void)SA;
    OS << " \"" << SA->getAliasee() << "\"";
  }
  void VisitAlignedAttr(const AlignedAttr *A) {
    const auto *SA = cast<AlignedAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    if (!SA->isAlignmentExpr())
      dumpType(SA->getAlignmentType()->getType());
  }
  void VisitAllocAlignAttr(const AllocAlignAttr *A) {
    const auto *SA = cast<AllocAlignAttr>(A); (void)SA;
    OS << " " << SA->getParamIndex().getSourceIndex();
  }
  void VisitAllocSizeAttr(const AllocSizeAttr *A) {
    const auto *SA = cast<AllocSizeAttr>(A); (void)SA;
    OS << " " << SA->getElemSizeParam().getSourceIndex();
    if (SA->getNumElemsParam().isValid())
      OS << " " << SA->getNumElemsParam().getSourceIndex();
  }
  void VisitAlwaysInlineAttr(const AlwaysInlineAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitAnnotateAttr(const AnnotateAttr *A) {
    const auto *SA = cast<AnnotateAttr>(A); (void)SA;
    OS << " \"" << SA->getAnnotation() << "\"";
  }
  void VisitAnnotateTypeAttr(const AnnotateTypeAttr *A) {
    const auto *SA = cast<AnnotateTypeAttr>(A); (void)SA;
    OS << " \"" << SA->getAnnotation() << "\"";
  }
  void VisitArgumentWithTypeTagAttr(const ArgumentWithTypeTagAttr *A) {
    const auto *SA = cast<ArgumentWithTypeTagAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    if (SA->getArgumentKind())
      OS << " " << SA->getArgumentKind()->getName();
    OS << " " << SA->getArgumentIdx().getSourceIndex();
    OS << " " << SA->getTypeTagIdx().getSourceIndex();
    if (SA->getIsPointer()) OS << " IsPointer";
  }
  void VisitArmAgnosticAttr(const ArmAgnosticAttr *A) {
    const auto *SA = cast<ArmAgnosticAttr>(A); (void)SA;
    for (const auto &Val : SA->agnosticArgs())
      OS << " " << Val;
  }
  void VisitArmBuiltinAliasAttr(const ArmBuiltinAliasAttr *A) {
    const auto *SA = cast<ArmBuiltinAliasAttr>(A); (void)SA;
    if (SA->getBuiltinName())
      OS << " " << SA->getBuiltinName()->getName();
  }
  void VisitArmInAttr(const ArmInAttr *A) {
    const auto *SA = cast<ArmInAttr>(A); (void)SA;
    for (const auto &Val : SA->inArgs())
      OS << " " << Val;
  }
  void VisitArmInOutAttr(const ArmInOutAttr *A) {
    const auto *SA = cast<ArmInOutAttr>(A); (void)SA;
    for (const auto &Val : SA->inOutArgs())
      OS << " " << Val;
  }
  void VisitArmNewAttr(const ArmNewAttr *A) {
    const auto *SA = cast<ArmNewAttr>(A); (void)SA;
    for (const auto &Val : SA->newArgs())
      OS << " " << Val;
  }
  void VisitArmOutAttr(const ArmOutAttr *A) {
    const auto *SA = cast<ArmOutAttr>(A); (void)SA;
    for (const auto &Val : SA->outArgs())
      OS << " " << Val;
  }
  void VisitArmPreservesAttr(const ArmPreservesAttr *A) {
    const auto *SA = cast<ArmPreservesAttr>(A); (void)SA;
    for (const auto &Val : SA->preserveArgs())
      OS << " " << Val;
  }
  void VisitAsmLabelAttr(const AsmLabelAttr *A) {
    const auto *SA = cast<AsmLabelAttr>(A); (void)SA;
    OS << " \"" << SA->getLabel() << "\"";
    if (SA->getIsLiteralLabel()) OS << " IsLiteralLabel";
  }
  void VisitAssertCapabilityAttr(const AssertCapabilityAttr *A) {
    const auto *SA = cast<AssertCapabilityAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
  }
  void VisitAtomicAttr(const AtomicAttr *A) {
    const auto *SA = cast<AtomicAttr>(A); (void)SA;
    for (AtomicAttr::atomicOptions_iterator I = SA->atomicOptions_begin(), E = SA->atomicOptions_end(); I != E; ++I) {
      switch(*I) {
    case AtomicAttr::ConsumedOption::remote_memory:
      OS << " remote_memory";
      break;
    case AtomicAttr::ConsumedOption::no_remote_memory:
      OS << " no_remote_memory";
      break;
    case AtomicAttr::ConsumedOption::fine_grained_memory:
      OS << " fine_grained_memory";
      break;
    case AtomicAttr::ConsumedOption::no_fine_grained_memory:
      OS << " no_fine_grained_memory";
      break;
    case AtomicAttr::ConsumedOption::ignore_denormal_mode:
      OS << " ignore_denormal_mode";
      break;
    case AtomicAttr::ConsumedOption::no_ignore_denormal_mode:
      OS << " no_ignore_denormal_mode";
      break;
      }
    }
  }
  void VisitAvailabilityAttr(const AvailabilityAttr *A) {
    const auto *SA = cast<AvailabilityAttr>(A); (void)SA;
    if (SA->getPlatform())
      OS << " " << SA->getPlatform()->getName();
    OS << " " << SA->getIntroduced();
    OS << " " << SA->getDeprecated();
    OS << " " << SA->getObsoleted();
    if (SA->getUnavailable()) OS << " Unavailable";
    OS << " \"" << SA->getMessage() << "\"";
    if (SA->getStrict()) OS << " Strict";
    OS << " \"" << SA->getReplacement() << "\"";
    OS << " " << SA->getPriority();
    if (SA->getEnvironment())
      OS << " " << SA->getEnvironment()->getName();
  }
  void VisitBTFDeclTagAttr(const BTFDeclTagAttr *A) {
    const auto *SA = cast<BTFDeclTagAttr>(A); (void)SA;
    OS << " \"" << SA->getBTFDeclTag() << "\"";
  }
  void VisitBTFTypeTagAttr(const BTFTypeTagAttr *A) {
    const auto *SA = cast<BTFTypeTagAttr>(A); (void)SA;
    OS << " \"" << SA->getBTFTypeTag() << "\"";
  }
  void VisitBlocksAttr(const BlocksAttr *A) {
    const auto *SA = cast<BlocksAttr>(A); (void)SA;
    switch(SA->getType()) {
    case BlocksAttr::BlockType::ByRef:
      OS << " ByRef";
      break;
    }
  }
  void VisitBuiltinAttr(const BuiltinAttr *A) {
    const auto *SA = cast<BuiltinAttr>(A); (void)SA;
    OS << " " << SA->getID();
  }
  void VisitBuiltinAliasAttr(const BuiltinAliasAttr *A) {
    const auto *SA = cast<BuiltinAliasAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    if (SA->getBuiltinName())
      OS << " " << SA->getBuiltinName()->getName();
  }
  void VisitCFGuardAttr(const CFGuardAttr *A) {
    const auto *SA = cast<CFGuardAttr>(A); (void)SA;
    switch(SA->getGuard()) {
    case CFGuardAttr::GuardArg::nocf:
      OS << " nocf";
      break;
    }
  }
  void VisitCPUDispatchAttr(const CPUDispatchAttr *A) {
    const auto *SA = cast<CPUDispatchAttr>(A); (void)SA;
    for (const auto &Val : SA->cpus())
      OS << " " << Val;
  }
  void VisitCPUSpecificAttr(const CPUSpecificAttr *A) {
    const auto *SA = cast<CPUSpecificAttr>(A); (void)SA;
    for (const auto &Val : SA->cpus())
      OS << " " << Val;
  }
  void VisitCXX11NoReturnAttr(const CXX11NoReturnAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitCallableWhenAttr(const CallableWhenAttr *A) {
    const auto *SA = cast<CallableWhenAttr>(A); (void)SA;
    for (CallableWhenAttr::callableStates_iterator I = SA->callableStates_begin(), E = SA->callableStates_end(); I != E; ++I) {
      switch(*I) {
    case CallableWhenAttr::ConsumedState::Unknown:
      OS << " Unknown";
      break;
    case CallableWhenAttr::ConsumedState::Consumed:
      OS << " Consumed";
      break;
    case CallableWhenAttr::ConsumedState::Unconsumed:
      OS << " Unconsumed";
      break;
      }
    }
  }
  void VisitCallbackAttr(const CallbackAttr *A) {
    const auto *SA = cast<CallbackAttr>(A); (void)SA;
    for (const auto &Val : SA->encoding())
      OS << " " << Val;
  }
  void VisitCapabilityAttr(const CapabilityAttr *A) {
    const auto *SA = cast<CapabilityAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    OS << " \"" << SA->getName() << "\"";
  }
  void VisitCleanupAttr(const CleanupAttr *A) {
    const auto *SA = cast<CleanupAttr>(A); (void)SA;
    OS << " ";
    dumpBareDeclRef(SA->getFunctionDecl());
  }
  void VisitCodeModelAttr(const CodeModelAttr *A) {
    const auto *SA = cast<CodeModelAttr>(A); (void)SA;
    switch(SA->getModel()) {
    case llvm::CodeModel::Model::Small:
      OS << " Small";
      break;
    case llvm::CodeModel::Model::Medium:
      OS << " Medium";
      break;
    case llvm::CodeModel::Model::Large:
      OS << " Large";
      break;
    default:
      llvm_unreachable("Invalid attribute value");
    }
  }
  void VisitCodeSegAttr(const CodeSegAttr *A) {
    const auto *SA = cast<CodeSegAttr>(A); (void)SA;
    OS << " \"" << SA->getName() << "\"";
  }
  void VisitConstInitAttr(const ConstInitAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitConstructorAttr(const ConstructorAttr *A) {
    const auto *SA = cast<ConstructorAttr>(A); (void)SA;
    OS << " " << SA->getPriority();
  }
  void VisitConsumableAttr(const ConsumableAttr *A) {
    const auto *SA = cast<ConsumableAttr>(A); (void)SA;
    switch(SA->getDefaultState()) {
    case ConsumableAttr::ConsumedState::Unknown:
      OS << " Unknown";
      break;
    case ConsumableAttr::ConsumedState::Consumed:
      OS << " Consumed";
      break;
    case ConsumableAttr::ConsumedState::Unconsumed:
      OS << " Unconsumed";
      break;
    }
  }
  void VisitCountedByAttr(const CountedByAttr *A) {
    const auto *SA = cast<CountedByAttr>(A); (void)SA;
    OS << " " << SA->getNestedLevel();
  }
  void VisitCountedByOrNullAttr(const CountedByOrNullAttr *A) {
    const auto *SA = cast<CountedByOrNullAttr>(A); (void)SA;
    OS << " " << SA->getNestedLevel();
  }
  void VisitDeprecatedAttr(const DeprecatedAttr *A) {
    const auto *SA = cast<DeprecatedAttr>(A); (void)SA;
    OS << " \"" << SA->getMessage() << "\"";
    OS << " \"" << SA->getReplacement() << "\"";
  }
  void VisitDestructorAttr(const DestructorAttr *A) {
    const auto *SA = cast<DestructorAttr>(A); (void)SA;
    OS << " " << SA->getPriority();
  }
  void VisitDiagnoseAsBuiltinAttr(const DiagnoseAsBuiltinAttr *A) {
    const auto *SA = cast<DiagnoseAsBuiltinAttr>(A); (void)SA;
    OS << " ";
    dumpBareDeclRef(SA->getFunction());
    for (const auto &Val : SA->argIndices())
      OS << " " << Val;
  }
  void VisitDiagnoseIfAttr(const DiagnoseIfAttr *A) {
    const auto *SA = cast<DiagnoseIfAttr>(A); (void)SA;
    OS << " \"" << SA->getMessage() << "\"";
    switch(SA->getDefaultSeverity()) {
    case DiagnoseIfAttr::DefaultSeverity::DS_error:
      OS << " DS_error";
      break;
    case DiagnoseIfAttr::DefaultSeverity::DS_warning:
      OS << " DS_warning";
      break;
    }
    OS << " \"" << SA->getWarningGroup() << "\"";
    if (SA->getArgDependent()) OS << " ArgDependent";
    OS << " ";
    dumpBareDeclRef(SA->getParent());
  }
  void VisitEnableIfAttr(const EnableIfAttr *A) {
    const auto *SA = cast<EnableIfAttr>(A); (void)SA;
    OS << " \"" << SA->getMessage() << "\"";
  }
  void VisitEnforceTCBAttr(const EnforceTCBAttr *A) {
    const auto *SA = cast<EnforceTCBAttr>(A); (void)SA;
    OS << " \"" << SA->getTCBName() << "\"";
  }
  void VisitEnforceTCBLeafAttr(const EnforceTCBLeafAttr *A) {
    const auto *SA = cast<EnforceTCBLeafAttr>(A); (void)SA;
    OS << " \"" << SA->getTCBName() << "\"";
  }
  void VisitEnumExtensibilityAttr(const EnumExtensibilityAttr *A) {
    const auto *SA = cast<EnumExtensibilityAttr>(A); (void)SA;
    switch(SA->getExtensibility()) {
    case EnumExtensibilityAttr::Kind::Closed:
      OS << " Closed";
      break;
    case EnumExtensibilityAttr::Kind::Open:
      OS << " Open";
      break;
    }
  }
  void VisitErrorAttr(const ErrorAttr *A) {
    const auto *SA = cast<ErrorAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    OS << " \"" << SA->getUserDiagnostic() << "\"";
  }
  void VisitExternalSourceSymbolAttr(const ExternalSourceSymbolAttr *A) {
    const auto *SA = cast<ExternalSourceSymbolAttr>(A); (void)SA;
    OS << " \"" << SA->getLanguage() << "\"";
    OS << " \"" << SA->getDefinedIn() << "\"";
    if (SA->getGeneratedDeclaration()) OS << " GeneratedDeclaration";
    OS << " \"" << SA->getUSR() << "\"";
  }
  void VisitFinalAttr(const FinalAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitFormatAttr(const FormatAttr *A) {
    const auto *SA = cast<FormatAttr>(A); (void)SA;
    if (SA->getType())
      OS << " " << SA->getType()->getName();
    OS << " " << SA->getFormatIdx();
    OS << " " << SA->getFirstArg();
  }
  void VisitFormatArgAttr(const FormatArgAttr *A) {
    const auto *SA = cast<FormatArgAttr>(A); (void)SA;
    OS << " " << SA->getFormatIdx().getSourceIndex();
  }
  void VisitFormatMatchesAttr(const FormatMatchesAttr *A) {
    const auto *SA = cast<FormatMatchesAttr>(A); (void)SA;
    if (SA->getType())
      OS << " " << SA->getType()->getName();
    OS << " " << SA->getFormatIdx();
  }
  void VisitFunctionReturnThunksAttr(const FunctionReturnThunksAttr *A) {
    const auto *SA = cast<FunctionReturnThunksAttr>(A); (void)SA;
    switch(SA->getThunkType()) {
    case FunctionReturnThunksAttr::Kind::Keep:
      OS << " Keep";
      break;
    case FunctionReturnThunksAttr::Kind::Extern:
      OS << " Extern";
      break;
    }
  }
  void VisitHLSLContainedTypeAttr(const HLSLContainedTypeAttr *A) {
    const auto *SA = cast<HLSLContainedTypeAttr>(A); (void)SA;
    OS << " " << SA->getType().getAsString();
  }
  void VisitHLSLControlFlowHintAttr(const HLSLControlFlowHintAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitHLSLLoopHintAttr(const HLSLLoopHintAttr *A) {
    const auto *SA = cast<HLSLLoopHintAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    OS << " " << SA->getDirective();
  }
  void VisitHLSLNumThreadsAttr(const HLSLNumThreadsAttr *A) {
    const auto *SA = cast<HLSLNumThreadsAttr>(A); (void)SA;
    OS << " " << SA->getX();
    OS << " " << SA->getY();
    OS << " " << SA->getZ();
  }
  void VisitHLSLPackOffsetAttr(const HLSLPackOffsetAttr *A) {
    const auto *SA = cast<HLSLPackOffsetAttr>(A); (void)SA;
    OS << " " << SA->getSubcomponent();
    OS << " " << SA->getComponent();
  }
  void VisitHLSLParamModifierAttr(const HLSLParamModifierAttr *A) {
    const auto *SA = cast<HLSLParamModifierAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    if (SA->getMergedSpelling()) OS << " MergedSpelling";
  }
  void VisitHLSLResourceBindingAttr(const HLSLResourceBindingAttr *A) {
    const auto *SA = cast<HLSLResourceBindingAttr>(A); (void)SA;
    OS << " \"" << SA->getSlot() << "\"";
    OS << " \"" << SA->getSpace() << "\"";
  }
  void VisitHLSLResourceClassAttr(const HLSLResourceClassAttr *A) {
    const auto *SA = cast<HLSLResourceClassAttr>(A); (void)SA;
    switch(SA->getResourceClass()) {
    case llvm::hlsl::ResourceClass::SRV:
      OS << " SRV";
      break;
    case llvm::hlsl::ResourceClass::UAV:
      OS << " UAV";
      break;
    case llvm::hlsl::ResourceClass::CBuffer:
      OS << " CBuffer";
      break;
    case llvm::hlsl::ResourceClass::Sampler:
      OS << " Sampler";
      break;
    }
  }
  void VisitHLSLShaderAttr(const HLSLShaderAttr *A) {
    const auto *SA = cast<HLSLShaderAttr>(A); (void)SA;
    switch(SA->getType()) {
    case llvm::Triple::EnvironmentType::Pixel:
      OS << " Pixel";
      break;
    case llvm::Triple::EnvironmentType::Vertex:
      OS << " Vertex";
      break;
    case llvm::Triple::EnvironmentType::Geometry:
      OS << " Geometry";
      break;
    case llvm::Triple::EnvironmentType::Hull:
      OS << " Hull";
      break;
    case llvm::Triple::EnvironmentType::Domain:
      OS << " Domain";
      break;
    case llvm::Triple::EnvironmentType::Compute:
      OS << " Compute";
      break;
    case llvm::Triple::EnvironmentType::RayGeneration:
      OS << " RayGeneration";
      break;
    case llvm::Triple::EnvironmentType::Intersection:
      OS << " Intersection";
      break;
    case llvm::Triple::EnvironmentType::AnyHit:
      OS << " AnyHit";
      break;
    case llvm::Triple::EnvironmentType::ClosestHit:
      OS << " ClosestHit";
      break;
    case llvm::Triple::EnvironmentType::Miss:
      OS << " Miss";
      break;
    case llvm::Triple::EnvironmentType::Callable:
      OS << " Callable";
      break;
    case llvm::Triple::EnvironmentType::Mesh:
      OS << " Mesh";
      break;
    case llvm::Triple::EnvironmentType::Amplification:
      OS << " Amplification";
      break;
    default:
      llvm_unreachable("Invalid attribute value");
    }
  }
  void VisitHLSLWaveSizeAttr(const HLSLWaveSizeAttr *A) {
    const auto *SA = cast<HLSLWaveSizeAttr>(A); (void)SA;
    OS << " " << SA->getMin();
    OS << " " << SA->getMax();
    OS << " " << SA->getPreferred();
  }
  void VisitIBOutletCollectionAttr(const IBOutletCollectionAttr *A) {
    const auto *SA = cast<IBOutletCollectionAttr>(A); (void)SA;
    if (SA->getInterfaceLoc())    OS << " " << SA->getInterface().getAsString();
  }
  void VisitIFuncAttr(const IFuncAttr *A) {
    const auto *SA = cast<IFuncAttr>(A); (void)SA;
    OS << " \"" << SA->getResolver() << "\"";
  }
  void VisitInitPriorityAttr(const InitPriorityAttr *A) {
    const auto *SA = cast<InitPriorityAttr>(A); (void)SA;
    OS << " " << SA->getPriority();
  }
  void VisitInitSegAttr(const InitSegAttr *A) {
    const auto *SA = cast<InitSegAttr>(A); (void)SA;
    OS << " \"" << SA->getSection() << "\"";
  }
  void VisitLayoutVersionAttr(const LayoutVersionAttr *A) {
    const auto *SA = cast<LayoutVersionAttr>(A); (void)SA;
    OS << " " << SA->getVersion();
  }
  void VisitLifetimeCaptureByAttr(const LifetimeCaptureByAttr *A) {
    const auto *SA = cast<LifetimeCaptureByAttr>(A); (void)SA;
    for (const auto &Val : SA->params())
      OS << " " << Val;
  }
  void VisitLoopHintAttr(const LoopHintAttr *A) {
    const auto *SA = cast<LoopHintAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    switch(SA->getOption()) {
    case LoopHintAttr::OptionType::Vectorize:
      OS << " Vectorize";
      break;
    case LoopHintAttr::OptionType::VectorizeWidth:
      OS << " VectorizeWidth";
      break;
    case LoopHintAttr::OptionType::Interleave:
      OS << " Interleave";
      break;
    case LoopHintAttr::OptionType::InterleaveCount:
      OS << " InterleaveCount";
      break;
    case LoopHintAttr::OptionType::Unroll:
      OS << " Unroll";
      break;
    case LoopHintAttr::OptionType::UnrollCount:
      OS << " UnrollCount";
      break;
    case LoopHintAttr::OptionType::UnrollAndJam:
      OS << " UnrollAndJam";
      break;
    case LoopHintAttr::OptionType::UnrollAndJamCount:
      OS << " UnrollAndJamCount";
      break;
    case LoopHintAttr::OptionType::PipelineDisabled:
      OS << " PipelineDisabled";
      break;
    case LoopHintAttr::OptionType::PipelineInitiationInterval:
      OS << " PipelineInitiationInterval";
      break;
    case LoopHintAttr::OptionType::Distribute:
      OS << " Distribute";
      break;
    case LoopHintAttr::OptionType::VectorizePredicate:
      OS << " VectorizePredicate";
      break;
    }
    switch(SA->getState()) {
    case LoopHintAttr::LoopHintState::Enable:
      OS << " Enable";
      break;
    case LoopHintAttr::LoopHintState::Disable:
      OS << " Disable";
      break;
    case LoopHintAttr::LoopHintState::Numeric:
      OS << " Numeric";
      break;
    case LoopHintAttr::LoopHintState::FixedWidth:
      OS << " FixedWidth";
      break;
    case LoopHintAttr::LoopHintState::ScalableWidth:
      OS << " ScalableWidth";
      break;
    case LoopHintAttr::LoopHintState::AssumeSafety:
      OS << " AssumeSafety";
      break;
    case LoopHintAttr::LoopHintState::Full:
      OS << " Full";
      break;
    }
  }
  void VisitM68kInterruptAttr(const M68kInterruptAttr *A) {
    const auto *SA = cast<M68kInterruptAttr>(A); (void)SA;
    OS << " " << SA->getNumber();
  }
  void VisitMSInheritanceAttr(const MSInheritanceAttr *A) {
    const auto *SA = cast<MSInheritanceAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    if (SA->getBestCase()) OS << " BestCase";
  }
  void VisitMSP430InterruptAttr(const MSP430InterruptAttr *A) {
    const auto *SA = cast<MSP430InterruptAttr>(A); (void)SA;
    OS << " " << SA->getNumber();
  }
  void VisitMSVtorDispAttr(const MSVtorDispAttr *A) {
    const auto *SA = cast<MSVtorDispAttr>(A); (void)SA;
    OS << " " << SA->getVdm();
  }
  void VisitMaxFieldAlignmentAttr(const MaxFieldAlignmentAttr *A) {
    const auto *SA = cast<MaxFieldAlignmentAttr>(A); (void)SA;
    OS << " " << SA->getAlignment();
  }
  void VisitMinVectorWidthAttr(const MinVectorWidthAttr *A) {
    const auto *SA = cast<MinVectorWidthAttr>(A); (void)SA;
    OS << " " << SA->getVectorWidth();
  }
  void VisitMipsInterruptAttr(const MipsInterruptAttr *A) {
    const auto *SA = cast<MipsInterruptAttr>(A); (void)SA;
    switch(SA->getInterrupt()) {
    case MipsInterruptAttr::InterruptType::sw0:
      OS << " sw0";
      break;
    case MipsInterruptAttr::InterruptType::sw1:
      OS << " sw1";
      break;
    case MipsInterruptAttr::InterruptType::hw0:
      OS << " hw0";
      break;
    case MipsInterruptAttr::InterruptType::hw1:
      OS << " hw1";
      break;
    case MipsInterruptAttr::InterruptType::hw2:
      OS << " hw2";
      break;
    case MipsInterruptAttr::InterruptType::hw3:
      OS << " hw3";
      break;
    case MipsInterruptAttr::InterruptType::hw4:
      OS << " hw4";
      break;
    case MipsInterruptAttr::InterruptType::hw5:
      OS << " hw5";
      break;
    case MipsInterruptAttr::InterruptType::eic:
      OS << " eic";
      break;
    }
  }
  void VisitMipsLongCallAttr(const MipsLongCallAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitMipsShortCallAttr(const MipsShortCallAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitModeAttr(const ModeAttr *A) {
    const auto *SA = cast<ModeAttr>(A); (void)SA;
    if (SA->getMode())
      OS << " " << SA->getMode()->getName();
  }
  void VisitNSErrorDomainAttr(const NSErrorDomainAttr *A) {
    const auto *SA = cast<NSErrorDomainAttr>(A); (void)SA;
    if (SA->getErrorDomain())
      OS << " " << SA->getErrorDomain()->getName();
  }
  void VisitNoBuiltinAttr(const NoBuiltinAttr *A) {
    const auto *SA = cast<NoBuiltinAttr>(A); (void)SA;
    for (const auto &Val : SA->builtinNames())
      OS << " " << Val;
  }
  void VisitNoSanitizeAttr(const NoSanitizeAttr *A) {
    const auto *SA = cast<NoSanitizeAttr>(A); (void)SA;
    for (const auto &Val : SA->sanitizers())
      OS << " " << Val;
  }
  void VisitNoSpecializationsAttr(const NoSpecializationsAttr *A) {
    const auto *SA = cast<NoSpecializationsAttr>(A); (void)SA;
    OS << " \"" << SA->getMessage() << "\"";
  }
  void VisitNoStackProtectorAttr(const NoStackProtectorAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitNonNullAttr(const NonNullAttr *A) {
    const auto *SA = cast<NonNullAttr>(A); (void)SA;
    for (const auto &Val : SA->args())
      OS << " " << Val.getSourceIndex();
  }
  void VisitOMPAllocateDeclAttr(const OMPAllocateDeclAttr *A) {
    const auto *SA = cast<OMPAllocateDeclAttr>(A); (void)SA;
    switch(SA->getAllocatorType()) {
    case OMPAllocateDeclAttr::AllocatorTypeTy::OMPNullMemAlloc:
      OS << " OMPNullMemAlloc";
      break;
    case OMPAllocateDeclAttr::AllocatorTypeTy::OMPDefaultMemAlloc:
      OS << " OMPDefaultMemAlloc";
      break;
    case OMPAllocateDeclAttr::AllocatorTypeTy::OMPLargeCapMemAlloc:
      OS << " OMPLargeCapMemAlloc";
      break;
    case OMPAllocateDeclAttr::AllocatorTypeTy::OMPConstMemAlloc:
      OS << " OMPConstMemAlloc";
      break;
    case OMPAllocateDeclAttr::AllocatorTypeTy::OMPHighBWMemAlloc:
      OS << " OMPHighBWMemAlloc";
      break;
    case OMPAllocateDeclAttr::AllocatorTypeTy::OMPLowLatMemAlloc:
      OS << " OMPLowLatMemAlloc";
      break;
    case OMPAllocateDeclAttr::AllocatorTypeTy::OMPCGroupMemAlloc:
      OS << " OMPCGroupMemAlloc";
      break;
    case OMPAllocateDeclAttr::AllocatorTypeTy::OMPPTeamMemAlloc:
      OS << " OMPPTeamMemAlloc";
      break;
    case OMPAllocateDeclAttr::AllocatorTypeTy::OMPThreadMemAlloc:
      OS << " OMPThreadMemAlloc";
      break;
    case OMPAllocateDeclAttr::AllocatorTypeTy::OMPUserDefinedMemAlloc:
      OS << " OMPUserDefinedMemAlloc";
      break;
    }
  }
  void VisitOMPAssumeAttr(const OMPAssumeAttr *A) {
    const auto *SA = cast<OMPAssumeAttr>(A); (void)SA;
    OS << " \"" << SA->getAssumption() << "\"";
  }
  void VisitOMPCaptureKindAttr(const OMPCaptureKindAttr *A) {
    const auto *SA = cast<OMPCaptureKindAttr>(A); (void)SA;
    OS << " " << SA->getCaptureKindVal();
  }
  void VisitOMPDeclareSimdDeclAttr(const OMPDeclareSimdDeclAttr *A) {
    const auto *SA = cast<OMPDeclareSimdDeclAttr>(A); (void)SA;
    switch(SA->getBranchState()) {
    case OMPDeclareSimdDeclAttr::BranchStateTy::BS_Undefined:
      OS << " BS_Undefined";
      break;
    case OMPDeclareSimdDeclAttr::BranchStateTy::BS_Inbranch:
      OS << " BS_Inbranch";
      break;
    case OMPDeclareSimdDeclAttr::BranchStateTy::BS_Notinbranch:
      OS << " BS_Notinbranch";
      break;
    }
    for (const auto &Val : SA->modifiers())
      OS << " " << Val;
  }
  void VisitOMPDeclareTargetDeclAttr(const OMPDeclareTargetDeclAttr *A) {
    const auto *SA = cast<OMPDeclareTargetDeclAttr>(A); (void)SA;
    switch(SA->getMapType()) {
    case OMPDeclareTargetDeclAttr::MapTypeTy::MT_To:
      OS << " MT_To";
      break;
    case OMPDeclareTargetDeclAttr::MapTypeTy::MT_Enter:
      OS << " MT_Enter";
      break;
    case OMPDeclareTargetDeclAttr::MapTypeTy::MT_Link:
      OS << " MT_Link";
      break;
    }
    switch(SA->getDevType()) {
    case OMPDeclareTargetDeclAttr::DevTypeTy::DT_Host:
      OS << " DT_Host";
      break;
    case OMPDeclareTargetDeclAttr::DevTypeTy::DT_NoHost:
      OS << " DT_NoHost";
      break;
    case OMPDeclareTargetDeclAttr::DevTypeTy::DT_Any:
      OS << " DT_Any";
      break;
    }
    if (SA->getIndirect()) OS << " Indirect";
    OS << " " << SA->getLevel();
  }
  void VisitOMPDeclareVariantAttr(const OMPDeclareVariantAttr *A) {
    const auto *SA = cast<OMPDeclareVariantAttr>(A); (void)SA;
    OS << " " << SA->getTraitInfos();
    for (OMPDeclareVariantAttr::appendArgs_iterator I = SA->appendArgs_begin(), E = SA->appendArgs_end(); I != E; ++I) {
      if (I->IsTarget && I->IsTargetSync)
        OS << " Target_TargetSync";
      else if (I->IsTarget)
        OS << " Target";
      else
        OS << " TargetSync";
    }
  }
  void VisitObjCBridgeAttr(const ObjCBridgeAttr *A) {
    const auto *SA = cast<ObjCBridgeAttr>(A); (void)SA;
    if (SA->getBridgedType())
      OS << " " << SA->getBridgedType()->getName();
  }
  void VisitObjCBridgeMutableAttr(const ObjCBridgeMutableAttr *A) {
    const auto *SA = cast<ObjCBridgeMutableAttr>(A); (void)SA;
    if (SA->getBridgedType())
      OS << " " << SA->getBridgedType()->getName();
  }
  void VisitObjCBridgeRelatedAttr(const ObjCBridgeRelatedAttr *A) {
    const auto *SA = cast<ObjCBridgeRelatedAttr>(A); (void)SA;
    if (SA->getRelatedClass())
      OS << " " << SA->getRelatedClass()->getName();
    if (SA->getClassMethod())
      OS << " " << SA->getClassMethod()->getName();
    if (SA->getInstanceMethod())
      OS << " " << SA->getInstanceMethod()->getName();
  }
  void VisitObjCGCAttr(const ObjCGCAttr *A) {
    const auto *SA = cast<ObjCGCAttr>(A); (void)SA;
    if (SA->getKind())
      OS << " " << SA->getKind()->getName();
  }
  void VisitObjCMethodFamilyAttr(const ObjCMethodFamilyAttr *A) {
    const auto *SA = cast<ObjCMethodFamilyAttr>(A); (void)SA;
    switch(SA->getFamily()) {
    case ObjCMethodFamilyAttr::FamilyKind::OMF_None:
      OS << " OMF_None";
      break;
    case ObjCMethodFamilyAttr::FamilyKind::OMF_alloc:
      OS << " OMF_alloc";
      break;
    case ObjCMethodFamilyAttr::FamilyKind::OMF_copy:
      OS << " OMF_copy";
      break;
    case ObjCMethodFamilyAttr::FamilyKind::OMF_init:
      OS << " OMF_init";
      break;
    case ObjCMethodFamilyAttr::FamilyKind::OMF_mutableCopy:
      OS << " OMF_mutableCopy";
      break;
    case ObjCMethodFamilyAttr::FamilyKind::OMF_new:
      OS << " OMF_new";
      break;
    }
  }
  void VisitObjCOwnershipAttr(const ObjCOwnershipAttr *A) {
    const auto *SA = cast<ObjCOwnershipAttr>(A); (void)SA;
    if (SA->getKind())
      OS << " " << SA->getKind()->getName();
  }
  void VisitObjCRuntimeNameAttr(const ObjCRuntimeNameAttr *A) {
    const auto *SA = cast<ObjCRuntimeNameAttr>(A); (void)SA;
    OS << " \"" << SA->getMetadataName() << "\"";
  }
  void VisitOpenCLAccessAttr(const OpenCLAccessAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitOpenCLConstantAddressSpaceAttr(const OpenCLConstantAddressSpaceAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitOpenCLGenericAddressSpaceAttr(const OpenCLGenericAddressSpaceAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitOpenCLGlobalAddressSpaceAttr(const OpenCLGlobalAddressSpaceAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitOpenCLIntelReqdSubGroupSizeAttr(const OpenCLIntelReqdSubGroupSizeAttr *A) {
    const auto *SA = cast<OpenCLIntelReqdSubGroupSizeAttr>(A); (void)SA;
    OS << " " << SA->getSubGroupSize();
  }
  void VisitOpenCLLocalAddressSpaceAttr(const OpenCLLocalAddressSpaceAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitOpenCLPrivateAddressSpaceAttr(const OpenCLPrivateAddressSpaceAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitOpenCLUnrollHintAttr(const OpenCLUnrollHintAttr *A) {
    const auto *SA = cast<OpenCLUnrollHintAttr>(A); (void)SA;
    OS << " " << SA->getUnrollHint();
  }
  void VisitOwnerAttr(const OwnerAttr *A) {
    const auto *SA = cast<OwnerAttr>(A); (void)SA;
    if (SA->getDerefTypeLoc())    OS << " " << SA->getDerefType().getAsString();
  }
  void VisitOwnershipAttr(const OwnershipAttr *A) {
    const auto *SA = cast<OwnershipAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    if (SA->getModule())
      OS << " " << SA->getModule()->getName();
    for (const auto &Val : SA->args())
      OS << " " << Val.getSourceIndex();
  }
  void VisitParamTypestateAttr(const ParamTypestateAttr *A) {
    const auto *SA = cast<ParamTypestateAttr>(A); (void)SA;
    switch(SA->getParamState()) {
    case ParamTypestateAttr::ConsumedState::Unknown:
      OS << " Unknown";
      break;
    case ParamTypestateAttr::ConsumedState::Consumed:
      OS << " Consumed";
      break;
    case ParamTypestateAttr::ConsumedState::Unconsumed:
      OS << " Unconsumed";
      break;
    }
  }
  void VisitPassObjectSizeAttr(const PassObjectSizeAttr *A) {
    const auto *SA = cast<PassObjectSizeAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    OS << " " << SA->getType();
  }
  void VisitPatchableFunctionEntryAttr(const PatchableFunctionEntryAttr *A) {
    const auto *SA = cast<PatchableFunctionEntryAttr>(A); (void)SA;
    OS << " " << SA->getCount();
    OS << " " << SA->getOffset();
    OS << " \"" << SA->getSection() << "\"";
  }
  void VisitPcsAttr(const PcsAttr *A) {
    const auto *SA = cast<PcsAttr>(A); (void)SA;
    switch(SA->getPCS()) {
    case PcsAttr::PCSType::AAPCS:
      OS << " AAPCS";
      break;
    case PcsAttr::PCSType::AAPCS_VFP:
      OS << " AAPCS_VFP";
      break;
    }
  }
  void VisitPointerAttr(const PointerAttr *A) {
    const auto *SA = cast<PointerAttr>(A); (void)SA;
    if (SA->getDerefTypeLoc())    OS << " " << SA->getDerefType().getAsString();
  }
  void VisitPointerAuthAttr(const PointerAuthAttr *A) {
    const auto *SA = cast<PointerAuthAttr>(A); (void)SA;
    OS << " " << SA->getKey();
    if (SA->getAddressDiscriminated()) OS << " AddressDiscriminated";
    OS << " " << SA->getExtraDiscriminator();
  }
  void VisitPragmaClangBSSSectionAttr(const PragmaClangBSSSectionAttr *A) {
    const auto *SA = cast<PragmaClangBSSSectionAttr>(A); (void)SA;
    OS << " \"" << SA->getName() << "\"";
  }
  void VisitPragmaClangDataSectionAttr(const PragmaClangDataSectionAttr *A) {
    const auto *SA = cast<PragmaClangDataSectionAttr>(A); (void)SA;
    OS << " \"" << SA->getName() << "\"";
  }
  void VisitPragmaClangRelroSectionAttr(const PragmaClangRelroSectionAttr *A) {
    const auto *SA = cast<PragmaClangRelroSectionAttr>(A); (void)SA;
    OS << " \"" << SA->getName() << "\"";
  }
  void VisitPragmaClangRodataSectionAttr(const PragmaClangRodataSectionAttr *A) {
    const auto *SA = cast<PragmaClangRodataSectionAttr>(A); (void)SA;
    OS << " \"" << SA->getName() << "\"";
  }
  void VisitPragmaClangTextSectionAttr(const PragmaClangTextSectionAttr *A) {
    const auto *SA = cast<PragmaClangTextSectionAttr>(A); (void)SA;
    OS << " \"" << SA->getName() << "\"";
  }
  void VisitPreferredNameAttr(const PreferredNameAttr *A) {
    const auto *SA = cast<PreferredNameAttr>(A); (void)SA;
    OS << " " << SA->getTypedefType().getAsString();
  }
  void VisitPreferredTypeAttr(const PreferredTypeAttr *A) {
    const auto *SA = cast<PreferredTypeAttr>(A); (void)SA;
    if (SA->getTypeLoc())    OS << " " << SA->getType().getAsString();
  }
  void VisitRISCVInterruptAttr(const RISCVInterruptAttr *A) {
    const auto *SA = cast<RISCVInterruptAttr>(A); (void)SA;
    switch(SA->getInterrupt()) {
    case RISCVInterruptAttr::InterruptType::supervisor:
      OS << " supervisor";
      break;
    case RISCVInterruptAttr::InterruptType::machine:
      OS << " machine";
      break;
    case RISCVInterruptAttr::InterruptType::qcinest:
      OS << " qcinest";
      break;
    case RISCVInterruptAttr::InterruptType::qcinonest:
      OS << " qcinonest";
      break;
    }
  }
  void VisitRISCVVLSCCAttr(const RISCVVLSCCAttr *A) {
    const auto *SA = cast<RISCVVLSCCAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    OS << " " << SA->getVectorWidth();
  }
  void VisitRISCVVectorCCAttr(const RISCVVectorCCAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitReleaseCapabilityAttr(const ReleaseCapabilityAttr *A) {
    const auto *SA = cast<ReleaseCapabilityAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
  }
  void VisitReleaseHandleAttr(const ReleaseHandleAttr *A) {
    const auto *SA = cast<ReleaseHandleAttr>(A); (void)SA;
    OS << " \"" << SA->getHandleType() << "\"";
  }
  void VisitRequiresCapabilityAttr(const RequiresCapabilityAttr *A) {
    const auto *SA = cast<RequiresCapabilityAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
  }
  void VisitRestrictAttr(const RestrictAttr *A) {
    const auto *SA = cast<RestrictAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    if (SA->getDeallocatorPtrArgIndex().isValid())
      OS << " " << SA->getDeallocatorPtrArgIndex().getSourceIndex();
  }
  void VisitReturnTypestateAttr(const ReturnTypestateAttr *A) {
    const auto *SA = cast<ReturnTypestateAttr>(A); (void)SA;
    switch(SA->getState()) {
    case ReturnTypestateAttr::ConsumedState::Unknown:
      OS << " Unknown";
      break;
    case ReturnTypestateAttr::ConsumedState::Consumed:
      OS << " Consumed";
      break;
    case ReturnTypestateAttr::ConsumedState::Unconsumed:
      OS << " Unconsumed";
      break;
    }
  }
  void VisitSYCLKernelEntryPointAttr(const SYCLKernelEntryPointAttr *A) {
    const auto *SA = cast<SYCLKernelEntryPointAttr>(A); (void)SA;
    OS << " " << SA->getKernelName().getAsString();
    if (SA->getInvalid()) OS << " Invalid";
  }
  void VisitSectionAttr(const SectionAttr *A) {
    const auto *SA = cast<SectionAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    OS << " \"" << SA->getName() << "\"";
  }
  void VisitSentinelAttr(const SentinelAttr *A) {
    const auto *SA = cast<SentinelAttr>(A); (void)SA;
    OS << " " << SA->getSentinel();
    OS << " " << SA->getNullPos();
  }
  void VisitSetTypestateAttr(const SetTypestateAttr *A) {
    const auto *SA = cast<SetTypestateAttr>(A); (void)SA;
    switch(SA->getNewState()) {
    case SetTypestateAttr::ConsumedState::Unknown:
      OS << " Unknown";
      break;
    case SetTypestateAttr::ConsumedState::Consumed:
      OS << " Consumed";
      break;
    case SetTypestateAttr::ConsumedState::Unconsumed:
      OS << " Unconsumed";
      break;
    }
  }
  void VisitSizedByAttr(const SizedByAttr *A) {
    const auto *SA = cast<SizedByAttr>(A); (void)SA;
    OS << " " << SA->getNestedLevel();
  }
  void VisitSizedByOrNullAttr(const SizedByOrNullAttr *A) {
    const auto *SA = cast<SizedByOrNullAttr>(A); (void)SA;
    OS << " " << SA->getNestedLevel();
  }
  void VisitSuppressAttr(const SuppressAttr *A) {
    const auto *SA = cast<SuppressAttr>(A); (void)SA;
    for (const auto &Val : SA->diagnosticIdentifiers())
      OS << " " << Val;
  }
  void VisitSwiftAsyncAttr(const SwiftAsyncAttr *A) {
    const auto *SA = cast<SwiftAsyncAttr>(A); (void)SA;
    switch(SA->getKind()) {
    case SwiftAsyncAttr::Kind::None:
      OS << " None";
      break;
    case SwiftAsyncAttr::Kind::SwiftPrivate:
      OS << " SwiftPrivate";
      break;
    case SwiftAsyncAttr::Kind::NotSwiftPrivate:
      OS << " NotSwiftPrivate";
      break;
    }
    if (SA->getCompletionHandlerIndex().isValid())
      OS << " " << SA->getCompletionHandlerIndex().getSourceIndex();
  }
  void VisitSwiftAsyncErrorAttr(const SwiftAsyncErrorAttr *A) {
    const auto *SA = cast<SwiftAsyncErrorAttr>(A); (void)SA;
    switch(SA->getConvention()) {
    case SwiftAsyncErrorAttr::ConventionKind::None:
      OS << " None";
      break;
    case SwiftAsyncErrorAttr::ConventionKind::NonNullError:
      OS << " NonNullError";
      break;
    case SwiftAsyncErrorAttr::ConventionKind::ZeroArgument:
      OS << " ZeroArgument";
      break;
    case SwiftAsyncErrorAttr::ConventionKind::NonZeroArgument:
      OS << " NonZeroArgument";
      break;
    }
    OS << " " << SA->getHandlerParamIdx();
  }
  void VisitSwiftAsyncNameAttr(const SwiftAsyncNameAttr *A) {
    const auto *SA = cast<SwiftAsyncNameAttr>(A); (void)SA;
    OS << " \"" << SA->getName() << "\"";
  }
  void VisitSwiftAttrAttr(const SwiftAttrAttr *A) {
    const auto *SA = cast<SwiftAttrAttr>(A); (void)SA;
    OS << " \"" << SA->getAttribute() << "\"";
  }
  void VisitSwiftBridgeAttr(const SwiftBridgeAttr *A) {
    const auto *SA = cast<SwiftBridgeAttr>(A); (void)SA;
    OS << " \"" << SA->getSwiftType() << "\"";
  }
  void VisitSwiftErrorAttr(const SwiftErrorAttr *A) {
    const auto *SA = cast<SwiftErrorAttr>(A); (void)SA;
    switch(SA->getConvention()) {
    case SwiftErrorAttr::ConventionKind::None:
      OS << " None";
      break;
    case SwiftErrorAttr::ConventionKind::NonNullError:
      OS << " NonNullError";
      break;
    case SwiftErrorAttr::ConventionKind::NullResult:
      OS << " NullResult";
      break;
    case SwiftErrorAttr::ConventionKind::ZeroResult:
      OS << " ZeroResult";
      break;
    case SwiftErrorAttr::ConventionKind::NonZeroResult:
      OS << " NonZeroResult";
      break;
    }
  }
  void VisitSwiftNameAttr(const SwiftNameAttr *A) {
    const auto *SA = cast<SwiftNameAttr>(A); (void)SA;
    OS << " \"" << SA->getName() << "\"";
  }
  void VisitSwiftNewTypeAttr(const SwiftNewTypeAttr *A) {
    const auto *SA = cast<SwiftNewTypeAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    switch(SA->getNewtypeKind()) {
    case SwiftNewTypeAttr::NewtypeKind::NK_Struct:
      OS << " NK_Struct";
      break;
    case SwiftNewTypeAttr::NewtypeKind::NK_Enum:
      OS << " NK_Enum";
      break;
    }
  }
  void VisitSwiftVersionedAdditionAttr(const SwiftVersionedAdditionAttr *A) {
    const auto *SA = cast<SwiftVersionedAdditionAttr>(A); (void)SA;
    OS << " " << SA->getVersion();
    if (SA->getIsReplacedByActive()) OS << " IsReplacedByActive";
  }
  void VisitSwiftVersionedRemovalAttr(const SwiftVersionedRemovalAttr *A) {
    const auto *SA = cast<SwiftVersionedRemovalAttr>(A); (void)SA;
    OS << " " << SA->getVersion();
    OS << " " << SA->getRawKind();
    if (SA->getIsReplacedByActive()) OS << " IsReplacedByActive";
  }
  void VisitTLSModelAttr(const TLSModelAttr *A) {
    const auto *SA = cast<TLSModelAttr>(A); (void)SA;
    OS << " \"" << SA->getModel() << "\"";
  }
  void VisitTargetAttr(const TargetAttr *A) {
    const auto *SA = cast<TargetAttr>(A); (void)SA;
    OS << " \"" << SA->getFeaturesStr() << "\"";
  }
  void VisitTargetClonesAttr(const TargetClonesAttr *A) {
    const auto *SA = cast<TargetClonesAttr>(A); (void)SA;
    for (const auto &Val : SA->featuresStrs())
      OS << " " << Val;
  }
  void VisitTargetVersionAttr(const TargetVersionAttr *A) {
    const auto *SA = cast<TargetVersionAttr>(A); (void)SA;
    OS << " \"" << SA->getNamesStr() << "\"";
  }
  void VisitTestTypestateAttr(const TestTypestateAttr *A) {
    const auto *SA = cast<TestTypestateAttr>(A); (void)SA;
    switch(SA->getTestState()) {
    case TestTypestateAttr::ConsumedState::Consumed:
      OS << " Consumed";
      break;
    case TestTypestateAttr::ConsumedState::Unconsumed:
      OS << " Unconsumed";
      break;
    }
  }
  void VisitTryAcquireCapabilityAttr(const TryAcquireCapabilityAttr *A) {
    const auto *SA = cast<TryAcquireCapabilityAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
  }
  void VisitTypeTagForDatatypeAttr(const TypeTagForDatatypeAttr *A) {
    const auto *SA = cast<TypeTagForDatatypeAttr>(A); (void)SA;
    if (SA->getArgumentKind())
      OS << " " << SA->getArgumentKind()->getName();
    OS << " " << SA->getMatchingCType().getAsString();
    if (SA->getLayoutCompatible()) OS << " LayoutCompatible";
    if (SA->getMustBeNull()) OS << " MustBeNull";
  }
  void VisitTypeVisibilityAttr(const TypeVisibilityAttr *A) {
    const auto *SA = cast<TypeVisibilityAttr>(A); (void)SA;
    switch(SA->getVisibility()) {
    case TypeVisibilityAttr::VisibilityType::Default:
      OS << " Default";
      break;
    case TypeVisibilityAttr::VisibilityType::Hidden:
      OS << " Hidden";
      break;
    case TypeVisibilityAttr::VisibilityType::Protected:
      OS << " Protected";
      break;
    }
  }
  void VisitUnavailableAttr(const UnavailableAttr *A) {
    const auto *SA = cast<UnavailableAttr>(A); (void)SA;
    OS << " \"" << SA->getMessage() << "\"";
    switch(SA->getImplicitReason()) {
    case UnavailableAttr::ImplicitReason::IR_None:
      OS << " IR_None";
      break;
    case UnavailableAttr::ImplicitReason::IR_ARCForbiddenType:
      OS << " IR_ARCForbiddenType";
      break;
    case UnavailableAttr::ImplicitReason::IR_ForbiddenWeak:
      OS << " IR_ForbiddenWeak";
      break;
    case UnavailableAttr::ImplicitReason::IR_ARCForbiddenConversion:
      OS << " IR_ARCForbiddenConversion";
      break;
    case UnavailableAttr::ImplicitReason::IR_ARCInitReturnsUnrelated:
      OS << " IR_ARCInitReturnsUnrelated";
      break;
    case UnavailableAttr::ImplicitReason::IR_ARCFieldWithOwnership:
      OS << " IR_ARCFieldWithOwnership";
      break;
    }
  }
  void VisitUnusedAttr(const UnusedAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitUseHandleAttr(const UseHandleAttr *A) {
    const auto *SA = cast<UseHandleAttr>(A); (void)SA;
    OS << " \"" << SA->getHandleType() << "\"";
  }
  void VisitUuidAttr(const UuidAttr *A) {
    const auto *SA = cast<UuidAttr>(A); (void)SA;
    OS << " \"" << SA->getGuid() << "\"";
    OS << " ";
    dumpBareDeclRef(SA->getGuidDecl());
  }
  void VisitVTablePointerAuthenticationAttr(const VTablePointerAuthenticationAttr *A) {
    const auto *SA = cast<VTablePointerAuthenticationAttr>(A); (void)SA;
    switch(SA->getKey()) {
    case VTablePointerAuthenticationAttr::VPtrAuthKeyType::DefaultKey:
      OS << " DefaultKey";
      break;
    case VTablePointerAuthenticationAttr::VPtrAuthKeyType::NoKey:
      OS << " NoKey";
      break;
    case VTablePointerAuthenticationAttr::VPtrAuthKeyType::ProcessDependent:
      OS << " ProcessDependent";
      break;
    case VTablePointerAuthenticationAttr::VPtrAuthKeyType::ProcessIndependent:
      OS << " ProcessIndependent";
      break;
    }
    switch(SA->getAddressDiscrimination()) {
    case VTablePointerAuthenticationAttr::AddressDiscriminationMode::DefaultAddressDiscrimination:
      OS << " DefaultAddressDiscrimination";
      break;
    case VTablePointerAuthenticationAttr::AddressDiscriminationMode::NoAddressDiscrimination:
      OS << " NoAddressDiscrimination";
      break;
    case VTablePointerAuthenticationAttr::AddressDiscriminationMode::AddressDiscrimination:
      OS << " AddressDiscrimination";
      break;
    }
    switch(SA->getExtraDiscrimination()) {
    case VTablePointerAuthenticationAttr::ExtraDiscrimination::DefaultExtraDiscrimination:
      OS << " DefaultExtraDiscrimination";
      break;
    case VTablePointerAuthenticationAttr::ExtraDiscrimination::NoExtraDiscrimination:
      OS << " NoExtraDiscrimination";
      break;
    case VTablePointerAuthenticationAttr::ExtraDiscrimination::TypeDiscrimination:
      OS << " TypeDiscrimination";
      break;
    case VTablePointerAuthenticationAttr::ExtraDiscrimination::CustomDiscrimination:
      OS << " CustomDiscrimination";
      break;
    }
    OS << " " << SA->getCustomDiscriminationValue();
  }
  void VisitVecTypeHintAttr(const VecTypeHintAttr *A) {
    const auto *SA = cast<VecTypeHintAttr>(A); (void)SA;
    OS << " " << SA->getTypeHint().getAsString();
  }
  void VisitVisibilityAttr(const VisibilityAttr *A) {
    const auto *SA = cast<VisibilityAttr>(A); (void)SA;
    switch(SA->getVisibility()) {
    case VisibilityAttr::VisibilityType::Default:
      OS << " Default";
      break;
    case VisibilityAttr::VisibilityType::Hidden:
      OS << " Hidden";
      break;
    case VisibilityAttr::VisibilityType::Protected:
      OS << " Protected";
      break;
    }
  }
  void VisitWarnUnusedResultAttr(const WarnUnusedResultAttr *A) {
    const auto *SA = cast<WarnUnusedResultAttr>(A); (void)SA;
    OS << " " << A->getSpelling();
    OS << " \"" << SA->getMessage() << "\"";
  }
  void VisitWeakRefAttr(const WeakRefAttr *A) {
    const auto *SA = cast<WeakRefAttr>(A); (void)SA;
    OS << " \"" << SA->getAliasee() << "\"";
  }
  void VisitWebAssemblyExportNameAttr(const WebAssemblyExportNameAttr *A) {
    const auto *SA = cast<WebAssemblyExportNameAttr>(A); (void)SA;
    OS << " \"" << SA->getExportName() << "\"";
  }
  void VisitWebAssemblyImportModuleAttr(const WebAssemblyImportModuleAttr *A) {
    const auto *SA = cast<WebAssemblyImportModuleAttr>(A); (void)SA;
    OS << " \"" << SA->getImportModule() << "\"";
  }
  void VisitWebAssemblyImportNameAttr(const WebAssemblyImportNameAttr *A) {
    const auto *SA = cast<WebAssemblyImportNameAttr>(A); (void)SA;
    OS << " \"" << SA->getImportName() << "\"";
  }
  void VisitXRayInstrumentAttr(const XRayInstrumentAttr *A) {
    OS << " " << A->getSpelling();
  }
  void VisitXRayLogArgsAttr(const XRayLogArgsAttr *A) {
    const auto *SA = cast<XRayLogArgsAttr>(A); (void)SA;
    OS << " " << SA->getArgumentCount();
  }
  void VisitZeroCallUsedRegsAttr(const ZeroCallUsedRegsAttr *A) {
    const auto *SA = cast<ZeroCallUsedRegsAttr>(A); (void)SA;
    switch(SA->getZeroCallUsedRegs()) {
    case ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind::Skip:
      OS << " Skip";
      break;
    case ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind::UsedGPRArg:
      OS << " UsedGPRArg";
      break;
    case ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind::UsedGPR:
      OS << " UsedGPR";
      break;
    case ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind::UsedArg:
      OS << " UsedArg";
      break;
    case ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind::Used:
      OS << " Used";
      break;
    case ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind::AllGPRArg:
      OS << " AllGPRArg";
      break;
    case ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind::AllGPR:
      OS << " AllGPR";
      break;
    case ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind::AllArg:
      OS << " AllArg";
      break;
    case ZeroCallUsedRegsAttr::ZeroCallUsedRegsKind::All:
      OS << " All";
      break;
    }
  }
