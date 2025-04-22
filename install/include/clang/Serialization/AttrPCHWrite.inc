/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Attribute serialization code                                               *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Attr.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

  switch (A->getKind()) {
  case attr::AArch64SVEPcs: {
    const auto *SA = cast<AArch64SVEPcsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::AArch64VectorPcs: {
    const auto *SA = cast<AArch64VectorPcsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::AMDGPUFlatWorkGroupSize: {
    const auto *SA = cast<AMDGPUFlatWorkGroupSizeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getMin());
    Record.AddStmt(SA->getMax());
    break;
  }
  case attr::AMDGPUKernelCall: {
    const auto *SA = cast<AMDGPUKernelCallAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::AMDGPUMaxNumWorkGroups: {
    const auto *SA = cast<AMDGPUMaxNumWorkGroupsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getMaxNumWorkGroupsX());
    Record.AddStmt(SA->getMaxNumWorkGroupsY());
    Record.AddStmt(SA->getMaxNumWorkGroupsZ());
    break;
  }
  case attr::AMDGPUNumSGPR: {
    const auto *SA = cast<AMDGPUNumSGPRAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getNumSGPR());
    break;
  }
  case attr::AMDGPUNumVGPR: {
    const auto *SA = cast<AMDGPUNumVGPRAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getNumVGPR());
    break;
  }
  case attr::AMDGPUWavesPerEU: {
    const auto *SA = cast<AMDGPUWavesPerEUAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getMin());
    Record.AddStmt(SA->getMax());
    break;
  }
  case attr::ARMInterrupt: {
    const auto *SA = cast<ARMInterruptAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getInterrupt()));
    break;
  }
  case attr::ARMInterruptSaveFP: {
    const auto *SA = cast<ARMInterruptSaveFPAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getInterrupt()));
    break;
  }
  case attr::ARMSaveFP: {
    const auto *SA = cast<ARMSaveFPAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::AVRInterrupt: {
    const auto *SA = cast<AVRInterruptAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::AVRSignal: {
    const auto *SA = cast<AVRSignalAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::AbiTag: {
    const auto *SA = cast<AbiTagAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->tags_size());
    for (auto &Val : SA->tags())
      Record.AddString(Val);
    break;
  }
  case attr::AcquireCapability: {
    const auto *SA = cast<AcquireCapabilityAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->args_size());
    for (auto &Val : SA->args())
      Record.AddStmt(Val);
    break;
  }
  case attr::AcquireHandle: {
    const auto *SA = cast<AcquireHandleAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getHandleType());
    break;
  }
  case attr::AcquiredAfter: {
    const auto *SA = cast<AcquiredAfterAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->args_size());
    for (auto &Val : SA->args())
      Record.AddStmt(Val);
    break;
  }
  case attr::AcquiredBefore: {
    const auto *SA = cast<AcquiredBeforeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->args_size());
    for (auto &Val : SA->args())
      Record.AddStmt(Val);
    break;
  }
  case attr::AddressSpace: {
    const auto *SA = cast<AddressSpaceAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getAddressSpace());
    break;
  }
  case attr::Alias: {
    const auto *SA = cast<AliasAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getAliasee());
    break;
  }
  case attr::AlignMac68k: {
    const auto *SA = cast<AlignMac68kAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::AlignNatural: {
    const auto *SA = cast<AlignNaturalAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::AlignValue: {
    const auto *SA = cast<AlignValueAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getAlignment());
    break;
  }
  case attr::Aligned: {
    const auto *SA = cast<AlignedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->isAlignmentExpr());
    if (SA->isAlignmentExpr())
      Record.AddStmt(SA->getAlignmentExpr());
    else
      Record.AddTypeSourceInfo(SA->getAlignmentType());
    break;
  }
  case attr::AllocAlign: {
    const auto *SA = cast<AllocAlignAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getParamIndex().serialize());
    break;
  }
  case attr::AllocSize: {
    const auto *SA = cast<AllocSizeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getElemSizeParam().serialize());
    Record.push_back(SA->getNumElemsParam().serialize());
    break;
  }
  case attr::Allocating: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::AlwaysDestroy: {
    const auto *SA = cast<AlwaysDestroyAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::AlwaysInline: {
    const auto *SA = cast<AlwaysInlineAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::AnalyzerNoReturn: {
    const auto *SA = cast<AnalyzerNoReturnAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Annotate: {
    const auto *SA = cast<AnnotateAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->delayedArgs_size());
    for (auto &Val : SA->delayedArgs())
      Record.AddStmt(Val);
    Record.AddString(SA->getAnnotation());
    Record.push_back(SA->args_size());
    for (auto &Val : SA->args())
      Record.AddStmt(Val);
    break;
  }
  case attr::AnnotateType: {
    const auto *SA = cast<AnnotateTypeAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->delayedArgs_size());
    for (auto &Val : SA->delayedArgs())
      Record.AddStmt(Val);
    Record.AddString(SA->getAnnotation());
    Record.push_back(SA->args_size());
    for (auto &Val : SA->args())
      Record.AddStmt(Val);
    break;
  }
  case attr::AnyX86Interrupt: {
    const auto *SA = cast<AnyX86InterruptAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::AnyX86NoCallerSavedRegisters: {
    const auto *SA = cast<AnyX86NoCallerSavedRegistersAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::AnyX86NoCfCheck: {
    const auto *SA = cast<AnyX86NoCfCheckAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ArcWeakrefUnavailable: {
    const auto *SA = cast<ArcWeakrefUnavailableAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ArgumentWithTypeTag: {
    const auto *SA = cast<ArgumentWithTypeTagAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getArgumentKind());
    Record.push_back(SA->getArgumentIdx().serialize());
    Record.push_back(SA->getTypeTagIdx().serialize());
    Record.push_back(SA->getIsPointer());
    break;
  }
  case attr::ArmAgnostic: {
    const auto *SA = cast<ArmAgnosticAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->agnosticArgs_size());
    for (auto &Val : SA->agnosticArgs())
      Record.AddString(Val);
    break;
  }
  case attr::ArmBuiltinAlias: {
    const auto *SA = cast<ArmBuiltinAliasAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getBuiltinName());
    break;
  }
  case attr::ArmIn: {
    const auto *SA = cast<ArmInAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->inArgs_size());
    for (auto &Val : SA->inArgs())
      Record.AddString(Val);
    break;
  }
  case attr::ArmInOut: {
    const auto *SA = cast<ArmInOutAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->inOutArgs_size());
    for (auto &Val : SA->inOutArgs())
      Record.AddString(Val);
    break;
  }
  case attr::ArmLocallyStreaming: {
    const auto *SA = cast<ArmLocallyStreamingAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ArmMveStrictPolymorphism: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ArmNew: {
    const auto *SA = cast<ArmNewAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->newArgs_size());
    for (auto &Val : SA->newArgs())
      Record.AddString(Val);
    break;
  }
  case attr::ArmOut: {
    const auto *SA = cast<ArmOutAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->outArgs_size());
    for (auto &Val : SA->outArgs())
      Record.AddString(Val);
    break;
  }
  case attr::ArmPreserves: {
    const auto *SA = cast<ArmPreservesAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->preserveArgs_size());
    for (auto &Val : SA->preserveArgs())
      Record.AddString(Val);
    break;
  }
  case attr::ArmStreaming: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ArmStreamingCompatible: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Artificial: {
    const auto *SA = cast<ArtificialAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::AsmLabel: {
    const auto *SA = cast<AsmLabelAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getLabel());
    Record.push_back(SA->getIsLiteralLabel());
    break;
  }
  case attr::AssertCapability: {
    const auto *SA = cast<AssertCapabilityAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->args_size());
    for (auto &Val : SA->args())
      Record.AddStmt(Val);
    break;
  }
  case attr::AssumeAligned: {
    const auto *SA = cast<AssumeAlignedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getAlignment());
    Record.AddStmt(SA->getOffset());
    break;
  }
  case attr::Atomic: {
    const auto *SA = cast<AtomicAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->atomicOptions_size());
    for (AtomicAttr::atomicOptions_iterator i = SA->atomicOptions_begin(), e = SA->atomicOptions_end(); i != e; ++i)
      Record.push_back((*i));
    break;
  }
  case attr::Availability: {
    const auto *SA = cast<AvailabilityAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getPlatform());
    Record.AddVersionTuple(SA->getIntroduced());
    Record.AddVersionTuple(SA->getDeprecated());
    Record.AddVersionTuple(SA->getObsoleted());
    Record.push_back(SA->getUnavailable());
    Record.AddString(SA->getMessage());
    Record.push_back(SA->getStrict());
    Record.AddString(SA->getReplacement());
    Record.push_back(SA->getPriority());
    Record.AddIdentifierRef(SA->getEnvironment());
    break;
  }
  case attr::AvailableOnlyInDefaultEvalMethod: {
    const auto *SA = cast<AvailableOnlyInDefaultEvalMethodAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::BPFFastCall: {
    const auto *SA = cast<BPFFastCallAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::BPFPreserveAccessIndex: {
    const auto *SA = cast<BPFPreserveAccessIndexAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::BPFPreserveStaticOffset: {
    const auto *SA = cast<BPFPreserveStaticOffsetAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::BTFDeclTag: {
    const auto *SA = cast<BTFDeclTagAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getBTFDeclTag());
    break;
  }
  case attr::BTFTypeTag: {
    const auto *SA = cast<BTFTypeTagAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getBTFTypeTag());
    break;
  }
  case attr::Blocking: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Blocks: {
    const auto *SA = cast<BlocksAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getType()));
    break;
  }
  case attr::Builtin: {
    const auto *SA = cast<BuiltinAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getID());
    break;
  }
  case attr::BuiltinAlias: {
    const auto *SA = cast<BuiltinAliasAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getBuiltinName());
    break;
  }
  case attr::C11NoReturn: {
    const auto *SA = cast<C11NoReturnAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CDecl: {
    const auto *SA = cast<CDeclAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CFAuditedTransfer: {
    const auto *SA = cast<CFAuditedTransferAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CFConsumed: {
    const auto *SA = cast<CFConsumedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CFGuard: {
    const auto *SA = cast<CFGuardAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getGuard()));
    break;
  }
  case attr::CFICanonicalJumpTable: {
    const auto *SA = cast<CFICanonicalJumpTableAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CFReturnsNotRetained: {
    const auto *SA = cast<CFReturnsNotRetainedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CFReturnsRetained: {
    const auto *SA = cast<CFReturnsRetainedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CFUnknownTransfer: {
    const auto *SA = cast<CFUnknownTransferAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CPUDispatch: {
    const auto *SA = cast<CPUDispatchAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->cpus_size());
    for (auto &Val : SA->cpus())
      Record.AddIdentifierRef(Val);
    break;
  }
  case attr::CPUSpecific: {
    const auto *SA = cast<CPUSpecificAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->cpus_size());
    for (auto &Val : SA->cpus())
      Record.AddIdentifierRef(Val);
    break;
  }
  case attr::CUDAConstant: {
    const auto *SA = cast<CUDAConstantAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CUDADevice: {
    const auto *SA = cast<CUDADeviceAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CUDADeviceBuiltinSurfaceType: {
    const auto *SA = cast<CUDADeviceBuiltinSurfaceTypeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CUDADeviceBuiltinTextureType: {
    const auto *SA = cast<CUDADeviceBuiltinTextureTypeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CUDAGlobal: {
    const auto *SA = cast<CUDAGlobalAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CUDAGridConstant: {
    const auto *SA = cast<CUDAGridConstantAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CUDAHost: {
    const auto *SA = cast<CUDAHostAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CUDAInvalidTarget: {
    const auto *SA = cast<CUDAInvalidTargetAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CUDALaunchBounds: {
    const auto *SA = cast<CUDALaunchBoundsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getMaxThreads());
    Record.AddStmt(SA->getMinBlocks());
    Record.AddStmt(SA->getMaxBlocks());
    break;
  }
  case attr::CUDAShared: {
    const auto *SA = cast<CUDASharedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CXX11NoReturn: {
    const auto *SA = cast<CXX11NoReturnAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CXXAssume: {
    const auto *SA = cast<CXXAssumeAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getAssumption());
    break;
  }
  case attr::CallableWhen: {
    const auto *SA = cast<CallableWhenAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->callableStates_size());
    for (CallableWhenAttr::callableStates_iterator i = SA->callableStates_begin(), e = SA->callableStates_end(); i != e; ++i)
      Record.push_back((*i));
    break;
  }
  case attr::Callback: {
    const auto *SA = cast<CallbackAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->encoding_size());
    for (auto &Val : SA->encoding())
      Record.push_back(Val);
    break;
  }
  case attr::CalledOnce: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Capability: {
    const auto *SA = cast<CapabilityAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getName());
    break;
  }
  case attr::CapturedRecord: {
    const auto *SA = cast<CapturedRecordAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CarriesDependency: {
    const auto *SA = cast<CarriesDependencyAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Cleanup: {
    const auto *SA = cast<CleanupAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddDeclRef(SA->getFunctionDecl());
    break;
  }
  case attr::ClspvLibclcBuiltin: {
    const auto *SA = cast<ClspvLibclcBuiltinAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CmseNSCall: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CmseNSEntry: {
    const auto *SA = cast<CmseNSEntryAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CodeAlign: {
    const auto *SA = cast<CodeAlignAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getAlignment());
    break;
  }
  case attr::CodeModel: {
    const auto *SA = cast<CodeModelAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getModel()));
    break;
  }
  case attr::CodeSeg: {
    const auto *SA = cast<CodeSegAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getName());
    break;
  }
  case attr::Cold: {
    const auto *SA = cast<ColdAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Common: {
    const auto *SA = cast<CommonAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Const: {
    const auto *SA = cast<ConstAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ConstInit: {
    const auto *SA = cast<ConstInitAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Constructor: {
    const auto *SA = cast<ConstructorAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getPriority());
    break;
  }
  case attr::Consumable: {
    const auto *SA = cast<ConsumableAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getDefaultState()));
    break;
  }
  case attr::ConsumableAutoCast: {
    const auto *SA = cast<ConsumableAutoCastAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ConsumableSetOnRead: {
    const auto *SA = cast<ConsumableSetOnReadAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Convergent: {
    const auto *SA = cast<ConvergentAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CoroAwaitElidable: {
    const auto *SA = cast<CoroAwaitElidableAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CoroAwaitElidableArgument: {
    const auto *SA = cast<CoroAwaitElidableArgumentAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CoroDisableLifetimeBound: {
    const auto *SA = cast<CoroDisableLifetimeBoundAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CoroLifetimeBound: {
    const auto *SA = cast<CoroLifetimeBoundAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CoroOnlyDestroyWhenComplete: {
    const auto *SA = cast<CoroOnlyDestroyWhenCompleteAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CoroReturnType: {
    const auto *SA = cast<CoroReturnTypeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CoroWrapper: {
    const auto *SA = cast<CoroWrapperAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::CountedBy: {
    const auto *SA = cast<CountedByAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getCount());
    Record.push_back(SA->getNestedLevel());
    break;
  }
  case attr::CountedByOrNull: {
    const auto *SA = cast<CountedByOrNullAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getCount());
    Record.push_back(SA->getNestedLevel());
    break;
  }
  case attr::DLLExport: {
    const auto *SA = cast<DLLExportAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::DLLExportStaticLocal: {
    const auto *SA = cast<DLLExportStaticLocalAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::DLLImport: {
    const auto *SA = cast<DLLImportAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::DLLImportStaticLocal: {
    const auto *SA = cast<DLLImportStaticLocalAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Deprecated: {
    const auto *SA = cast<DeprecatedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getMessage());
    Record.AddString(SA->getReplacement());
    break;
  }
  case attr::Destructor: {
    const auto *SA = cast<DestructorAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getPriority());
    break;
  }
  case attr::DiagnoseAsBuiltin: {
    const auto *SA = cast<DiagnoseAsBuiltinAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddDeclRef(SA->getFunction());
    Record.push_back(SA->argIndices_size());
    for (auto &Val : SA->argIndices())
      Record.push_back(Val);
    break;
  }
  case attr::DiagnoseIf: {
    const auto *SA = cast<DiagnoseIfAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getCond());
    Record.AddString(SA->getMessage());
Record.push_back(static_cast<uint64_t>(SA->getDefaultSeverity()));
    Record.AddString(SA->getWarningGroup());
    Record.push_back(SA->getArgDependent());
    Record.AddDeclRef(SA->getParent());
    break;
  }
  case attr::DisableSanitizerInstrumentation: {
    const auto *SA = cast<DisableSanitizerInstrumentationAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::DisableTailCalls: {
    const auto *SA = cast<DisableTailCallsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::EmptyBases: {
    const auto *SA = cast<EmptyBasesAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::EnableIf: {
    const auto *SA = cast<EnableIfAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getCond());
    Record.AddString(SA->getMessage());
    break;
  }
  case attr::EnforceTCB: {
    const auto *SA = cast<EnforceTCBAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getTCBName());
    break;
  }
  case attr::EnforceTCBLeaf: {
    const auto *SA = cast<EnforceTCBLeafAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getTCBName());
    break;
  }
  case attr::EnumExtensibility: {
    const auto *SA = cast<EnumExtensibilityAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getExtensibility()));
    break;
  }
  case attr::Error: {
    const auto *SA = cast<ErrorAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getUserDiagnostic());
    break;
  }
  case attr::ExcludeFromExplicitInstantiation: {
    const auto *SA = cast<ExcludeFromExplicitInstantiationAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ExplicitInit: {
    const auto *SA = cast<ExplicitInitAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ExtVectorType: {
    const auto *SA = cast<ExtVectorTypeAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getNumElements());
    break;
  }
  case attr::ExternalSourceSymbol: {
    const auto *SA = cast<ExternalSourceSymbolAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getLanguage());
    Record.AddString(SA->getDefinedIn());
    Record.push_back(SA->getGeneratedDeclaration());
    Record.AddString(SA->getUSR());
    break;
  }
  case attr::FallThrough: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::FastCall: {
    const auto *SA = cast<FastCallAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Final: {
    const auto *SA = cast<FinalAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::FlagEnum: {
    const auto *SA = cast<FlagEnumAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Flatten: {
    const auto *SA = cast<FlattenAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Format: {
    const auto *SA = cast<FormatAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getType());
    Record.push_back(SA->getFormatIdx());
    Record.push_back(SA->getFirstArg());
    break;
  }
  case attr::FormatArg: {
    const auto *SA = cast<FormatArgAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getFormatIdx().serialize());
    break;
  }
  case attr::FormatMatches: {
    const auto *SA = cast<FormatMatchesAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getType());
    Record.push_back(SA->getFormatIdx());
    Record.AddStmt(SA->getExpectedFormat());
    break;
  }
  case attr::FunctionReturnThunks: {
    const auto *SA = cast<FunctionReturnThunksAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getThunkType()));
    break;
  }
  case attr::GNUInline: {
    const auto *SA = cast<GNUInlineAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::GuardedBy: {
    const auto *SA = cast<GuardedByAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getArg());
    break;
  }
  case attr::GuardedVar: {
    const auto *SA = cast<GuardedVarAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::HIPManaged: {
    const auto *SA = cast<HIPManagedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::HLSLContainedType: {
    const auto *SA = cast<HLSLContainedTypeAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddTypeSourceInfo(SA->getTypeLoc());
    break;
  }
  case attr::HLSLControlFlowHint: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::HLSLGroupSharedAddressSpace: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::HLSLLoopHint: {
    const auto *SA = cast<HLSLLoopHintAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getDirective());
    break;
  }
  case attr::HLSLNumThreads: {
    const auto *SA = cast<HLSLNumThreadsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getX());
    Record.push_back(SA->getY());
    Record.push_back(SA->getZ());
    break;
  }
  case attr::HLSLPackOffset: {
    const auto *SA = cast<HLSLPackOffsetAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getSubcomponent());
    Record.push_back(SA->getComponent());
    break;
  }
  case attr::HLSLParamModifier: {
    const auto *SA = cast<HLSLParamModifierAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getMergedSpelling());
    break;
  }
  case attr::HLSLROV: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::HLSLRawBuffer: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::HLSLResourceBinding: {
    const auto *SA = cast<HLSLResourceBindingAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getSlot());
    Record.AddString(SA->getSpace());
    break;
  }
  case attr::HLSLResourceClass: {
    const auto *SA = cast<HLSLResourceClassAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getResourceClass()));
    break;
  }
  case attr::HLSLSV_DispatchThreadID: {
    const auto *SA = cast<HLSLSV_DispatchThreadIDAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::HLSLSV_GroupID: {
    const auto *SA = cast<HLSLSV_GroupIDAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::HLSLSV_GroupIndex: {
    const auto *SA = cast<HLSLSV_GroupIndexAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::HLSLSV_GroupThreadID: {
    const auto *SA = cast<HLSLSV_GroupThreadIDAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::HLSLShader: {
    const auto *SA = cast<HLSLShaderAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getType()));
    break;
  }
  case attr::HLSLWaveSize: {
    const auto *SA = cast<HLSLWaveSizeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getMin());
    Record.push_back(SA->getMax());
    Record.push_back(SA->getPreferred());
    break;
  }
  case attr::Hot: {
    const auto *SA = cast<HotAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::HybridPatchable: {
    const auto *SA = cast<HybridPatchableAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::IBAction: {
    const auto *SA = cast<IBActionAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::IBOutlet: {
    const auto *SA = cast<IBOutletAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::IBOutletCollection: {
    const auto *SA = cast<IBOutletCollectionAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddTypeSourceInfo(SA->getInterfaceLoc());
    break;
  }
  case attr::IFunc: {
    const auto *SA = cast<IFuncAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getResolver());
    break;
  }
  case attr::InitPriority: {
    const auto *SA = cast<InitPriorityAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getPriority());
    break;
  }
  case attr::InitSeg: {
    const auto *SA = cast<InitSegAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getSection());
    break;
  }
  case attr::IntelOclBicc: {
    const auto *SA = cast<IntelOclBiccAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::InternalLinkage: {
    const auto *SA = cast<InternalLinkageAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::LTOVisibilityPublic: {
    const auto *SA = cast<LTOVisibilityPublicAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::LayoutVersion: {
    const auto *SA = cast<LayoutVersionAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getVersion());
    break;
  }
  case attr::Leaf: {
    const auto *SA = cast<LeafAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::LifetimeBound: {
    const auto *SA = cast<LifetimeBoundAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::LifetimeCaptureBy: {
    const auto *SA = cast<LifetimeCaptureByAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->params_size());
    for (auto &Val : SA->params())
      Record.push_back(Val);
    break;
  }
  case attr::Likely: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::LoaderUninitialized: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::LockReturned: {
    const auto *SA = cast<LockReturnedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getArg());
    break;
  }
  case attr::LocksExcluded: {
    const auto *SA = cast<LocksExcludedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->args_size());
    for (auto &Val : SA->args())
      Record.AddStmt(Val);
    break;
  }
  case attr::LoopHint: {
    const auto *SA = cast<LoopHintAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getOption()));
Record.push_back(static_cast<uint64_t>(SA->getState()));
    Record.AddStmt(SA->getValue());
    break;
  }
  case attr::M68kInterrupt: {
    const auto *SA = cast<M68kInterruptAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getNumber());
    break;
  }
  case attr::M68kRTD: {
    const auto *SA = cast<M68kRTDAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::MIGServerRoutine: {
    const auto *SA = cast<MIGServerRoutineAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::MSABI: {
    const auto *SA = cast<MSABIAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::MSAllocator: {
    const auto *SA = cast<MSAllocatorAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::MSConstexpr: {
    const auto *SA = cast<MSConstexprAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::MSInheritance: {
    const auto *SA = cast<MSInheritanceAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getBestCase());
    break;
  }
  case attr::MSNoVTable: {
    const auto *SA = cast<MSNoVTableAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::MSP430Interrupt: {
    const auto *SA = cast<MSP430InterruptAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getNumber());
    break;
  }
  case attr::MSStruct: {
    const auto *SA = cast<MSStructAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::MSVtorDisp: {
    const auto *SA = cast<MSVtorDispAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getVdm());
    break;
  }
  case attr::MaxFieldAlignment: {
    const auto *SA = cast<MaxFieldAlignmentAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getAlignment());
    break;
  }
  case attr::MayAlias: {
    const auto *SA = cast<MayAliasAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::MaybeUndef: {
    const auto *SA = cast<MaybeUndefAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::MicroMips: {
    const auto *SA = cast<MicroMipsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::MinSize: {
    const auto *SA = cast<MinSizeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::MinVectorWidth: {
    const auto *SA = cast<MinVectorWidthAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getVectorWidth());
    break;
  }
  case attr::Mips16: {
    const auto *SA = cast<Mips16Attr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::MipsInterrupt: {
    const auto *SA = cast<MipsInterruptAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getInterrupt()));
    break;
  }
  case attr::MipsLongCall: {
    const auto *SA = cast<MipsLongCallAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::MipsShortCall: {
    const auto *SA = cast<MipsShortCallAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Mode: {
    const auto *SA = cast<ModeAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getMode());
    break;
  }
  case attr::MustTail: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NSConsumed: {
    const auto *SA = cast<NSConsumedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NSConsumesSelf: {
    const auto *SA = cast<NSConsumesSelfAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NSErrorDomain: {
    const auto *SA = cast<NSErrorDomainAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getErrorDomain());
    break;
  }
  case attr::NSReturnsAutoreleased: {
    const auto *SA = cast<NSReturnsAutoreleasedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NSReturnsNotRetained: {
    const auto *SA = cast<NSReturnsNotRetainedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NSReturnsRetained: {
    const auto *SA = cast<NSReturnsRetainedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NVPTXKernel: {
    const auto *SA = cast<NVPTXKernelAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Naked: {
    const auto *SA = cast<NakedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoAlias: {
    const auto *SA = cast<NoAliasAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoBuiltin: {
    const auto *SA = cast<NoBuiltinAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->builtinNames_size());
    for (auto &Val : SA->builtinNames())
      Record.AddString(Val);
    break;
  }
  case attr::NoCommon: {
    const auto *SA = cast<NoCommonAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoConvergent: {
    const auto *SA = cast<NoConvergentAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoDebug: {
    const auto *SA = cast<NoDebugAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoDeref: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoDestroy: {
    const auto *SA = cast<NoDestroyAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoDuplicate: {
    const auto *SA = cast<NoDuplicateAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoEscape: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoInline: {
    const auto *SA = cast<NoInlineAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoInstrumentFunction: {
    const auto *SA = cast<NoInstrumentFunctionAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoMerge: {
    const auto *SA = cast<NoMergeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoMicroMips: {
    const auto *SA = cast<NoMicroMipsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoMips16: {
    const auto *SA = cast<NoMips16Attr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoProfileFunction: {
    const auto *SA = cast<NoProfileFunctionAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoRandomizeLayout: {
    const auto *SA = cast<NoRandomizeLayoutAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoReturn: {
    const auto *SA = cast<NoReturnAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoSanitize: {
    const auto *SA = cast<NoSanitizeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->sanitizers_size());
    for (auto &Val : SA->sanitizers())
      Record.AddString(Val);
    break;
  }
  case attr::NoSpecializations: {
    const auto *SA = cast<NoSpecializationsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getMessage());
    break;
  }
  case attr::NoSpeculativeLoadHardening: {
    const auto *SA = cast<NoSpeculativeLoadHardeningAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoSplitStack: {
    const auto *SA = cast<NoSplitStackAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoStackProtector: {
    const auto *SA = cast<NoStackProtectorAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoThreadSafetyAnalysis: {
    const auto *SA = cast<NoThreadSafetyAnalysisAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoThrow: {
    const auto *SA = cast<NoThrowAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoTrivialAutoVarInit: {
    const auto *SA = cast<NoTrivialAutoVarInitAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoUniqueAddress: {
    const auto *SA = cast<NoUniqueAddressAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NoUwtable: {
    const auto *SA = cast<NoUwtableAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::NonAllocating: {
    const auto *SA = cast<NonAllocatingAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getCond());
    break;
  }
  case attr::NonBlocking: {
    const auto *SA = cast<NonBlockingAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getCond());
    break;
  }
  case attr::NonNull: {
    const auto *SA = cast<NonNullAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->args_size());
    for (auto &Val : SA->args())
      Record.push_back(Val.serialize());
    break;
  }
  case attr::NotTailCalled: {
    const auto *SA = cast<NotTailCalledAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OMPAllocateDecl: {
    const auto *SA = cast<OMPAllocateDeclAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getAllocatorType()));
    Record.AddStmt(SA->getAllocator());
    Record.AddStmt(SA->getAlignment());
    break;
  }
  case attr::OMPAssume: {
    const auto *SA = cast<OMPAssumeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getAssumption());
    break;
  }
  case attr::OMPCaptureKind: {
    const auto *SA = cast<OMPCaptureKindAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getCaptureKindVal());
    break;
  }
  case attr::OMPCaptureNoInit: {
    const auto *SA = cast<OMPCaptureNoInitAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OMPDeclareSimdDecl: {
    const auto *SA = cast<OMPDeclareSimdDeclAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getBranchState()));
    Record.AddStmt(SA->getSimdlen());
    Record.push_back(SA->uniforms_size());
    for (auto &Val : SA->uniforms())
      Record.AddStmt(Val);
    Record.push_back(SA->aligneds_size());
    for (auto &Val : SA->aligneds())
      Record.AddStmt(Val);
    Record.push_back(SA->alignments_size());
    for (auto &Val : SA->alignments())
      Record.AddStmt(Val);
    Record.push_back(SA->linears_size());
    for (auto &Val : SA->linears())
      Record.AddStmt(Val);
    Record.push_back(SA->modifiers_size());
    for (auto &Val : SA->modifiers())
      Record.push_back(Val);
    Record.push_back(SA->steps_size());
    for (auto &Val : SA->steps())
      Record.AddStmt(Val);
    break;
  }
  case attr::OMPDeclareTargetDecl: {
    const auto *SA = cast<OMPDeclareTargetDeclAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getMapType()));
Record.push_back(static_cast<uint64_t>(SA->getDevType()));
    Record.AddStmt(SA->getIndirectExpr());
    Record.push_back(SA->getIndirect());
    Record.push_back(SA->getLevel());
    break;
  }
  case attr::OMPDeclareVariant: {
    const auto *SA = cast<OMPDeclareVariantAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getVariantFuncRef());
    Record.writeOMPTraitInfo(SA->getTraitInfos());
    Record.push_back(SA->adjustArgsNothing_size());
    for (auto &Val : SA->adjustArgsNothing())
      Record.AddStmt(Val);
    Record.push_back(SA->adjustArgsNeedDevicePtr_size());
    for (auto &Val : SA->adjustArgsNeedDevicePtr())
      Record.AddStmt(Val);
    Record.push_back(SA->appendArgs_size());
    for (OMPDeclareVariantAttr::appendArgs_iterator I = SA->appendArgs_begin(), E = SA->appendArgs_end(); I != E; ++I) {
      Record.writeBool(I->IsTarget);
      Record.writeBool(I->IsTargetSync);
    }
    break;
  }
  case attr::OMPReferencedVar: {
    const auto *SA = cast<OMPReferencedVarAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getRef());
    break;
  }
  case attr::OMPThreadPrivateDecl: {
    const auto *SA = cast<OMPThreadPrivateDeclAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OSConsumed: {
    const auto *SA = cast<OSConsumedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OSConsumesThis: {
    const auto *SA = cast<OSConsumesThisAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OSReturnsNotRetained: {
    const auto *SA = cast<OSReturnsNotRetainedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OSReturnsRetained: {
    const auto *SA = cast<OSReturnsRetainedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OSReturnsRetainedOnNonZero: {
    const auto *SA = cast<OSReturnsRetainedOnNonZeroAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OSReturnsRetainedOnZero: {
    const auto *SA = cast<OSReturnsRetainedOnZeroAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCBoxable: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCBridge: {
    const auto *SA = cast<ObjCBridgeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getBridgedType());
    break;
  }
  case attr::ObjCBridgeMutable: {
    const auto *SA = cast<ObjCBridgeMutableAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getBridgedType());
    break;
  }
  case attr::ObjCBridgeRelated: {
    const auto *SA = cast<ObjCBridgeRelatedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getRelatedClass());
    Record.AddIdentifierRef(SA->getClassMethod());
    Record.AddIdentifierRef(SA->getInstanceMethod());
    break;
  }
  case attr::ObjCClassStub: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCDesignatedInitializer: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCDirect: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCDirectMembers: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCException: {
    const auto *SA = cast<ObjCExceptionAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCExplicitProtocolImpl: {
    const auto *SA = cast<ObjCExplicitProtocolImplAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCExternallyRetained: {
    const auto *SA = cast<ObjCExternallyRetainedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCGC: {
    const auto *SA = cast<ObjCGCAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getKind());
    break;
  }
  case attr::ObjCIndependentClass: {
    const auto *SA = cast<ObjCIndependentClassAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCInertUnsafeUnretained: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCKindOf: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCMethodFamily: {
    const auto *SA = cast<ObjCMethodFamilyAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getFamily()));
    break;
  }
  case attr::ObjCNSObject: {
    const auto *SA = cast<ObjCNSObjectAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCNonLazyClass: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCNonRuntimeProtocol: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCOwnership: {
    const auto *SA = cast<ObjCOwnershipAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getKind());
    break;
  }
  case attr::ObjCPreciseLifetime: {
    const auto *SA = cast<ObjCPreciseLifetimeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCRequiresPropertyDefs: {
    const auto *SA = cast<ObjCRequiresPropertyDefsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCRequiresSuper: {
    const auto *SA = cast<ObjCRequiresSuperAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCReturnsInnerPointer: {
    const auto *SA = cast<ObjCReturnsInnerPointerAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCRootClass: {
    const auto *SA = cast<ObjCRootClassAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCRuntimeName: {
    const auto *SA = cast<ObjCRuntimeNameAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getMetadataName());
    break;
  }
  case attr::ObjCRuntimeVisible: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ObjCSubclassingRestricted: {
    const auto *SA = cast<ObjCSubclassingRestrictedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OpenACCRoutineAnnot: {
    const auto *SA = cast<OpenACCRoutineAnnotAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OpenACCRoutineDecl: {
    const auto *SA = cast<OpenACCRoutineDeclAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddOpenACCRoutineDeclAttr(SA);
    break;
  }
  case attr::OpenCLAccess: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OpenCLConstantAddressSpace: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OpenCLGenericAddressSpace: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OpenCLGlobalAddressSpace: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OpenCLGlobalDeviceAddressSpace: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OpenCLGlobalHostAddressSpace: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OpenCLIntelReqdSubGroupSize: {
    const auto *SA = cast<OpenCLIntelReqdSubGroupSizeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getSubGroupSize());
    break;
  }
  case attr::OpenCLKernel: {
    const auto *SA = cast<OpenCLKernelAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OpenCLLocalAddressSpace: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OpenCLPrivateAddressSpace: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::OpenCLUnrollHint: {
    const auto *SA = cast<OpenCLUnrollHintAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getUnrollHint());
    break;
  }
  case attr::OptimizeNone: {
    const auto *SA = cast<OptimizeNoneAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Overloadable: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Override: {
    const auto *SA = cast<OverrideAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Owner: {
    const auto *SA = cast<OwnerAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddTypeSourceInfo(SA->getDerefTypeLoc());
    break;
  }
  case attr::Ownership: {
    const auto *SA = cast<OwnershipAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getModule());
    Record.push_back(SA->args_size());
    for (auto &Val : SA->args())
      Record.push_back(Val.serialize());
    break;
  }
  case attr::Packed: {
    const auto *SA = cast<PackedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ParamTypestate: {
    const auto *SA = cast<ParamTypestateAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getParamState()));
    break;
  }
  case attr::Pascal: {
    const auto *SA = cast<PascalAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::PassObjectSize: {
    const auto *SA = cast<PassObjectSizeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getType());
    break;
  }
  case attr::PatchableFunctionEntry: {
    const auto *SA = cast<PatchableFunctionEntryAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getCount());
    Record.push_back(SA->getOffset());
    Record.AddString(SA->getSection());
    break;
  }
  case attr::Pcs: {
    const auto *SA = cast<PcsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getPCS()));
    break;
  }
  case attr::Pointer: {
    const auto *SA = cast<PointerAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddTypeSourceInfo(SA->getDerefTypeLoc());
    break;
  }
  case attr::PointerAuth: {
    const auto *SA = cast<PointerAuthAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getKey());
    Record.push_back(SA->getAddressDiscriminated());
    Record.push_back(SA->getExtraDiscriminator());
    break;
  }
  case attr::PragmaClangBSSSection: {
    const auto *SA = cast<PragmaClangBSSSectionAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getName());
    break;
  }
  case attr::PragmaClangDataSection: {
    const auto *SA = cast<PragmaClangDataSectionAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getName());
    break;
  }
  case attr::PragmaClangRelroSection: {
    const auto *SA = cast<PragmaClangRelroSectionAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getName());
    break;
  }
  case attr::PragmaClangRodataSection: {
    const auto *SA = cast<PragmaClangRodataSectionAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getName());
    break;
  }
  case attr::PragmaClangTextSection: {
    const auto *SA = cast<PragmaClangTextSectionAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getName());
    break;
  }
  case attr::PreferredName: {
    const auto *SA = cast<PreferredNameAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddTypeSourceInfo(SA->getTypedefTypeLoc());
    break;
  }
  case attr::PreferredType: {
    const auto *SA = cast<PreferredTypeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddTypeSourceInfo(SA->getTypeLoc());
    break;
  }
  case attr::PreserveAll: {
    const auto *SA = cast<PreserveAllAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::PreserveMost: {
    const auto *SA = cast<PreserveMostAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::PreserveNone: {
    const auto *SA = cast<PreserveNoneAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::PtGuardedBy: {
    const auto *SA = cast<PtGuardedByAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getArg());
    break;
  }
  case attr::PtGuardedVar: {
    const auto *SA = cast<PtGuardedVarAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Ptr32: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Ptr64: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Pure: {
    const auto *SA = cast<PureAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::RISCVInterrupt: {
    const auto *SA = cast<RISCVInterruptAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getInterrupt()));
    break;
  }
  case attr::RISCVVLSCC: {
    const auto *SA = cast<RISCVVLSCCAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getVectorWidth());
    break;
  }
  case attr::RISCVVectorCC: {
    const auto *SA = cast<RISCVVectorCCAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::RandomizeLayout: {
    const auto *SA = cast<RandomizeLayoutAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ReadOnlyPlacement: {
    const auto *SA = cast<ReadOnlyPlacementAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::RegCall: {
    const auto *SA = cast<RegCallAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Reinitializes: {
    const auto *SA = cast<ReinitializesAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ReleaseCapability: {
    const auto *SA = cast<ReleaseCapabilityAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->args_size());
    for (auto &Val : SA->args())
      Record.AddStmt(Val);
    break;
  }
  case attr::ReleaseHandle: {
    const auto *SA = cast<ReleaseHandleAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getHandleType());
    break;
  }
  case attr::ReqdWorkGroupSize: {
    const auto *SA = cast<ReqdWorkGroupSizeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getXDim());
    Record.AddStmt(SA->getYDim());
    Record.AddStmt(SA->getZDim());
    break;
  }
  case attr::RequiresCapability: {
    const auto *SA = cast<RequiresCapabilityAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->args_size());
    for (auto &Val : SA->args())
      Record.AddStmt(Val);
    break;
  }
  case attr::Restrict: {
    const auto *SA = cast<RestrictAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getDeallocator());
    Record.push_back(SA->getDeallocatorPtrArgIndex().serialize());
    break;
  }
  case attr::Retain: {
    const auto *SA = cast<RetainAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ReturnTypestate: {
    const auto *SA = cast<ReturnTypestateAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getState()));
    break;
  }
  case attr::ReturnsNonNull: {
    const auto *SA = cast<ReturnsNonNullAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ReturnsTwice: {
    const auto *SA = cast<ReturnsTwiceAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::SPtr: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::SYCLKernel: {
    const auto *SA = cast<SYCLKernelAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::SYCLKernelEntryPoint: {
    const auto *SA = cast<SYCLKernelEntryPointAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddTypeSourceInfo(SA->getKernelNameLoc());
    Record.push_back(SA->getInvalid());
    break;
  }
  case attr::SYCLSpecialClass: {
    const auto *SA = cast<SYCLSpecialClassAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::ScopedLockable: {
    const auto *SA = cast<ScopedLockableAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Section: {
    const auto *SA = cast<SectionAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getName());
    break;
  }
  case attr::SelectAny: {
    const auto *SA = cast<SelectAnyAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Sentinel: {
    const auto *SA = cast<SentinelAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getSentinel());
    Record.push_back(SA->getNullPos());
    break;
  }
  case attr::SetTypestate: {
    const auto *SA = cast<SetTypestateAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getNewState()));
    break;
  }
  case attr::SizedBy: {
    const auto *SA = cast<SizedByAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getSize());
    Record.push_back(SA->getNestedLevel());
    break;
  }
  case attr::SizedByOrNull: {
    const auto *SA = cast<SizedByOrNullAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getSize());
    Record.push_back(SA->getNestedLevel());
    break;
  }
  case attr::SpeculativeLoadHardening: {
    const auto *SA = cast<SpeculativeLoadHardeningAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::StandaloneDebug: {
    const auto *SA = cast<StandaloneDebugAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::StdCall: {
    const auto *SA = cast<StdCallAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::StrictFP: {
    const auto *SA = cast<StrictFPAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::StrictGuardStackCheck: {
    const auto *SA = cast<StrictGuardStackCheckAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Suppress: {
    const auto *SA = cast<SuppressAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->diagnosticIdentifiers_size());
    for (auto &Val : SA->diagnosticIdentifiers())
      Record.AddString(Val);
    break;
  }
  case attr::SwiftAsync: {
    const auto *SA = cast<SwiftAsyncAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getKind()));
    Record.push_back(SA->getCompletionHandlerIndex().serialize());
    break;
  }
  case attr::SwiftAsyncCall: {
    const auto *SA = cast<SwiftAsyncCallAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::SwiftAsyncContext: {
    const auto *SA = cast<SwiftAsyncContextAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::SwiftAsyncError: {
    const auto *SA = cast<SwiftAsyncErrorAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getConvention()));
    Record.push_back(SA->getHandlerParamIdx());
    break;
  }
  case attr::SwiftAsyncName: {
    const auto *SA = cast<SwiftAsyncNameAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getName());
    break;
  }
  case attr::SwiftAttr: {
    const auto *SA = cast<SwiftAttrAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getAttribute());
    break;
  }
  case attr::SwiftBridge: {
    const auto *SA = cast<SwiftBridgeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getSwiftType());
    break;
  }
  case attr::SwiftBridgedTypedef: {
    const auto *SA = cast<SwiftBridgedTypedefAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::SwiftCall: {
    const auto *SA = cast<SwiftCallAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::SwiftContext: {
    const auto *SA = cast<SwiftContextAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::SwiftError: {
    const auto *SA = cast<SwiftErrorAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getConvention()));
    break;
  }
  case attr::SwiftErrorResult: {
    const auto *SA = cast<SwiftErrorResultAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::SwiftImportAsNonGeneric: {
    const auto *SA = cast<SwiftImportAsNonGenericAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::SwiftImportPropertyAsAccessors: {
    const auto *SA = cast<SwiftImportPropertyAsAccessorsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::SwiftIndirectResult: {
    const auto *SA = cast<SwiftIndirectResultAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::SwiftName: {
    const auto *SA = cast<SwiftNameAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getName());
    break;
  }
  case attr::SwiftNewType: {
    const auto *SA = cast<SwiftNewTypeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getNewtypeKind()));
    break;
  }
  case attr::SwiftObjCMembers: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::SwiftPrivate: {
    const auto *SA = cast<SwiftPrivateAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::SwiftVersionedAddition: {
    const auto *SA = cast<SwiftVersionedAdditionAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddVersionTuple(SA->getVersion());
    AddAttr(SA->getAdditionalAttr());    Record.push_back(SA->getIsReplacedByActive());
    break;
  }
  case attr::SwiftVersionedRemoval: {
    const auto *SA = cast<SwiftVersionedRemovalAttr>(A);
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddVersionTuple(SA->getVersion());
    Record.push_back(SA->getRawKind());
    Record.push_back(SA->getIsReplacedByActive());
    break;
  }
  case attr::SysVABI: {
    const auto *SA = cast<SysVABIAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::TLSModel: {
    const auto *SA = cast<TLSModelAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getModel());
    break;
  }
  case attr::Target: {
    const auto *SA = cast<TargetAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getFeaturesStr());
    break;
  }
  case attr::TargetClones: {
    const auto *SA = cast<TargetClonesAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->featuresStrs_size());
    for (auto &Val : SA->featuresStrs())
      Record.AddString(Val);
    break;
  }
  case attr::TargetVersion: {
    const auto *SA = cast<TargetVersionAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getNamesStr());
    break;
  }
  case attr::TestTypestate: {
    const auto *SA = cast<TestTypestateAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getTestState()));
    break;
  }
  case attr::ThisCall: {
    const auto *SA = cast<ThisCallAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Thread: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::TransparentUnion: {
    const auto *SA = cast<TransparentUnionAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::TrivialABI: {
    const auto *SA = cast<TrivialABIAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::TryAcquireCapability: {
    const auto *SA = cast<TryAcquireCapabilityAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getSuccessValue());
    Record.push_back(SA->args_size());
    for (auto &Val : SA->args())
      Record.AddStmt(Val);
    break;
  }
  case attr::TypeNonNull: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::TypeNullUnspecified: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::TypeNullable: {
    const auto *SA = cast<TypeNullableAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::TypeNullableResult: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::TypeTagForDatatype: {
    const auto *SA = cast<TypeTagForDatatypeAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddIdentifierRef(SA->getArgumentKind());
    Record.AddTypeSourceInfo(SA->getMatchingCTypeLoc());
    Record.push_back(SA->getLayoutCompatible());
    Record.push_back(SA->getMustBeNull());
    break;
  }
  case attr::TypeVisibility: {
    const auto *SA = cast<TypeVisibilityAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getVisibility()));
    break;
  }
  case attr::UPtr: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Unavailable: {
    const auto *SA = cast<UnavailableAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getMessage());
Record.push_back(static_cast<uint64_t>(SA->getImplicitReason()));
    break;
  }
  case attr::Uninitialized: {
    const auto *SA = cast<UninitializedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Unlikely: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::UnsafeBufferUsage: {
    const auto *SA = cast<UnsafeBufferUsageAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Unused: {
    const auto *SA = cast<UnusedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::UseHandle: {
    const auto *SA = cast<UseHandleAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getHandleType());
    break;
  }
  case attr::Used: {
    const auto *SA = cast<UsedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::UsingIfExists: {
    const auto *SA = cast<UsingIfExistsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Uuid: {
    const auto *SA = cast<UuidAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getGuid());
    Record.AddDeclRef(SA->getGuidDecl());
    break;
  }
  case attr::VTablePointerAuthentication: {
    const auto *SA = cast<VTablePointerAuthenticationAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getKey()));
Record.push_back(static_cast<uint64_t>(SA->getAddressDiscrimination()));
Record.push_back(static_cast<uint64_t>(SA->getExtraDiscrimination()));
    Record.push_back(SA->getCustomDiscriminationValue());
    break;
  }
  case attr::VecReturn: {
    const auto *SA = cast<VecReturnAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::VecTypeHint: {
    const auto *SA = cast<VecTypeHintAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddTypeSourceInfo(SA->getTypeHintLoc());
    break;
  }
  case attr::VectorCall: {
    const auto *SA = cast<VectorCallAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::Visibility: {
    const auto *SA = cast<VisibilityAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getVisibility()));
    break;
  }
  case attr::WarnUnused: {
    const auto *SA = cast<WarnUnusedAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::WarnUnusedResult: {
    const auto *SA = cast<WarnUnusedResultAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getMessage());
    break;
  }
  case attr::Weak: {
    const auto *SA = cast<WeakAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::WeakImport: {
    const auto *SA = cast<WeakImportAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::WeakRef: {
    const auto *SA = cast<WeakRefAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getAliasee());
    break;
  }
  case attr::WebAssemblyExportName: {
    const auto *SA = cast<WebAssemblyExportNameAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getExportName());
    break;
  }
  case attr::WebAssemblyFuncref: {
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::WebAssemblyImportModule: {
    const auto *SA = cast<WebAssemblyImportModuleAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getImportModule());
    break;
  }
  case attr::WebAssemblyImportName: {
    const auto *SA = cast<WebAssemblyImportNameAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddString(SA->getImportName());
    break;
  }
  case attr::WorkGroupSizeHint: {
    const auto *SA = cast<WorkGroupSizeHintAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.AddStmt(SA->getXDim());
    Record.AddStmt(SA->getYDim());
    Record.AddStmt(SA->getZDim());
    break;
  }
  case attr::X86ForceAlignArgPointer: {
    const auto *SA = cast<X86ForceAlignArgPointerAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::XRayInstrument: {
    const auto *SA = cast<XRayInstrumentAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    break;
  }
  case attr::XRayLogArgs: {
    const auto *SA = cast<XRayLogArgsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
    Record.push_back(SA->getArgumentCount());
    break;
  }
  case attr::ZeroCallUsedRegs: {
    const auto *SA = cast<ZeroCallUsedRegsAttr>(A);
    Record.push_back(SA->isInherited());
    Record.push_back(A->isImplicit());
    Record.push_back(A->isPackExpansion());
Record.push_back(static_cast<uint64_t>(SA->getZeroCallUsedRegs()));
    break;
  }
  }
