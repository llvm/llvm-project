/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Attribute text node traverser                                              *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Attr.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

  void VisitAMDGPUFlatWorkGroupSizeAttr(const AMDGPUFlatWorkGroupSizeAttr *A) {
    const auto *SA = cast<AMDGPUFlatWorkGroupSizeAttr>(A); (void)SA;
    Visit(SA->getMin());
    Visit(SA->getMax());
  }
  void VisitAMDGPUMaxNumWorkGroupsAttr(const AMDGPUMaxNumWorkGroupsAttr *A) {
    const auto *SA = cast<AMDGPUMaxNumWorkGroupsAttr>(A); (void)SA;
    Visit(SA->getMaxNumWorkGroupsX());
    Visit(SA->getMaxNumWorkGroupsY());
    Visit(SA->getMaxNumWorkGroupsZ());
  }
  void VisitAMDGPUWavesPerEUAttr(const AMDGPUWavesPerEUAttr *A) {
    const auto *SA = cast<AMDGPUWavesPerEUAttr>(A); (void)SA;
    Visit(SA->getMin());
    Visit(SA->getMax());
  }
  void VisitAcquireCapabilityAttr(const AcquireCapabilityAttr *A) {
    const auto *SA = cast<AcquireCapabilityAttr>(A); (void)SA;
    for (AcquireCapabilityAttr::args_iterator I = SA->args_begin(), E = SA->args_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitAcquiredAfterAttr(const AcquiredAfterAttr *A) {
    const auto *SA = cast<AcquiredAfterAttr>(A); (void)SA;
    for (AcquiredAfterAttr::args_iterator I = SA->args_begin(), E = SA->args_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitAcquiredBeforeAttr(const AcquiredBeforeAttr *A) {
    const auto *SA = cast<AcquiredBeforeAttr>(A); (void)SA;
    for (AcquiredBeforeAttr::args_iterator I = SA->args_begin(), E = SA->args_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitAlignValueAttr(const AlignValueAttr *A) {
    const auto *SA = cast<AlignValueAttr>(A); (void)SA;
    Visit(SA->getAlignment());
  }
  void VisitAlignedAttr(const AlignedAttr *A) {
    const auto *SA = cast<AlignedAttr>(A); (void)SA;
    if (SA->isAlignmentExpr())
      Visit(SA->getAlignmentExpr());
  }
  void VisitAnnotateAttr(const AnnotateAttr *A) {
    const auto *SA = cast<AnnotateAttr>(A); (void)SA;
    for (AnnotateAttr::args_iterator I = SA->args_begin(), E = SA->args_end(); I != E; ++I)
      Visit(*I);
    for (AnnotateAttr::delayedArgs_iterator I = SA->delayedArgs_begin(), E = SA->delayedArgs_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitAnnotateTypeAttr(const AnnotateTypeAttr *A) {
    const auto *SA = cast<AnnotateTypeAttr>(A); (void)SA;
    for (AnnotateTypeAttr::args_iterator I = SA->args_begin(), E = SA->args_end(); I != E; ++I)
      Visit(*I);
    for (AnnotateTypeAttr::delayedArgs_iterator I = SA->delayedArgs_begin(), E = SA->delayedArgs_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitAssertCapabilityAttr(const AssertCapabilityAttr *A) {
    const auto *SA = cast<AssertCapabilityAttr>(A); (void)SA;
    for (AssertCapabilityAttr::args_iterator I = SA->args_begin(), E = SA->args_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitAssumeAlignedAttr(const AssumeAlignedAttr *A) {
    const auto *SA = cast<AssumeAlignedAttr>(A); (void)SA;
    Visit(SA->getAlignment());
    Visit(SA->getOffset());
  }
  void VisitCUDALaunchBoundsAttr(const CUDALaunchBoundsAttr *A) {
    const auto *SA = cast<CUDALaunchBoundsAttr>(A); (void)SA;
    Visit(SA->getMaxThreads());
    Visit(SA->getMinBlocks());
    Visit(SA->getMaxBlocks());
  }
  void VisitCXXAssumeAttr(const CXXAssumeAttr *A) {
    const auto *SA = cast<CXXAssumeAttr>(A); (void)SA;
    Visit(SA->getAssumption());
  }
  void VisitCodeAlignAttr(const CodeAlignAttr *A) {
    const auto *SA = cast<CodeAlignAttr>(A); (void)SA;
    Visit(SA->getAlignment());
  }
  void VisitCountedByAttr(const CountedByAttr *A) {
    const auto *SA = cast<CountedByAttr>(A); (void)SA;
    Visit(SA->getCount());
  }
  void VisitCountedByOrNullAttr(const CountedByOrNullAttr *A) {
    const auto *SA = cast<CountedByOrNullAttr>(A); (void)SA;
    Visit(SA->getCount());
  }
  void VisitDiagnoseIfAttr(const DiagnoseIfAttr *A) {
    const auto *SA = cast<DiagnoseIfAttr>(A); (void)SA;
    Visit(SA->getCond());
  }
  void VisitEnableIfAttr(const EnableIfAttr *A) {
    const auto *SA = cast<EnableIfAttr>(A); (void)SA;
    Visit(SA->getCond());
  }
  void VisitExtVectorTypeAttr(const ExtVectorTypeAttr *A) {
    const auto *SA = cast<ExtVectorTypeAttr>(A); (void)SA;
    Visit(SA->getNumElements());
  }
  void VisitFormatMatchesAttr(const FormatMatchesAttr *A) {
    const auto *SA = cast<FormatMatchesAttr>(A); (void)SA;
    Visit(SA->getExpectedFormat());
  }
  void VisitGuardedByAttr(const GuardedByAttr *A) {
    const auto *SA = cast<GuardedByAttr>(A); (void)SA;
    Visit(SA->getArg());
  }
  void VisitLockReturnedAttr(const LockReturnedAttr *A) {
    const auto *SA = cast<LockReturnedAttr>(A); (void)SA;
    Visit(SA->getArg());
  }
  void VisitLocksExcludedAttr(const LocksExcludedAttr *A) {
    const auto *SA = cast<LocksExcludedAttr>(A); (void)SA;
    for (LocksExcludedAttr::args_iterator I = SA->args_begin(), E = SA->args_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitLoopHintAttr(const LoopHintAttr *A) {
    const auto *SA = cast<LoopHintAttr>(A); (void)SA;
    Visit(SA->getValue());
  }
  void VisitNonAllocatingAttr(const NonAllocatingAttr *A) {
    const auto *SA = cast<NonAllocatingAttr>(A); (void)SA;
    Visit(SA->getCond());
  }
  void VisitNonBlockingAttr(const NonBlockingAttr *A) {
    const auto *SA = cast<NonBlockingAttr>(A); (void)SA;
    Visit(SA->getCond());
  }
  void VisitOMPAllocateDeclAttr(const OMPAllocateDeclAttr *A) {
    const auto *SA = cast<OMPAllocateDeclAttr>(A); (void)SA;
    Visit(SA->getAllocator());
    Visit(SA->getAlignment());
  }
  void VisitOMPDeclareSimdDeclAttr(const OMPDeclareSimdDeclAttr *A) {
    const auto *SA = cast<OMPDeclareSimdDeclAttr>(A); (void)SA;
    Visit(SA->getSimdlen());
    for (OMPDeclareSimdDeclAttr::uniforms_iterator I = SA->uniforms_begin(), E = SA->uniforms_end(); I != E; ++I)
      Visit(*I);
    for (OMPDeclareSimdDeclAttr::aligneds_iterator I = SA->aligneds_begin(), E = SA->aligneds_end(); I != E; ++I)
      Visit(*I);
    for (OMPDeclareSimdDeclAttr::alignments_iterator I = SA->alignments_begin(), E = SA->alignments_end(); I != E; ++I)
      Visit(*I);
    for (OMPDeclareSimdDeclAttr::linears_iterator I = SA->linears_begin(), E = SA->linears_end(); I != E; ++I)
      Visit(*I);
    for (OMPDeclareSimdDeclAttr::steps_iterator I = SA->steps_begin(), E = SA->steps_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitOMPDeclareTargetDeclAttr(const OMPDeclareTargetDeclAttr *A) {
    const auto *SA = cast<OMPDeclareTargetDeclAttr>(A); (void)SA;
    Visit(SA->getIndirectExpr());
  }
  void VisitOMPDeclareVariantAttr(const OMPDeclareVariantAttr *A) {
    const auto *SA = cast<OMPDeclareVariantAttr>(A); (void)SA;
    Visit(SA->getVariantFuncRef());
    for (OMPDeclareVariantAttr::adjustArgsNothing_iterator I = SA->adjustArgsNothing_begin(), E = SA->adjustArgsNothing_end(); I != E; ++I)
      Visit(*I);
    for (OMPDeclareVariantAttr::adjustArgsNeedDevicePtr_iterator I = SA->adjustArgsNeedDevicePtr_begin(), E = SA->adjustArgsNeedDevicePtr_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitOMPReferencedVarAttr(const OMPReferencedVarAttr *A) {
    const auto *SA = cast<OMPReferencedVarAttr>(A); (void)SA;
    Visit(SA->getRef());
  }
  void VisitPtGuardedByAttr(const PtGuardedByAttr *A) {
    const auto *SA = cast<PtGuardedByAttr>(A); (void)SA;
    Visit(SA->getArg());
  }
  void VisitReleaseCapabilityAttr(const ReleaseCapabilityAttr *A) {
    const auto *SA = cast<ReleaseCapabilityAttr>(A); (void)SA;
    for (ReleaseCapabilityAttr::args_iterator I = SA->args_begin(), E = SA->args_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitReqdWorkGroupSizeAttr(const ReqdWorkGroupSizeAttr *A) {
    const auto *SA = cast<ReqdWorkGroupSizeAttr>(A); (void)SA;
    Visit(SA->getXDim());
    Visit(SA->getYDim());
    Visit(SA->getZDim());
  }
  void VisitRequiresCapabilityAttr(const RequiresCapabilityAttr *A) {
    const auto *SA = cast<RequiresCapabilityAttr>(A); (void)SA;
    for (RequiresCapabilityAttr::args_iterator I = SA->args_begin(), E = SA->args_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitRestrictAttr(const RestrictAttr *A) {
    const auto *SA = cast<RestrictAttr>(A); (void)SA;
    Visit(SA->getDeallocator());
  }
  void VisitSizedByAttr(const SizedByAttr *A) {
    const auto *SA = cast<SizedByAttr>(A); (void)SA;
    Visit(SA->getSize());
  }
  void VisitSizedByOrNullAttr(const SizedByOrNullAttr *A) {
    const auto *SA = cast<SizedByOrNullAttr>(A); (void)SA;
    Visit(SA->getSize());
  }
  void VisitSwiftVersionedAdditionAttr(const SwiftVersionedAdditionAttr *A) {
    const auto *SA = cast<SwiftVersionedAdditionAttr>(A); (void)SA;
    Visit(SA->getAdditionalAttr());
  }
  void VisitTryAcquireCapabilityAttr(const TryAcquireCapabilityAttr *A) {
    const auto *SA = cast<TryAcquireCapabilityAttr>(A); (void)SA;
    Visit(SA->getSuccessValue());
    for (TryAcquireCapabilityAttr::args_iterator I = SA->args_begin(), E = SA->args_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitWorkGroupSizeHintAttr(const WorkGroupSizeHintAttr *A) {
    const auto *SA = cast<WorkGroupSizeHintAttr>(A); (void)SA;
    Visit(SA->getXDim());
    Visit(SA->getYDim());
    Visit(SA->getZDim());
  }
