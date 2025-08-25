//===---- MissingFeatures.h - Checks for unimplemented features -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file introduces some helper classes to guard against features that
// CIR dialect supports that we do not have and also do not have great ways to
// assert against.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_MISSINGFEATURES_H
#define CLANG_CIR_MISSINGFEATURES_H

namespace cir {

// As a way to track features that haven't yet been implemented this class
// explicitly contains a list of static fns that will return false that you
// can guard against. If and when a feature becomes implemented simply changing
// this return to true will cause compilation to fail at all the points in which
// we noted that we needed to address. This is a much more explicit way to
// handle "TODO"s.
struct MissingFeatures {
  // Address space related
  static bool addressSpace() { return false; }

  // Unhandled global/linkage information.
  static bool opGlobalThreadLocal() { return false; }
  static bool opGlobalConstant() { return false; }
  static bool opGlobalWeakRef() { return false; }
  static bool opGlobalUnnamedAddr() { return false; }
  static bool opGlobalSection() { return false; }
  static bool opGlobalVisibility() { return false; }
  static bool opGlobalDLLImportExport() { return false; }
  static bool opGlobalPartition() { return false; }
  static bool opGlobalUsedOrCompilerUsed() { return false; }

  static bool supportIFuncAttr() { return false; }
  static bool supportVisibility() { return false; }
  static bool hiddenVisibility() { return false; }
  static bool protectedVisibility() { return false; }
  static bool defaultVisibility() { return false; }

  // Load/store attributes
  static bool opLoadStoreThreadLocal() { return false; }
  static bool opLoadEmitScalarRangeCheck() { return false; }
  static bool opLoadBooleanRepresentation() { return false; }
  static bool opLoadStoreTbaa() { return false; }
  static bool opLoadStoreVolatile() { return false; }
  static bool opLoadStoreAtomic() { return false; }
  static bool opLoadStoreObjC() { return false; }

  // AllocaOp handling
  static bool opAllocaStaticLocal() { return false; }
  static bool opAllocaNonGC() { return false; }
  static bool opAllocaImpreciseLifetime() { return false; }
  static bool opAllocaPreciseLifetime() { return false; }
  static bool opAllocaTLS() { return false; }
  static bool opAllocaOpenMPThreadPrivate() { return false; }
  static bool opAllocaEscapeByReference() { return false; }
  static bool opAllocaReference() { return false; }
  static bool opAllocaAnnotations() { return false; }
  static bool opAllocaDynAllocSize() { return false; }
  static bool opAllocaCaptureByInit() { return false; }

  // FuncOp handling
  static bool opFuncOpenCLKernelMetadata() { return false; }
  static bool opFuncAstDeclAttr() { return false; }
  static bool opFuncAttributesForDefinition() { return false; }
  static bool opFuncCallingConv() { return false; }
  static bool opFuncCPUAndFeaturesAttributes() { return false; }
  static bool opFuncExceptions() { return false; }
  static bool opFuncExtraAttrs() { return false; }
  static bool opFuncMaybeHandleStaticInExternC() { return false; }
  static bool opFuncMultipleReturnVals() { return false; }
  static bool opFuncOperandBundles() { return false; }
  static bool opFuncParameterAttributes() { return false; }
  static bool opFuncSection() { return false; }
  static bool setLLVMFunctionFEnvAttributes() { return false; }
  static bool setFunctionAttributes() { return false; }

  // CallOp handling
  static bool opCallAggregateArgs() { return false; }
  static bool opCallPaddingArgs() { return false; }
  static bool opCallABIExtendArg() { return false; }
  static bool opCallABIIndirectArg() { return false; }
  static bool opCallWidenArg() { return false; }
  static bool opCallBitcastArg() { return false; }
  static bool opCallImplicitObjectSizeArgs() { return false; }
  static bool opCallReturn() { return false; }
  static bool opCallArgEvaluationOrder() { return false; }
  static bool opCallCallConv() { return false; }
  static bool opCallMustTail() { return false; }
  static bool opCallInAlloca() { return false; }
  static bool opCallAttrs() { return false; }
  static bool opCallSurroundingTry() { return false; }
  static bool opCallASTAttr() { return false; }
  static bool opCallObjCMethod() { return false; }
  static bool opCallExtParameterInfo() { return false; }
  static bool opCallCIRGenFuncInfoParamInfo() { return false; }
  static bool opCallCIRGenFuncInfoExtParamInfo() { return false; }
  static bool opCallLandingPad() { return false; }
  static bool opCallContinueBlock() { return false; }
  static bool opCallChain() { return false; }

  // CXXNewExpr
  static bool exprNewNullCheck() { return false; }

  // FnInfoOpts -- This is used to track whether calls are chain calls or
  // instance methods. Classic codegen uses chain call to track and extra free
  // register for x86 and uses instance method as a condition for a thunk
  // generation special case. It's not clear that we need either of these in
  // pre-lowering CIR codegen.
  static bool opCallFnInfoOpts() { return false; }

  // ScopeOp handling
  static bool opScopeCleanupRegion() { return false; }

  // Unary operator handling
  static bool opUnaryPromotionType() { return false; }

  // SwitchOp handling
  static bool foldRangeCase() { return false; }

  // Clang early optimizations or things defered to LLVM lowering.
  static bool mayHaveIntegerOverflow() { return false; }
  static bool shouldReverseUnaryCondOnBoolExpr() { return false; }

  // RecordType
  static bool skippedLayout() { return false; }
  static bool astRecordDeclAttr() { return false; }
  static bool cxxSupport() { return false; }
  static bool recordZeroInit() { return false; }
  static bool zeroSizeRecordMembers() { return false; }
  static bool recordLayoutVirtualBases() { return false; }

  // Various handling of deferred processing in CIRGenModule.
  static bool cgmRelease() { return false; }
  static bool deferredVtables() { return false; }
  static bool deferredFuncDecls() { return false; }

  // CXXABI
  static bool cxxABI() { return false; }
  static bool cxxabiThisAlignment() { return false; }
  static bool cxxabiUseARMMethodPtrABI() { return false; }
  static bool cxxabiUseARMGuardVarABI() { return false; }
  static bool cxxabiAppleARM64CXXABI() { return false; }
  static bool cxxabiStructorImplicitParam() { return false; }
  static bool isDiscreteBitFieldABI() { return false; }

  // Address class
  static bool addressOffset() { return false; }
  static bool addressIsKnownNonNull() { return false; }
  static bool addressPointerAuthInfo() { return false; }

  // Atomic
  static bool atomicExpr() { return false; }
  static bool atomicInfo() { return false; }
  static bool atomicInfoGetAtomicPointer() { return false; }
  static bool atomicInfoGetAtomicAddress() { return false; }
  static bool atomicUseLibCall() { return false; }
  static bool atomicScope() { return false; }
  static bool atomicSyncScopeID() { return false; }

  // Misc
  static bool abiArgInfo() { return false; }
  static bool addHeapAllocSiteMetadata() { return false; }
  static bool aggValueSlot() { return false; }
  static bool aggValueSlotAlias() { return false; }
  static bool aggValueSlotDestructedFlag() { return false; }
  static bool aggValueSlotGC() { return false; }
  static bool aggValueSlotMayOverlap() { return false; }
  static bool aggValueSlotVolatile() { return false; }
  static bool alignCXXRecordDecl() { return false; }
  static bool armComputeVolatileBitfields() { return false; }
  static bool asmGoto() { return false; }
  static bool asmInputOperands() { return false; }
  static bool asmLabelAttr() { return false; }
  static bool asmMemoryEffects() { return false; }
  static bool asmOutputOperands() { return false; }
  static bool asmUnwindClobber() { return false; }
  static bool assignMemcpyizer() { return false; }
  static bool astVarDeclInterface() { return false; }
  static bool attributeBuiltin() { return false; }
  static bool attributeNoBuiltin() { return false; }
  static bool bitfields() { return false; }
  static bool builtinCall() { return false; }
  static bool builtinCallF128() { return false; }
  static bool builtinCallMathErrno() { return false; }
  static bool builtinCheckKind() { return false; }
  static bool cgFPOptionsRAII() { return false; }
  static bool cirgenABIInfo() { return false; }
  static bool cleanupAfterErrorDiags() { return false; }
  static bool cleanupsToDeactivate() { return false; }
  static bool constEmitterArrayILE() { return false; }
  static bool constEmitterVectorILE() { return false; }
  static bool constantFoldSwitchStatement() { return false; }
  static bool constructABIArgDirectExtend() { return false; }
  static bool coverageMapping() { return false; }
  static bool createInvariantGroup() { return false; }
  static bool createProfileWeightsForLoop() { return false; }
  static bool ctorMemcpyizer() { return false; }
  static bool cudaSupport() { return false; }
  static bool cxxRecordStaticMembers() { return false; }
  static bool dataLayoutTypeIsSized() { return false; }
  static bool dataLayoutTypeAllocSize() { return false; }
  static bool dataLayoutTypeStoreSize() { return false; }
  static bool deferredCXXGlobalInit() { return false; }
  static bool devirtualizeMemberFunction() { return false; }
  static bool ehCleanupFlags() { return false; }
  static bool ehCleanupScope() { return false; }
  static bool ehCleanupScopeRequiresEHCleanup() { return false; }
  static bool ehCleanupBranchFixups() { return false; }
  static bool ehstackBranches() { return false; }
  static bool emitCheckedInBoundsGEP() { return false; }
  static bool emitCondLikelihoodViaExpectIntrinsic() { return false; }
  static bool emitLifetimeMarkers() { return false; }
  static bool emitLValueAlignmentAssumption() { return false; }
  static bool emitNullabilityCheck() { return false; }
  static bool emitTypeCheck() { return false; }
  static bool emitTypeMetadataCodeForVCall() { return false; }
  static bool fastMathFlags() { return false; }
  static bool fpConstraints() { return false; }
  static bool generateDebugInfo() { return false; }
  static bool globalViewIndices() { return false; }
  static bool globalViewIntLowering() { return false; }
  static bool hip() { return false; }
  static bool implicitConstructorArgs() { return false; }
  static bool incrementProfileCounter() { return false; }
  static bool innermostEHScope() { return false; }
  static bool insertBuiltinUnpredictable() { return false; }
  static bool instrumentation() { return false; }
  static bool intrinsics() { return false; }
  static bool isMemcpyEquivalentSpecialMember() { return false; }
  static bool isTrivialCtorOrDtor() { return false; }
  static bool lambdaCaptures() { return false; }
  static bool lambdaFieldToName() { return false; }
  static bool loopInfoStack() { return false; }
  static bool lowerAggregateLoadStore() { return false; }
  static bool lowerModeOptLevel() { return false; }
  static bool maybeHandleStaticInExternC() { return false; }
  static bool mergeAllConstants() { return false; }
  static bool metaDataNode() { return false; }
  static bool moduleNameHash() { return false; }
  static bool msabi() { return false; }
  static bool needsGlobalCtorDtor() { return false; }
  static bool objCBlocks() { return false; }
  static bool objCGC() { return false; }
  static bool objCLifetime() { return false; }
  static bool openCL() { return false; }
  static bool openMP() { return false; }
  static bool opTBAA() { return false; }
  static bool peepholeProtection() { return false; }
  static bool pgoUse() { return false; }
  static bool pointerOverflowSanitizer() { return false; }
  static bool preservedAccessIndexRegion() { return false; }
  static bool requiresCleanups() { return false; }
  static bool runCleanupsScope() { return false; }
  static bool sanitizers() { return false; }
  static bool setDLLStorageClass() { return false; }
  static bool setNonGC() { return false; }
  static bool setObjCGCLValueClass() { return false; }
  static bool setTargetAttributes() { return false; }
  static bool sourceLanguageCases() { return false; }
  static bool stackBase() { return false; }
  static bool stackSaveOp() { return false; }
  static bool targetCIRGenInfoArch() { return false; }
  static bool targetCIRGenInfoOS() { return false; }
  static bool targetCodeGenInfoGetNullPointer() { return false; }
  static bool thunks() { return false; }
  static bool tryEmitAsConstant() { return false; }
  static bool typeChecks() { return false; }
  static bool weakRefReference() { return false; }
  static bool writebacks() { return false; }
  static bool appleKext() { return false; }
  static bool dtorCleanups() { return false; }
  static bool vtableInitialization() { return false; }
  static bool vtableEmitMetadata() { return false; }
  static bool vtableRelativeLayout() { return false; }
  static bool msvcBuiltins() { return false; }
  static bool vaArgABILowering() { return false; }
  static bool vlas() { return false; }

  // Missing types
  static bool dataMemberType() { return false; }
  static bool matrixType() { return false; }
  static bool methodType() { return false; }
  static bool scalableVectors() { return false; }
  static bool unsizedTypes() { return false; }
  static bool vectorType() { return false; }
  static bool complexType() { return false; }
  static bool fixedPointType() { return false; }
  static bool stringTypeWithDifferentArraySize() { return false; }

  // Future CIR operations
  static bool awaitOp() { return false; }
  static bool callOp() { return false; }
  static bool complexImagOp() { return false; }
  static bool complexRealOp() { return false; }
  static bool ifOp() { return false; }
  static bool invokeOp() { return false; }
  static bool labelOp() { return false; }
  static bool ptrDiffOp() { return false; }
  static bool ptrStrideOp() { return false; }
  static bool switchOp() { return false; }
  static bool throwOp() { return false; }
  static bool tryOp() { return false; }
  static bool vecTernaryOp() { return false; }
  static bool zextOp() { return false; }

  // Future CIR attributes
  static bool optInfoAttr() { return false; }
};

} // namespace cir

#endif // CLANG_CIR_MISSINGFEATURES_H
