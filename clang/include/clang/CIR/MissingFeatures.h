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

struct MissingFeatures {
  // TODO(CIR): Implement the CIRGenFunction::buildTypeCheck method that handles
  // sanitizer related type check features
  static bool buildTypeCheck() { return false; }
  static bool tbaa() { return false; }
  static bool cleanups() { return false; }
  static bool emitNullabilityCheck() { return false; }

  // GNU vectors are done, but other kinds of vectors haven't been implemented.
  static bool scalableVectors() { return false; }
  static bool vectorConstants() { return false; }

  // Address space related
  static bool addressSpace() { return false; }
  static bool addressSpaceInGlobalVar() { return false; }

  // Clang codegen options
  static bool strictVTablePointers() { return false; }

  // Unhandled global/linkage information.
  static bool unnamedAddr() { return false; }
  static bool setComdat() { return false; }
  static bool setDSOLocal() { return false; }
  static bool threadLocal() { return false; }
  static bool setDLLStorageClass() { return false; }
  static bool setDLLImportDLLExport() { return false; }
  static bool setPartition() { return false; }
  static bool setGlobalVisibility() { return false; }
  static bool hiddenVisibility() { return false; }
  static bool protectedVisibility() { return false; }
  static bool addCompilerUsedGlobal() { return false; }

  // Sanitizers
  static bool reportGlobalToASan() { return false; }
  static bool emitAsanPrologueOrEpilogue() { return false; }
  static bool emitCheckedInBoundsGEP() { return false; }
  static bool pointerOverflowSanitizer() { return false; }
  static bool sanitizeDtor() { return false; }
  static bool sanitizeVLABound() { return false; }
  static bool sanitizerBuiltin() { return false; }
  static bool sanitizerReturn() { return false; }

  // ObjC
  static bool setObjCGCLValueClass() { return false; }
  static bool objCLifetime() { return false; }

  // Debug info
  static bool generateDebugInfo() { return false; }

  // LLVM Attributes
  static bool setFunctionAttributes() { return false; }
  static bool attributeBuiltin() { return false; }
  static bool attributeNoBuiltin() { return false; }
  static bool parameterAttributes() { return false; }
  static bool minLegalVectorWidthAttr() { return false; }
  static bool vscaleRangeAttr() { return false; }

  // Coroutines
  static bool unhandledException() { return false; }

  // Missing Emissions
  static bool variablyModifiedTypeEmission() { return false; }
  static bool buildLValueAlignmentAssumption() { return false; }
  static bool buildDerivedToBaseCastForDevirt() { return false; }
  static bool emitFunctionEpilog() { return false; }

  // References related stuff
  static bool ARC() { return false; } // Automatic reference counting

  // Clang early optimizations or things defered to LLVM lowering.
  static bool shouldUseBZeroPlusStoresToInitialize() { return false; }
  static bool shouldUseMemSetToInitialize() { return false; }
  static bool shouldSplitConstantStore() { return false; }
  static bool shouldCreateMemCpyFromGlobal() { return false; }
  static bool shouldReverseUnaryCondOnBoolExpr() { return false; }
  static bool fieldMemcpyizerBuildMemcpy() { return false; }
  static bool isTrivialAndisDefaultConstructor() { return false; }
  static bool isMemcpyEquivalentSpecialMember() { return false; }
  static bool constructABIArgDirectExtend() { return false; }
  static bool mayHaveIntegerOverflow() { return false; }
  static bool llvmLoweringPtrDiffConsidersPointee() { return false; }
  static bool emitNullCheckForDeleteCalls() { return false; }

  // Folding methods.
  static bool foldBinOpFMF() { return false; }

  // Fast math.
  static bool fastMathGuard() { return false; }
  static bool fastMathFlags() { return false; }
  static bool fastMathFuncAttributes() { return false; }

  // Exception handling
  static bool setLandingPadCleanup() { return false; }
  static bool isSEHTryScope() { return false; }
  static bool ehStack() { return false; }
  static bool emitStartEHSpec() { return false; }
  static bool emitEndEHSpec() { return false; }
  static bool simplifyCleanupEntry() { return false; }

  // Type qualifiers.
  static bool atomicTypes() { return false; }
  static bool volatileTypes() { return false; }
  static bool syncScopeID() { return false; }

  static bool capturedByInit() { return false; }
  static bool tryEmitAsConstant() { return false; }
  static bool incrementProfileCounter() { return false; }
  static bool createProfileWeightsForLoop() { return false; }
  static bool getProfileCount() { return false; }
  static bool emitCondLikelihoodViaExpectIntrinsic() { return false; }
  static bool requiresReturnValueCheck() { return false; }
  static bool shouldEmitLifetimeMarkers() { return false; }
  static bool peepholeProtection() { return false; }
  static bool CGCapturedStmtInfo() { return false; }
  static bool CGFPOptionsRAII() { return false; }
  static bool getFPFeaturesInEffect() { return false; }
  static bool cxxABI() { return false; }
  static bool openCL() { return false; }
  static bool CUDA() { return false; }
  static bool openMP() { return false; }
  static bool openMPRuntime() { return false; }
  static bool openMPRegionInfo() { return false; }
  static bool openMPTarget() { return false; }
  static bool isVarArg() { return false; }
  static bool setNonGC() { return false; }
  static bool volatileLoadOrStore() { return false; }
  static bool armComputeVolatileBitfields() { return false; }
  static bool setCommonAttributes() { return false; }
  static bool insertBuiltinUnpredictable() { return false; }
  static bool createInvariantGroup() { return false; }
  static bool addAutoInitAnnotation() { return false; }
  static bool addHeapAllocSiteMetadata() { return false; }
  static bool loopInfoStack() { return false; }
  static bool requiresCleanups() { return false; }
  static bool constantFoldsToSimpleInteger() { return false; }
  static bool checkFunctionCallABI() { return false; }
  static bool zeroInitializer() { return false; }
  static bool targetCodeGenInfoIsProtoCallVariadic() { return false; }
  static bool targetCodeGenInfoGetNullPointer() { return false; }
  static bool chainCalls() { return false; }
  static bool operandBundles() { return false; }
  static bool exceptions() { return false; }
  static bool metaDataNode() { return false; }
  static bool emitDeclMetadata() { return false; }
  static bool emitScalarRangeCheck() { return false; }
  static bool stmtExprEvaluation() { return false; }
  static bool setCallingConv() { return false; }
  static bool tryMarkNoThrow() { return false; }
  static bool indirectBranch() { return false; }
  static bool escapedLocals() { return false; }
  static bool deferredReplacements() { return false; }
  static bool shouldInstrumentFunction() { return false; }

  // Inline assembly
  static bool asmGoto() { return false; }
  static bool asmUnwindClobber() { return false; }
  static bool asmMemoryEffects() { return false; }
  static bool asmVectorType() { return false; }
  static bool asmLLVMAssume() { return false; }

  // C++ ABI support
  static bool handleBigEndian() { return false; }
  static bool handleAArch64Indirect() { return false; }
  static bool classifyArgumentTypeForAArch64() { return false; }
  static bool supportgetCoerceToTypeForAArch64() { return false; }
  static bool supportTySizeQueryForAArch64() { return false; }
  static bool supportTyAlignQueryForAArch64() { return false; }
  static bool supportisHomogeneousAggregateQueryForAArch64() { return false; }
  static bool supportisEndianQueryForAArch64() { return false; }
  static bool supportisAggregateTypeForABIAArch64() { return false; }

  //===--- ABI lowering --===//

  // Parameters may have additional attributes (e.g. [[noescape]]) that affect
  // the compiler. This is not yet supported in CIR.
  static bool extParamInfo() { return true; }

  // LangOpts may affect lowering, but we do not carry this information into CIR
  // just yet. Right now, it only instantiates the default lang options.
  static bool langOpts() { return true; }

  // Several type qualifiers are not yet supported in CIR, but important when
  // evaluating ABI-specific lowering.
  static bool qualifiedTypes() { return true; }

  // We're ignoring several details regarding ABI-halding for Swift.
  static bool swift() { return true; }

  // Despite carrying some information about variadics, we are currently
  // ignoring this to focus only on the code necessary to lower non-variadics.
  static bool variadicFunctions() { return true; }
};

} // namespace cir

#endif // CLANG_CIR_MISSINGFEATURES_H
