//===---- UnimplementedFeatureGuarding.h - Checks against NYI ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file introduces some helper classes to guard against features that
// CodeGen supports that we do not have and also do not have great ways to
// assert against.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_UFG
#define LLVM_CLANG_LIB_CIR_UFG

namespace cir {
struct UnimplementedFeature {
  // TODO(CIR): Implement the CIRGenFunction::buildTypeCheck method that handles
  // sanitizer related type check features
  static bool buildTypeCheck() { return false; }
  static bool tbaa() { return false; }
  static bool cleanups() { return false; }

  // cir::VectorType is in progress, so cirVectorType() will go away soon.
  // Start adding feature flags for more advanced vector types and operations
  // that will take longer to implement.
  static bool cirVectorType() { return false; }
  static bool scalableVectors() { return false; }
  static bool vectorConstants() { return false; }

  // Address space related
  static bool addressSpace() { return false; }
  static bool addressSpaceInGlobalVar() { return false; }
  static bool getASTAllocaAddressSpace() { return false; }

  // Clang codegen options
  static bool strictVTablePointers() { return false; }

  // Unhandled global/linkage information.
  static bool unnamedAddr() { return false; }
  static bool setComdat() { return false; }
  static bool setGlobalVarSection() { return false; }
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

  // ObjC
  static bool setObjCGCLValueClass() { return false; }

  // Debug info
  static bool generateDebugInfo() { return false; }

  // LLVM Attributes
  static bool attributeBuiltin() { return false; }
  static bool attributeNoBuiltin() { return false; }
  static bool parameterAttributes() { return false; }

  // Coroutines
  static bool unhandledException() { return false; }

  // Missing Emissions
  static bool variablyModifiedTypeEmission() { return false; }
  static bool buildLValueAlignmentAssumption() { return false; }
  static bool buildDerivedToBaseCastForDevirt() { return false; }

  // Data layout
  static bool dataLayoutGetIndexTypeSizeInBits() { return false; }

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

  // Folding methods.
  static bool foldBinOpFMF() { return false; }

  // Fast math.
  static bool fastMathGuard() { return false; }
  static bool fastMathFlags() { return false; }
  static bool fastMathFuncAttributes() { return false; }

  // Type qualifiers.
  static bool atomicTypes() { return false; }
  static bool volatileTypes() { return false; }

  static bool capturedByInit() { return false; }
  static bool tryEmitAsConstant() { return false; }
  static bool incrementProfileCounter() { return false; }
  static bool createProfileWeightsForLoop() { return false; }
  static bool emitCondLikelihoodViaExpectIntrinsic() { return false; }
  static bool requiresReturnValueCheck() { return false; }
  static bool shouldEmitLifetimeMarkers() { return false; }
  static bool peepholeProtection() { return false; }
  static bool CGCapturedStmtInfo() { return false; }
  static bool cxxABI() { return false; }
  static bool openCL() { return false; }
  static bool openMP() { return false; }
  static bool openMPRuntime() { return false; }
  static bool openMPTarget() { return false; }
  static bool ehStack() { return false; }
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
  static bool alignedLoad() { return false; }
  static bool checkFunctionCallABI() { return false; }
  static bool zeroInitializer() { return false; }
  static bool targetCodeGenInfoIsProtoCallVariadic() { return false; }
  static bool chainCalls() { return false; }
  static bool operandBundles() { return false; }
  static bool exceptions() { return false; }
  static bool metaDataNode() { return false; }
  static bool isSEHTryScope() { return false; }
  static bool emitScalarRangeCheck() { return false; }
  static bool stmtExprEvaluation() { return false; }
};
} // namespace cir

#endif
