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

  // CIRGenFunction implementation details
  static bool cgfSymbolTable() { return false; }

  // Unhandled global/linkage information.
  static bool opGlobalDSOLocal() { return false; }
  static bool opGlobalThreadLocal() { return false; }
  static bool opGlobalConstant() { return false; }
  static bool opGlobalAlignment() { return false; }
  static bool opGlobalWeakRef() { return false; }

  static bool supportIFuncAttr() { return false; }
  static bool supportVisibility() { return false; }
  static bool supportComdat() { return false; }

  // Load/store attributes
  static bool opLoadStoreThreadLocal() { return false; }
  static bool opLoadEmitScalarRangeCheck() { return false; }
  static bool opLoadBooleanRepresentation() { return false; }
  static bool opLoadStoreTbaa() { return false; }
  static bool opLoadStoreMemOrder() { return false; }
  static bool opLoadStoreVolatile() { return false; }
  static bool opLoadStoreAlignment() { return false; }
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
  static bool opFuncCallingConv() { return false; }
  static bool opFuncExtraAttrs() { return false; }
  static bool opFuncDsolocal() { return false; }
  static bool opFuncLinkage() { return false; }
  static bool opFuncVisibility() { return false; }

  // CallOp handling
  static bool opCallBuiltinFunc() { return false; }
  static bool opCallPseudoDtor() { return false; }
  static bool opCallArgs() { return false; }
  static bool opCallReturn() { return false; }
  static bool opCallArgEvaluationOrder() { return false; }
  static bool opCallCallConv() { return false; }
  static bool opCallSideEffect() { return false; }
  static bool opCallChainCall() { return false; }
  static bool opCallNoPrototypeFunc() { return false; }
  static bool opCallMustTail() { return false; }
  static bool opCallIndirect() { return false; }
  static bool opCallVirtual() { return false; }
  static bool opCallInAlloca() { return false; }
  static bool opCallAttrs() { return false; }
  static bool opCallSurroundingTry() { return false; }
  static bool opCallASTAttr() { return false; }

  // ScopeOp handling
  static bool opScopeCleanupRegion() { return false; }

  // Unary operator handling
  static bool opUnaryPromotionType() { return false; }

  // Clang early optimizations or things defered to LLVM lowering.
  static bool mayHaveIntegerOverflow() { return false; }
  static bool shouldReverseUnaryCondOnBoolExpr() { return false; }

  // Misc
  static bool cxxABI() { return false; }
  static bool tryEmitAsConstant() { return false; }
  static bool constructABIArgDirectExtend() { return false; }
  static bool opGlobalViewAttr() { return false; }
  static bool lowerModeOptLevel() { return false; }
  static bool opTBAA() { return false; }
  static bool objCLifetime() { return false; }
  static bool objCBlocks() { return false; }
  static bool emitNullabilityCheck() { return false; }
  static bool emitLValueAlignmentAssumption() { return false; }
  static bool emitLifetimeMarkers() { return false; }
  static bool astVarDeclInterface() { return false; }
  static bool stackSaveOp() { return false; }
  static bool aggValueSlot() { return false; }
  static bool generateDebugInfo() { return false; }
  static bool pointerOverflowSanitizer() { return false; }
  static bool fpConstraints() { return false; }
  static bool sanitizers() { return false; }
  static bool addHeapAllocSiteMetadata() { return false; }
  static bool targetCodeGenInfoGetNullPointer() { return false; }
  static bool loopInfoStack() { return false; }
  static bool requiresCleanups() { return false; }
  static bool createProfileWeightsForLoop() { return false; }
  static bool emitCondLikelihoodViaExpectIntrinsic() { return false; }
  static bool pgoUse() { return false; }
  static bool cgFPOptionsRAII() { return false; }
  static bool metaDataNode() { return false; }
  static bool fastMathFlags() { return false; }
  static bool alignCXXRecordDecl() { return false; }
  static bool setNonGC() { return false; }
  static bool incrementProfileCounter() { return false; }
  static bool insertBuiltinUnpredictable() { return false; }
  static bool objCGC() { return false; }
  static bool weakRefReference() { return false; }
  static bool hip() { return false; }
  static bool setObjCGCLValueClass() { return false; }
  static bool mangledNames() { return false; }
  static bool setDLLStorageClass() { return false; }
  static bool openMP() { return false; }

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
  static bool complexCreateOp() { return false; }
  static bool complexImagOp() { return false; }
  static bool complexRealOp() { return false; }
  static bool ifOp() { return false; }
  static bool invokeOp() { return false; }
  static bool labelOp() { return false; }
  static bool ptrDiffOp() { return false; }
  static bool ptrStrideOp() { return false; }
  static bool selectOp() { return false; }
  static bool switchOp() { return false; }
  static bool ternaryOp() { return false; }
  static bool tryOp() { return false; }
  static bool zextOp() { return false; }

  // Future CIR attributes
  static bool optInfoAttr() { return false; }
};

} // namespace cir

#endif // CLANG_CIR_MISSINGFEATURES_H
