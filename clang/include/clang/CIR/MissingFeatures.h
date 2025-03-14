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

  // Misc
  static bool scalarConversionOpts() { return false; }
  static bool tryEmitAsConstant() { return false; }
  static bool constructABIArgDirectExtend() { return false; }
  static bool opGlobalViewAttr() { return false; }
  static bool lowerModeOptLevel() { return false; }
  static bool opTBAA() { return false; }
  static bool objCLifetime() { return false; }
  static bool emitNullabilityCheck() { return false; }
  static bool astVarDeclInterface() { return false; }
  static bool stackSaveOp() { return false; }
  static bool aggValueSlot() { return false; }

  static bool unsizedTypes() { return false; }
};

} // namespace cir

#endif // CLANG_CIR_MISSINGFEATURES_H
