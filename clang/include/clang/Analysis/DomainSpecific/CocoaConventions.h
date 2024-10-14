//===- CocoaConventions.h - Special handling of Cocoa conventions -*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements cocoa naming convention analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_DOMAINSPECIFIC_COCOACONVENTIONS_H
#define LLVM_CLANG_ANALYSIS_DOMAINSPECIFIC_COCOACONVENTIONS_H

#include "clang/Basic/LLVM.h"
#include "clang/Support/Compiler.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
class FunctionDecl;
class QualType;

namespace ento {
namespace cocoa {

  CLANG_ABI bool isRefType(QualType RetTy, StringRef Prefix,
                 StringRef Name = StringRef());

  CLANG_ABI bool isCocoaObjectRef(QualType T);

}

namespace coreFoundation {
  CLANG_ABI bool isCFObjectRef(QualType T);

  CLANG_ABI bool followsCreateRule(const FunctionDecl *FD);
}

}} // end: "clang:ento"

#endif
