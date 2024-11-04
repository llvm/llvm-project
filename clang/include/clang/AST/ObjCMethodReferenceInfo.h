//===- ObjcMethodReferenceInfo.h - API for ObjC method tracing --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines APIs for ObjC method tracing.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_OBJC_METHOD_REFERENCE_INFO_H
#define LLVM_CLANG_OBJC_METHOD_REFERENCE_INFO_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>

namespace clang {

class ObjCMethodDecl;

struct ObjCMethodReferenceInfo {
  static constexpr unsigned FormatVersion = 1;
  std::string Target, TargetVariant;

  /// Paths to the files in which ObjC methods are referenced.
  llvm::SmallVector<std::string, 4> FilePaths;

  /// A map from the index of a file in FilePaths to the list of ObjC methods.
  std::map<unsigned, llvm::SmallVector<const ObjCMethodDecl *, 4>> References;
};

/// This function serializes the ObjC message tracing information in JSON.
void serializeObjCMethodReferencesAsJson(const ObjCMethodReferenceInfo &Info,
                                         llvm::raw_ostream &OS);

} // namespace clang

#endif // LLVM_CLANG_OBJC_METHOD_REFERENCE_INFO_H
