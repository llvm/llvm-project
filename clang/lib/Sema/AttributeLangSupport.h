//===- AttributeLangSupport.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of AttributeLangSupport, which is used in
/// diagnostics to indicate the language in which an attribute is (not)
/// supported.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_ATTRIBUTE_LANG_SUPPORT_H
#define LLVM_CLANG_SEMA_ATTRIBUTE_LANG_SUPPORT_H

// NOTE: The order should match the order of the %select in
//       err_attribute_not_supported_in_lang in DiagnosticSemaKinds.td
namespace clang::AttributeLangSupport {
enum LANG {
  C,
  Cpp,
  HLSL,
  ObjC,
  SYCL,
};
} // end namespace clang::AttributeLangSupport

#endif // LLVM_CLANG_SEMA_ATTRIBUTE_LANG_SUPPORT_H
