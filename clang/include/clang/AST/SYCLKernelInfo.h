//===--- SYCLKernelInfo.h --- Information about SYCL kernels --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares types used to describe SYCL kernels.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_SYCLKERNELINFO_H
#define LLVM_CLANG_AST_SYCLKERNELINFO_H

#include "clang/AST/CanonicalType.h"
#include "clang/AST/Decl.h"
#include "clang/AST/TypeBase.h"

namespace clang {

class SYCLKernelInfo {
public:
  SYCLKernelInfo(CanQualType KernelNameType,
                 const FunctionDecl *KernelEntryPointDecl,
                 const std::string &KernelName)
      : KernelNameType(KernelNameType),
        KernelEntryPointDecl(KernelEntryPointDecl), KernelName(KernelName) {}

  CanQualType getKernelNameType() const { return KernelNameType; }

  const FunctionDecl *getKernelEntryPointDecl() const {
    return KernelEntryPointDecl;
  }

  const std::string &GetKernelName() const { return KernelName; }

private:
  CanQualType KernelNameType;
  const FunctionDecl *KernelEntryPointDecl;
  std::string KernelName;
};

} // namespace clang

#endif // LLVM_CLANG_AST_SYCLKERNELINFO_H
