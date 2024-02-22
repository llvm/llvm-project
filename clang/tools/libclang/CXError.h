//===- CXError.h - Routines for manipulating CXErrors ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_LIBCLANG_CXERROR_H
#define LLVM_CLANG_TOOLS_LIBCLANG_CXERROR_H

#include "clang-c/CXErrorCode.h"
#include "llvm/Support/Error.h"

namespace clang::cxerror {

CXError create(llvm::Error E, CXErrorCode Code = CXError_Failure);
CXError create(llvm::StringRef ErrorDescription,
               CXErrorCode Code = CXError_Failure);

} // namespace clang::cxerror

#endif
