//===- CXError.cpp - Routines for manipulating CXErrors -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CXError.h"
#include "llvm/Support/CBindingWrapping.h"

using namespace clang;
using llvm::Error;

namespace {

struct WrappedError {
  CXErrorCode Code;
  std::string Description;
};

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(WrappedError, CXError)

} // namespace

CXError cxerror::create(Error E, CXErrorCode Code) {
  if (E)
    return wrap(new WrappedError{Code, llvm::toString(std::move(E))});
  return nullptr;
}

CXError cxerror::create(llvm::StringRef ErrorDescription, CXErrorCode Code) {
  return wrap(new WrappedError{Code, std::string(ErrorDescription)});
}

enum CXErrorCode clang_Error_getCode(CXError CE) { return unwrap(CE)->Code; }

const char *clang_Error_getDescription(CXError CE) {
  return unwrap(CE)->Description.c_str();
}

void clang_Error_dispose(CXError CE) { delete unwrap(CE); }
