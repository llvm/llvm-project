//===- ErrorBuilder.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Support/ErrorBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

namespace clang::ssaf {

ErrorBuilder ErrorBuilder::wrap(llvm::Error E) {
  if (!E) {
    llvm::consumeError(std::move(E));
    // Return builder with generic error code for success case.
    return ErrorBuilder(std::make_error_code(std::errc::invalid_argument));
  }

  std::error_code EC;
  bool FirstError = true;
  ErrorBuilder Builder(std::make_error_code(std::errc::invalid_argument));

  llvm::handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EI) {
    // Capture error code from the first error only.
    if (FirstError) {
      EC = EI.convertToErrorCode();
      Builder.Code = EC;
      FirstError = false;
    }

    // Collect messages from all errors.
    std::string ErrorMsg = EI.message();
    if (!ErrorMsg.empty()) {
      Builder.ContextStack.push_back(std::move(ErrorMsg));
    }
  });

  return Builder;
}

ErrorBuilder &ErrorBuilder::context(const char *Msg) {
  ContextStack.push_back(Msg);
  return *this;
}

llvm::Error ErrorBuilder::build() {
  if (ContextStack.empty())
    return llvm::Error::success();

  // Reverse the context stack so that the most recent context appears first
  // and the wrapped error (if any) appears last.
  return llvm::createStringError(llvm::join(llvm::reverse(ContextStack), "\n"),
                                 Code);
}

} // namespace clang::ssaf
