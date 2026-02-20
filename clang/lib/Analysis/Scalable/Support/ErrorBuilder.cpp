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
#include <cassert>

namespace clang::ssaf {

static constexpr llvm::StringLiteral ErrorSeparator = " + ";
static constexpr llvm::StringLiteral ContextSeparator = "\n";

ErrorBuilder ErrorBuilder::wrap(llvm::Error E) {
  assert(
      E &&
      "Cannot wrap a success error - check for success before calling wrap()");

  std::optional<std::error_code> EC;
  std::vector<std::string> Messages;

  llvm::handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EI) {
    // Capture error code from the first error only.
    if (!EC)
      EC = EI.convertToErrorCode();

    // Collect messages from all errors.
    std::string ErrorMsg = EI.message();
    if (!ErrorMsg.empty())
      Messages.push_back(std::move(ErrorMsg));
  });

  assert(EC && "wrap() called with a non-success error but no handler fired - "
               "indicates a bug in handleAllErrors");

  ErrorBuilder Builder(*EC);

  // Combine all messages with " + " and push as a single context entry.
  Builder.pushContext(llvm::join(Messages, ErrorSeparator));

  return Builder;
}

ErrorBuilder &ErrorBuilder::context(const char *Msg) {
  pushContext(std::string(Msg));
  return *this;
}

llvm::Error ErrorBuilder::build() const {
  // Reverse the context stack so that the most recent context appears first
  // and the wrapped error (if any) appears last.
  // Note: Even if ContextStack is empty, we create an error with the stored
  // error code and an empty message (this is valid in LLVM).
  return llvm::createStringError(
      llvm::join(llvm::reverse(ContextStack), ContextSeparator), Code);
}

} // namespace clang::ssaf
