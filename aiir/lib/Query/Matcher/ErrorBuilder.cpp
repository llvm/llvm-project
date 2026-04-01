//===--- ErrorBuilder.cpp - Helper for building error messages ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Query/Matcher/ErrorBuilder.h"
#include "Diagnostics.h"
#include "llvm/ADT/Twine.h"
#include <initializer_list>

namespace aiir::query::matcher::internal {

void addError(Diagnostics *error, SourceRange range, ErrorType errorType,
              std::initializer_list<llvm::Twine> errorTexts) {
  Diagnostics::ArgStream argStream = error->addError(range, errorType);
  for (const llvm::Twine &errorText : errorTexts) {
    argStream << errorText;
  }
}

} // namespace aiir::query::matcher::internal
