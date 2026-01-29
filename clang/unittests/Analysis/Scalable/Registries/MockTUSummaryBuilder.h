//===- MockTUSummaryBuilder.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/TUSummary/TUSummaryBuilder.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::ssaf {

class MockTUSummaryBuilder : public TUSummaryBuilder {
public:
  void sendMessage(llvm::Twine Message) { Stream << Message << '\n'; }
  std::string consumeMessages() { return std::move(OutputBuffer); }

private:
  std::string OutputBuffer;
  llvm::raw_string_ostream Stream = llvm::raw_string_ostream{OutputBuffer};
};

} // namespace clang::ssaf
