//===- MockTUSummaryBuilder.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSISFRAMEWORK_REGISTRIES_MOCKTUSUMMARYBUILDER_H
#define LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSISFRAMEWORK_REGISTRIES_MOCKTUSUMMARYBUILDER_H

#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryBuilder.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::ssaf {

class MockTUSummaryBuilder : public TUSummaryBuilder {
public:
  using TUSummaryBuilder::TUSummaryBuilder;
  void sendMessage(llvm::Twine Message) { Stream << Message << '\n'; }
  std::string consumeMessages() { return std::move(OutputBuffer); }

private:
  std::string OutputBuffer;
  llvm::raw_string_ostream Stream = llvm::raw_string_ostream{OutputBuffer};
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSISFRAMEWORK_REGISTRIES_MOCKTUSUMMARYBUILDER_H
