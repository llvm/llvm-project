//===--- MockHeaders.h - Mock headers for dataflow analyses -*- C++ -----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines mock headers for testing of dataflow analyses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_MOCK_HEADERS_H_
#define LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_MOCK_HEADERS_H_

#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace dataflow {
namespace test {

std::vector<std::pair<std::string, std::string>> getMockHeaders();

} // namespace test
} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_MOCK_HEADERS_H_
