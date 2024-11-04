//===--------- QueryProfile.h - clang-query ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_PROFILE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_PROFILE_H

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Timer.h"
#include <optional>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace clang::query {

class QueryProfile {
public:
  llvm::StringMap<llvm::TimeRecord> Records;
  QueryProfile() = default;
  ~QueryProfile();

private:
  std::optional<llvm::TimerGroup> TG;
  void printUserFriendlyTable(llvm::raw_ostream &OS);
};

} // namespace clang::query

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_PROFILE_H
