//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <vector>
#include <utility>

#define CODESIZE_BM [[gnu::abi_tag("BENCHMARK_THIS")]]

CODESIZE_BM auto bm_default_constructor() {
  return std::vector<int>();
}

CODESIZE_BM auto bm_copy_constructor(const std::vector<int>& v) {
  return v;
}

CODESIZE_BM auto bm_copy_assign(std::vector<int>& lhs, const std::vector<int>& rhs) {
  lhs = rhs;
}

CODESIZE_BM auto bm_move_constructor(std::vector<int>&& v) {
  return std::move(v);
}

CODESIZE_BM auto bm_push_back(std::vector<int>& v) {
  v.push_back(1);
}

CODESIZE_BM auto bm_emplace_back(std::vector<int>& v) {
  v.emplace_back(1);
}
