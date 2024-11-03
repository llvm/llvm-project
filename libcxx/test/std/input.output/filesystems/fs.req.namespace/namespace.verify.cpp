//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++03 || c++11 || c++14
// UNSUPPORTED: no-filesystem

// <filesystem>

// namespace std::filesystem

#include <filesystem>

namespace fs = std::filesystem; // expected-error-re {{{{(no namespace named 'filesystem' in namespace 'std';)|(expected namespace name)}}}}
