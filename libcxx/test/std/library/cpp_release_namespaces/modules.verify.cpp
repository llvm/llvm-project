//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29
// UNSUPPORTED: gcc
// XFAIL: has-no-cxx-module-support

// C++20 modules are incompatible with Clang modules
// ADDITIONAL_COMPILE_FLAGS: -fno-modules

// MODULE_DEPENDENCIES: std

// PROTOTYPE: P3866R0 (V2: An Evolution Path for the Standard Library)
//
// Check what's visible through the release namespaces with `import std;`.

import std;

namespace cpp = std::cpp29;

// Entities explicitly exported from the release namespace work.
static_assert(std::is_same_v<cpp::vector<int>, std::vector2<int>>);
static_assert(std::is_same_v<cpp::pmr::vector<int>, std::pmr::vector2<int>>);

// Entities that are only reachable through the using-directive in the release namespace. This requires the
// std module to re-export the using-directives, see cpp_release_namespaces.inc.
static_assert(std::is_same_v<cpp::string, std::string>);
static_assert(std::is_same_v<std::cpp26::vector<int>, std::vector<int>>);
static_assert(std::is_same_v<cpp::pmr::string, std::pmr::string>);

// The using-directive ambiguity also happens with modules.
void f() {
  using namespace std::cpp29;
  vector<int> v; // expected-error {{reference to 'vector' is ambiguous}}
}
