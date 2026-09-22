//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-rtti

// Compilers may prefix the type_info::name() result with a '*' to indicate that the type has internal linkage. This test checks that libc++ strips the '*' prefix from the name() result.

#include <typeinfo>
#include <cassert>

namespace {
struct AnonymousType {};
} // namespace

const std::type_info& local_type() {
  struct LocalType {};
  return typeid(LocalType);
}

int main(int, char**) {
  assert(typeid(AnonymousType).name()[0] != '*');
  assert(typeid(AnonymousType*).name()[0] != '*');
  assert(local_type().name()[0] != '*');
  return 0;
}
