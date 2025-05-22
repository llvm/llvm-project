//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that user specializations of std::hash have default visibility

// RUN: %{cxx} %s %{flags} %{compile_flags} %{link_flags} -DSHARED -fPIC -shared -o %t.shared_lib
// RUN: %{build} %t.shared_lib
// RUN: %{run}

#include <__fwd/functional.h>

struct S {};

template <>
struct std::hash<S> {
  void operator()();
};

#ifdef SHARED
void std::hash<S>::operator()() {}
#else
int main(int, char**) {
  std::hash<S>()();
  return 0;
}
#endif
