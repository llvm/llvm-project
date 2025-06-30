// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

#include "Inputs/std-coroutine.h"

namespace GH58172 {
template<typename Fn>
int f2(int, Fn&&)
{
  return 0;
}

int f1()
{
  return f2(v1, []() -> task<int> {   // expected-error {{no template named 'task'}} \
                                         expected-error {{use of undeclared identifier 'v1'}}
    co_return v2;                     // expected-error {{use of undeclared identifier 'v2'}}
  });
}
}
