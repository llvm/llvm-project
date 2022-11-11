//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_META_UNARY_COMP_COMMON_H
#define TEST_META_UNARY_COMP_COMMON_H

#include "test_macros.h"

#if TEST_STD_VER >= 11
struct TrivialNotNoexcept {
  TrivialNotNoexcept() noexcept(false)                                     = default;
  TrivialNotNoexcept(const TrivialNotNoexcept&) noexcept(false)            = default;
  TrivialNotNoexcept(TrivialNotNoexcept&&) noexcept(false)                 = default;
  TrivialNotNoexcept& operator=(const TrivialNotNoexcept&) noexcept(false) = default;
  TrivialNotNoexcept& operator=(TrivialNotNoexcept&&) noexcept(false)      = default;
};
#endif

class Empty {};

struct NotEmpty {
  virtual ~NotEmpty();
};

union Union {};

struct bit_zero {
  int : 0;
};

struct A {
  A();
  A(const A&);
  A& operator=(const A&);
};

class Abstract
{
    virtual ~Abstract() = 0;
};

#endif // TEST_META_UNARY_COMP_COMMON_H
