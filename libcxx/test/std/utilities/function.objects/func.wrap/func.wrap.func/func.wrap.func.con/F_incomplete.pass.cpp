//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>

// class function<R(ArgTypes...)>

// template<class F> function(F);

// Allow incomplete argument types in the __is_callable check

#include <functional>
#include <cassert>

#include "test_macros.h"

struct X{
    typedef std::function<void(X&)> callback_type;
    virtual ~X() {}
private:
    callback_type _cb;
};

struct IncompleteReturnType {
  std::function<IncompleteReturnType ()> fn;
};


int called = 0;
IncompleteReturnType test_fn() {
  ++called;
  IncompleteReturnType I;
  return I;
}

// See llvm.org/PR34298
void test_pr34298()
{
  static_assert(std::is_copy_constructible<IncompleteReturnType>::value, "");
  static_assert(std::is_copy_assignable<IncompleteReturnType>::value, "");
  {
    IncompleteReturnType X;
    X.fn = test_fn;
    const IncompleteReturnType& CX = X;
    IncompleteReturnType X2 = CX;
    assert(X2.fn);
    assert(called == 0);
    X2.fn();
    assert(called == 1);
  }
  {
    IncompleteReturnType Empty;
    IncompleteReturnType X2 = Empty;
    assert(!X2.fn);
  }
}

struct LateReturnType;

void copy_while_incomplete(const std::function<LateReturnType(int)>& f) {
  std::function<LateReturnType(int)> copy = f;
  (void)copy;
}

struct LateReturnType {
  int value;
};

LateReturnType make_late_return_type(int value) { return LateReturnType{value}; }

void test_return_type_completed_after_copy() {
  std::function<LateReturnType(int)> f = make_late_return_type;
  assert(f(42).value == 42);
}

int main(int, char**) {
  test_pr34298();
  test_return_type_completed_after_copy();

  return 0;
}
