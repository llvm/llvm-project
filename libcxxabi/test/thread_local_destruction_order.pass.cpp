//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-threads

// XFAIL: LIBCXX-FREEBSD-FIXME

// TODO: This test does start working with newer updates of the mingw-w64
// toolchain, when it includes the following commit:
// https://github.com/mingw-w64/mingw-w64/commit/71eddccd746c56d9cde28bb5620d027d49259de9
// Thus, remove this UNSUPPORTED marking after the next update of the CI
// toolchain.
// UNSUPPORTED: target={{.*-windows-gnu}}

#include <cassert>
#include <thread>

#include "make_test_thread.h"

int seq = 0;

class OrderChecker {
public:
  explicit OrderChecker(int n) : n_{n} { }

  ~OrderChecker() {
    assert(seq++ == n_);
  }

private:
  int n_;
};

template <int ID>
class CreatesThreadLocalInDestructor {
public:
  ~CreatesThreadLocalInDestructor() {
    thread_local OrderChecker checker{ID};
  }
};

OrderChecker global{7};

void thread_fn() {
  static OrderChecker fn_static{5};
  thread_local CreatesThreadLocalInDestructor<2> creates_tl2;
  thread_local OrderChecker fn_thread_local{1};
  thread_local CreatesThreadLocalInDestructor<0> creates_tl0;
}

int main(int, char**) {
  static OrderChecker fn_static{6};

  support::make_test_thread(thread_fn).join();
  assert(seq == 3);

  thread_local OrderChecker fn_thread_local{4};
  thread_local CreatesThreadLocalInDestructor<3> creates_tl;

  return 0;
}
