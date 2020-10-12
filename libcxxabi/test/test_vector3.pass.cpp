//===------------------------- test_vector3.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

#include "cxxabi.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <exception>

#include <memory>

// use dtors instead of try/catch
namespace test1 {
    struct B {
         ~B() {
            printf("should not be run\n");
            exit(10);
            }
};

struct A {
 ~A()
#if __has_feature(cxx_noexcept)
    noexcept(false)
#endif
 {
   B b;
   throw 0;
 }
};
}  // test1

void my_terminate() { exit(0); }

template <class T>
void destroy(void* v)
{
  T* t = static_cast<T*>(v);
  t->~T();
}

int main(int, char**)
{
  std::set_terminate(my_terminate);
  {
  typedef test1::A Array[10];
  Array a[10]; // calls _cxa_vec_dtor
  __cxxabiv1::__cxa_vec_dtor(a, 10, sizeof(test1::A), destroy<test1::A>);
  assert(false);
  }

  return 0;
}
