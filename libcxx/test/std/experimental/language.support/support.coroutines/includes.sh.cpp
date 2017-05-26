// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// REQUIRES: fcoroutines-ts

// RUN: %build -fcoroutines-ts
// RUN: %run

// <experimental/coroutine>

// Test that <experimental/coroutine> includes <new>

#include <experimental/coroutine>



int main(){
  // std::nothrow is not implicitly defined by the compiler when the include is
  // missing, unlike other parts of <new>. Therefore we use std::nothrow to
  // test for #include <new>
  (void)std::nothrow;

}
