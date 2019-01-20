// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/coroutine>

// Test that <experimental/coroutine> includes <new>

#include <experimental/coroutine>

int main(){
  // std::nothrow is not implicitly defined by the compiler when the include is
  // missing, unlike other parts of <new>. Therefore we use std::nothrow to
  // test for #include <new>
  (void)std::nothrow;

}
