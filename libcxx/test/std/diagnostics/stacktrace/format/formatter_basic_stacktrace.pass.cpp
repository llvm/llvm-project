//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

/*
  (19.6.5) Formatting support [stacktrace.format]

  template<class Allocator> struct formatter<basic_stacktrace<Allocator>>;
*/

#include <cassert>
// #include <stacktrace>

int main(int, char**) {
  /*
    For formatter<basic_stacktrace<Allocator>>, format-spec is empty.
    
    A basic_stacktrace<Allocator> object s is formatted as if by copying to_string(s) through the
    output iterator of the context.
  */

  // TODO: stacktrace formatter: https://github.com/llvm/llvm-project/issues/105257

  return 0;
}
