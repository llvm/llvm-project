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

  template<> struct formatter<stacktrace_entry>;
*/

#include <cassert>
// #include <stacktrace>

int main(int, char**) {
  /*
    formatter<stacktrace_entry> interprets format-spec as a stacktrace-entry-format-spec.
    The syntax of format specifications is as follows:

      stacktrace-entry-format-spec :
        fill-and-align_[opt] width_[opt]

    [Note 1: The productions fill-and-align and width are described in [format.string.std]. - end note]

    A stacktrace_entry object se is formatted as if by copying to_string(se) through the output iterator
    of the context with additional padding and adjustments as specified by the format specifiers.
  */

  // TODO: stacktrace formatter: https://github.com/llvm/llvm-project/issues/105257

  return 0;
}
