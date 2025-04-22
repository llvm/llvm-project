//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cassert>
#include <stacktrace>

/*
  (19.6.5) Formatting support [stacktrace.format]

  template<> struct formatter<stacktrace_entry>;                            [1]
  template<class Allocator> struct formatter<basic_stacktrace<Allocator>>;  [2]
*/

int main(int, char**) {
  /*
    [1]
    template<> struct formatter<stacktrace_entry>;

    formatter<stacktrace_entry> interprets format-spec as a stacktrace-entry-format-spec.
    The syntax of format specifications is as follows:

      stacktrace-entry-format-spec :
        fill-and-align_[opt] width_[opt]

    [Note 1: The productions fill-and-align and width are described in [format.string.std]. — end note]

    A stacktrace_entry object se is formatted as if by copying to_string(se) through the output iterator
    of the context with additional padding and adjustments as specified by the format specifiers.
  */

  // TODO(stacktrace23): needs `formatter`

  /*
    [2]
    template<class Allocator> struct formatter<basic_stacktrace<Allocator>>;
    
    For formatter<basic_stacktrace<Allocator>>, format-spec is empty.
    
    A basic_stacktrace<Allocator> object s is formatted as if by copying to_string(s) through the
    output iterator of the context.
  */

  // TODO(stacktrace23): needs `formatter`

  return 0;
}
