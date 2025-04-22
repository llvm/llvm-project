//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <stacktrace>

#include <cassert>
#include <sstream>

/*
  (19.6.4.6) Non-member functions

  template<class Allocator>
    void swap(basic_stacktrace<Allocator>& a, basic_stacktrace<Allocator>& b)
      noexcept(noexcept(a.swap(b)));

  string to_string(const stacktrace_entry& f);

  template<class Allocator>
    string to_string(const basic_stacktrace<Allocator>& st);

  ostream& operator<<(ostream& os, const stacktrace_entry& f);
  template<class Allocator>
    ostream& operator<<(ostream& os, const basic_stacktrace<Allocator>& st);
*/
int main(int, char**) {
  /*
  template<class Allocator>
  void swap(basic_stacktrace<Allocator>& a, basic_stacktrace<Allocator>& b)
    noexcept(noexcept(a.swap(b)));
  Effects: Equivalent to a.swap(b).
  */
  std::stacktrace empty;
  auto current = std::stacktrace::current();

  std::stacktrace a(empty);
  std::stacktrace b(current);
  assert(a == empty);
  assert(b == current);

  std::swap(a, b);
  assert(a == current);
  assert(b == empty);

  /*
  string to_string(const stacktrace_entry& f);
  Returns: A string with a description of f.
  Recommended practice: The description should provide information about the contained evaluation,
  including information from f.source_file() and f.source_line().
  */

  assert(std::to_string(a[0]).contains("main"));
  assert(std::to_string(a[0]).contains("basic.nonmem.pass"));

  /*
  template<class Allocator>
  string to_string(const basic_stacktrace<Allocator>& st);
  Returns: A string with a description of st.
  [Note 1: The number of lines is not guaranteed to be equal to st.size(). — end note]
  */

  assert(std::to_string(a).contains("main"));
  assert(std::to_string(a).contains("basic.nonmem.pass"));

  /*
  ostream& operator<<(ostream& os, const stacktrace_entry& f);
  Effects: Equivalent to: return os << to_string(f);
  */

  std::stringstream entry_os;
  entry_os << a[0];
  assert(entry_os.str() == std::to_string(a[0]));

  /*
  template<class Allocator>
    ostream& operator<<(ostream& os, const basic_stacktrace<Allocator>& st);
  Effects: Equivalent to: return os << to_string(st);
  */

  std::stringstream trace_os;
  trace_os << a;
  assert(trace_os.str() == std::to_string(a));

  return 0;
}
