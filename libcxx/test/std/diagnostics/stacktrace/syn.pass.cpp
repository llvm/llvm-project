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

#include <memory>
#include <type_traits>

/*

// (19.6.2) Header <stacktrace> synopsis [stacktrace.syn]

namespace std {

  // [stacktrace.entry], class stacktrace_entry
  class stacktrace_entry;                                               [1]

  // [stacktrace.basic], class template basic_stacktrace
  template<class Allocator>
    class basic_stacktrace;                                             [2]

  // basic_stacktrace typedef-names
  using stacktrace = basic_stacktrace<allocator<stacktrace_entry>>;     [3]

  // [stacktrace.basic.nonmem], non-member functions
  template<class Allocator>
    void swap(basic_stacktrace<Allocator>& a,
              basic_stacktrace<Allocator>& b)
                noexcept(noexcept(a.swap(b)));                          [4]

  string to_string(const stacktrace_entry& f);                          [5]

  template<class Allocator>
    string to_string(const basic_stacktrace<Allocator>& st);            [6]

  ostream& operator<<(ostream& os, const stacktrace_entry& f);          [7]
  template<class Allocator>
    ostream& operator<<(ostream& os,
                        const basic_stacktrace<Allocator>& st);         [8]

  // [stacktrace.format], formatting support
  template<> struct formatter<stacktrace_entry>;                        [9]
  template<class Allocator>
    struct formatter<basic_stacktrace<Allocator>>;                      [10]

  namespace pmr {
    using stacktrace =
        basic_stacktrace<polymorphic_allocator<stacktrace_entry>>;      [11]
  }

  // [stacktrace.basic.hash], hash support
  template<> struct hash<stacktrace_entry>;                             [12]
  template<class Allocator> struct hash<basic_stacktrace<Allocator>>;   [13]
}
*/

// static_assert(__cpp_lib_stacktrace == 202011L);

int main(int, char**) {
  // Very basic tests to ensure the required things are declared.
  // Only checking for types' and functions' existence, parameter and return types,
  // and type-completeness.  Functional tests exist in the other .cpp's in this directory.

  using Alloc = std::allocator<std::stacktrace_entry>;

  // [1]
  // [stacktrace.entry], class stacktrace_entry
  // class stacktrace_entry;
  using T1 = std::stacktrace_entry;
  assert(std::is_constructible_v<T1>);

  // [2]
  // [stacktrace.basic], class template basic_stacktrace
  // template<class Allocator>
  //   class basic_stacktrace;
  using T2 = std::basic_stacktrace<Alloc>;
  assert(std::is_constructible_v<T2>);

  // [3]
  // basic_stacktrace typedef-names
  // using stacktrace = basic_stacktrace<allocator<stacktrace_entry>>;
  using T3 = std::stacktrace;
  static_assert(std::is_same_v<T3, std::basic_stacktrace<std::allocator<std::stacktrace_entry>>>);

  // [4]
  // [stacktrace.basic.nonmem], non-member functions
  // template<class Allocator>
  //   void swap(basic_stacktrace<Allocator>& a, basic_stacktrace<Allocator>& b)
  //     noexcept(noexcept(a.swap(b)));
  std::basic_stacktrace<Alloc> a;
  std::basic_stacktrace<Alloc> b;
  std::swap(a, b);

  // [5]
  // string to_string(const stacktrace_entry& f);
  using T5 = decltype(std::to_string(std::stacktrace_entry()));
  static_assert(std::is_same_v<std::string, T5>);

  // [6]
  // template<class Allocator>
  // string to_string(const basic_stacktrace<Allocator>& st);
  using T6 = decltype(std::to_string(std::basic_stacktrace<Alloc>()));
  static_assert(std::is_same_v<std::string, T6>);

  // [7]
  // ostream& operator<<(ostream& os, const stacktrace_entry& f);
  std::ostream* os;
  using T7 = decltype(operator<<(*os, std::stacktrace_entry()));
  static_assert(std::is_same_v<std::ostream&, T7>);

  // [8]
  // template<class Allocator>
  //   ostream& operator<<(ostream& os, const basic_stacktrace<Allocator>& st);
  using T8 = decltype(operator<<(*os, std::basic_stacktrace<Alloc>()));
  static_assert(std::is_same_v<std::ostream&, T8>);

  // [9]
  // template<> struct formatter<stacktrace_entry>;
  //   using T9 = std::formatter<std::stacktrace_entry>;
  //   static_assert(std::is_constructible_v<T9>);

  // TODO(stacktrace23): needs `formatter`

  // [10]
  // template<class Allocator> struct formatter<basic_stacktrace<Allocator>>;
  //   using T10 = std::formatter<std::basic_stacktrace<Alloc>>;
  //   static_assert(std::is_constructible_v<T10>);

  // TODO(stacktrace23): needs `formatter`

  // [11]
  // namespace pmr { using stacktrace = basic_stacktrace<polymorphic_allocator<stacktrace_entry>>; }
  using AllocPMR = std::pmr::polymorphic_allocator<std::stacktrace_entry>;
  using BasicPMR = std::basic_stacktrace<AllocPMR>;
  using T11      = std::pmr::stacktrace;
  static_assert(std::is_same_v<T11, BasicPMR>);

  return 0;
}
