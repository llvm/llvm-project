//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// Make sure basic_string constructors and functions operate properly for allocators with small `size_type`s.
// Related issue: https://github.com/llvm/llvm-project/issues/125187

// constexpr basic_string(
//     basic_string&& str, size_type pos, size_type n, const Allocator& a = Allocator());      // C++23
// basic_string(const basic_string& str, size_type pos, size_type n,
//              const Allocator& a = Allocator());                                             // constexpr since C++20
// basic_string& assign(const basic_string& str, size_type pos, size_type n=npos);             // constexpr since C++20
// template <class T>
//     basic_string& assign(const T& t, size_type pos, size_type n=npos);                      // C++17, constexpr since C++20
// basic_string& append(const basic_string& str, size_type pos, size_type n=npos);             // constexpr since C++20
// template <class T>
//     basic_string& append(const T& t, size_type pos, size_type n=npos);                      // C++17, constexpr since C++20
// basic_string& insert(size_type pos1, const basic_string& str,
//                      size_type pos2, size_type n=npos);                                     // constexpr since C++20
// template <class T>
//     basic_string& insert(size_type pos1, const T& t, size_type pos2, size_type n=npos);     // C++17, constexpr since C++20
// basic_string& erase(size_type pos = 0, size_type n = npos);                                 // constexpr since C++20
// basic_string& replace(size_type pos, size_type n1, const value_type* s, size_type n2);      // constexpr since C++20
// basic_string& replace(size_type pos, size_type n1, size_type n2, value_type c);             // constexpr since C++20
// basic_string& replace(size_type pos1, size_type n1, const basic_string& str,
//                       size_type pos2, size_type n2=npos);                                   // constexpr since C++20
// template <class T>
//     basic_string& replace(size_type pos1, size_type n1, const T& t,
//                           size_type pos2, size_type n2= npos);                              // C++17, constexpr since C++20
// size_type copy(value_type* s, size_type n, size_type pos = 0) const;                        // constexpr since C++20
// template <class T>
//     int compare(const T& t) const noexcept;                                                 // C++17, constexpr since C++20
// int compare(size_type pos1, size_type n1, const value_type* s, size_type n2) const;         // constexpr since C++20

#include <cassert>
#include <cstdint>
#include <string>
#include <string_view>

#include "sized_allocator.h"
#include "test_macros.h"

template <class SizeT, class DiffT, class CharT = char, class Traits = std::char_traits<CharT> >
TEST_CONSTEXPR_CXX20 void test_with_custom_size_type() {
  using Alloc  = sized_allocator<CharT, SizeT, DiffT>;
  using string = std::basic_string<CharT, Traits, Alloc>;
  string s     = "hello world";

  // The following tests validate all possible calls to std::min within <basic_string>
  { // basic_string(const basic_string& str, size_type pos, size_type n, const Allocator& a = Allocator())
    assert(string(s, 0, 5) == "hello");
    assert(string(s, 0, 5, Alloc(3)) == "hello");
    assert(string(s, 6, 5) == "world");
    assert(string(s, 6, 5, Alloc(3)) == "world");
    assert(string(s, 6, 100) == "world");
    assert(string(s, 6, 100, Alloc(3)) == "world");
  }
#if TEST_STD_VER >= 23
  { // constexpr basic_string(basic_string&& str, size_type pos, size_type n, const Allocator& a = Allocator());
    assert(string(string(s), 0, 5) == "hello");
    assert(string(string(s), 0, 5, Alloc(3)) == "hello");
    assert(string(string(s), 6, 5) == "world");
    assert(string(string(s), 6, 5, Alloc(3)) == "world");
    assert(string(string(s), 6, 100) == "world");
    assert(string(string(s), 6, 100, Alloc(3)) == "world");
  }
#endif
  { // basic_string& assign(const basic_string& str, size_type pos, size_type n=npos)
    string s1 = s;
    string s2 = "cplusplus";
    s1.assign(s2, 0, 5);
    assert(s1 == "cplus");
    s1.assign(s2, 0, 9);
    assert(s1 == "cplusplus");
    s1.assign(s2, 5);
    assert(s1 == "plus");
    s1.assign(s2, 0, 100);
    assert(s1 == "cplusplus");
    s1.assign(s2, 4);
    assert(s1 == "splus");
    s1.assign(s2, 0);
    assert(s1 == "cplusplus");
  }
#if TEST_STD_VER >= 17
  { // template <class T> basic_string& assign(const T& t, size_type pos, size_type n=npos)
    std::string_view sv = "cplusplus";
    string s1           = s;
    s1.assign(sv, 0, 5);
    assert(s1 == "cplus");
    s1.assign(sv, 0, 9);
    assert(s1 == "cplusplus");
    s1.assign(sv, 5);
    assert(s1 == "plus");
    s1.assign(sv, 0, 100);
    assert(s1 == "cplusplus");
    s1.assign(sv, 4);
    assert(s1 == "splus");
    s1.assign(sv, 0);
    assert(s1 == "cplusplus");
  }
#endif
  { // basic_string& append(const basic_string& str, size_type pos, size_type n=npos)
    string s1 = s;
    string s2 = " of cplusplus";
    s1.append(s2, 0, 5);
    assert(s1 == "hello world of c");
    s1 = s;
    s1.append(s2, 0, 100);
    assert(s1 == "hello world of cplusplus");
    s1 = s;
    s1.append(s2, 0);
    assert(s1 == "hello world of cplusplus");
  }
#if TEST_STD_VER >= 17
  { // template <class T> basic_string& append(const T& t, size_type pos, size_type n=npos)
    string s1           = s;
    std::string_view sv = " of cplusplus";
    s1.append(sv, 0, 5);
    assert(s1 == "hello world of c");
    s1 = s;
    s1.append(sv, 0, 100);
    assert(s1 == "hello world of cplusplus");
    s1 = s;
    s1.append(sv, 0);
    assert(s1 == "hello world of cplusplus");
  }
#endif
  { // basic_string& insert(size_type pos1, const basic_string& str, size_type pos2, size_type n=npos)
    string s1 = s;
    string s2 = " cplusplus";
    s1.insert(5, s2, 0, 2);
    assert(s1 == "hello c world");
    s1 = s;
    s1.insert(5, s2, 0, 100);
    assert(s1 == "hello cplusplus world");
    s1 = s;
    s1.insert(5, s2, 0);
    assert(s1 == "hello cplusplus world");
  }
#if TEST_STD_VER >= 17
  { // template <class T> basic_string& insert(size_type pos1, const T& t, size_type pos2, size_type n=npos)
    string s1           = s;
    std::string_view sv = " cplusplus";
    s1.insert(5, sv, 0, 2);
    assert(s1 == "hello c world");
    s1 = s;
    s1.insert(5, sv, 0, 100);
    assert(s1 == "hello cplusplus world");
    s1 = s;
    s1.insert(5, sv, 0);
    assert(s1 == "hello cplusplus world");
  }
#endif
  { // basic_string& erase(size_type pos = 0, size_type n = npos)
    string s1 = s;
    assert(s1.erase(5, 100) == "hello");
    s1 = s;
    assert(s1.erase(5) == "hello");
    assert(s1.erase().empty());
  }
  { // basic_string& replace(size_type pos, size_type n1, const value_type* s, size_type n2)
    string s1      = s;
    const char* s2 = "cpluscplus";
    assert(s1.replace(6, 5, s2, 1) == "hello c");
    assert(s1.replace(6, 1, s2, 10) == "hello cpluscplus");
  }
  { // basic_string& replace(size_type pos, size_type n1, size_type n2, value_type c)
    string s1 = s;
    assert(s1.replace(5, 6, 2, 'o') == "hellooo");
    assert(s1.replace(5, 2, 0, 'o') == "hello");
  }
  { // basic_string& replace(size_type pos1, size_type n1, const basic_string& str, size_type pos2, size_type n2=npos)
    string s1 = s;
    string s2 = "cplusplus";
    assert(s1.replace(6, 5, s2, 0, 1) == "hello c");
    assert(s1.replace(7, 0, s2, 1, 9) == "hello cplusplus");
    s1 = s;
    assert(s1.replace(6, 5, s2, 0, 100) == "hello cplusplus");
    s1 = s;
    assert(s1.replace(6, 5, s2, 0) == "hello cplusplus");
  }
#if TEST_STD_VER >= 17
  { // template <class T> basic_string& replace(size_type pos1, size_type n1, const T& t, size_type pos2, size_type n2= npos)
    string s1           = s;
    std::string_view sv = "cplusplus";
    assert(s1.replace(6, 5, sv, 0, 1) == "hello c");
    assert(s1.replace(7, 0, sv, 1, 9) == "hello cplusplus");
    s1 = s;
    assert(s1.replace(6, 5, sv, 0, 100) == "hello cplusplus");
    s1 = s;
    assert(s1.replace(6, 5, sv, 0) == "hello cplusplus");
  }
#endif
  { // size_type copy(value_type* s, size_type n, size_type pos = 0) const
    string s1     = s;
    char bar[100] = {};
    s1.copy(bar, s1.size());
    assert(s1 == bar);
  }
  { // int compare(size_type pos1, size_type n1, const value_type* s, size_type n2) const
    string s1 = s;
    assert(s1.compare(0, 11, "hello world", 11) == 0);
    assert(s1.compare(0, 11, "hello", 11) > 0);
    assert(s1.compare(0, 11, "hello world C++", 100) < 0);
  }
#if TEST_STD_VER >= 17
  { // template <class T> int compare(const T& t) const noexcept;
    using std::operator""sv;
    string s1 = s;
    assert(s1.compare("hello world"sv) == 0);
    assert(s1.compare("hello"sv) > 0);
    assert(s1.compare("hello world C++"sv) < 0);
  }
#endif
}

TEST_CONSTEXPR_CXX20 bool test() {
  // test_with_custom_size_type<std::uint8_t, std::int8_t>();
  test_with_custom_size_type<std::uint16_t, std::int16_t>();
  test_with_custom_size_type<std::uint32_t, std::int32_t>();
  test_with_custom_size_type<std::uint64_t, std::int64_t>();
  test_with_custom_size_type<std::size_t, std::ptrdiff_t>();
  // test_with_custom_size_type<unsigned char, int>();
  test_with_custom_size_type<unsigned short, short>();
  test_with_custom_size_type<unsigned, int>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
