//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/filesystem>

// class path

// template <class Source>
//      path& operator=(Source const&);
// template <class Source>
//      path& assign(Source const&);
// template <class InputIterator>
//      path& assign(InputIterator first, InputIterator last);


#include <experimental/filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.hpp"
#include "filesystem_test_helper.hpp"
#include <iostream>

namespace fs = std::experimental::filesystem;

template <class CharT>
void RunTestCase(MultiStringType const& MS) {
  using namespace fs;
  const char* Expect = MS;
  const CharT* TestPath = MS;
  const CharT* TestPathEnd = StrEnd(TestPath);
  const std::size_t Size = TestPathEnd - TestPath;
  const std::size_t SSize = StrEnd(Expect) - Expect;
  assert(Size == SSize);
  //////////////////////////////////////////////////////////////////////////////
  // basic_string<Char, Traits, Alloc>
  {
    const std::basic_string<CharT> S(TestPath);
    path p; PathReserve(p, S.length() + 1);
    {
      // string provides a contigious iterator. No allocation needed.
      DisableAllocationGuard g;
      path& pref = (p = S);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
    assert(p.string<CharT>() == S);
  }
  {
    const std::basic_string<CharT> S(TestPath);
    path p; PathReserve(p, S.length() + 1);
    {
      DisableAllocationGuard g;
      path& pref = p.assign(S);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
    assert(p.string<CharT>() == S);
  }
  //////////////////////////////////////////////////////////////////////////////
  // Char* pointers
  {
    path p; PathReserve(p, Size + 1);
    {
      // char* pointers are contigious and can be used with code_cvt directly.
      // no allocations needed.
      DisableAllocationGuard g;
      path& pref = (p = TestPath);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
  {
    path p; PathReserve(p, Size + 1);
    {
      DisableAllocationGuard g;
      path& pref = p.assign(TestPath);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
  {
    path p; PathReserve(p, Size + 1);
    {
      DisableAllocationGuard g;
      path& pref = p.assign(TestPath, TestPathEnd);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
  //////////////////////////////////////////////////////////////////////////////
  // Iterators
  {
    using It = input_iterator<const CharT*>;
    path p; PathReserve(p, Size + 1);
    It it(TestPath);
    {
      // Iterators cannot be used with code_cvt directly. This assignment
      // may allocate if it's larger than a "short-string".
      path& pref = (p = it);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
  {
    using It = input_iterator<const CharT*>;
    path p; PathReserve(p, Size + 1);
    It it(TestPath);
    {
      path& pref = p.assign(it);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
  {
    using It = input_iterator<const CharT*>;
    path p; PathReserve(p, Size + 1);
    It it(TestPath);
    It e(TestPathEnd);
    {
      path& pref = p.assign(it, e);
      assert(&pref == &p);
    }
    assert(p.native() == Expect);
    assert(p.string<CharT>() == TestPath);
  }
}

int main() {
  for (auto const& MS : PathList) {
    RunTestCase<char>(MS);
    RunTestCase<wchar_t>(MS);
    RunTestCase<char16_t>(MS);
    RunTestCase<char32_t>(MS);
  }
}
