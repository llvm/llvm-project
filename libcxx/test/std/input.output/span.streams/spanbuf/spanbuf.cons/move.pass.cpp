//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <spanstream>

//   template<class charT, class traits = char_traits<charT>>
//   class basic_spanbuf
//     : public basic_streambuf<charT, traits> {

//     // [spanbuf.cons], constructors
//
//     basic_spanbuf(basic_spanbuf&& rhs);

#include <cassert>
#include <concepts>
#include <span>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_convertible.h"
#include "test_macros.h"

template <typename CharT, typename TraitsT>
struct TestSpanBuf : std::basic_spanbuf<CharT, TraitsT> {
  using std::basic_spanbuf<CharT, TraitsT>::basic_spanbuf;

  TestSpanBuf(std::basic_spanbuf<CharT, TraitsT>&& rhs) : std::basic_spanbuf<CharT, TraitsT>(std::move(rhs)) {}

  void check_postconditions(TestSpanBuf<CharT, TraitsT> const& rhs_p) const {
    assert(this->span().data() == rhs_p.span().data());
    assert(this->span().size() == rhs_p.span().size());
    assert(this->eback() == rhs_p.eback());
    assert(this->gptr() == rhs_p.gptr());
    assert(this->egptr() == rhs_p.egptr());
    assert(this->pbase() == rhs_p.pbase());
    assert(this->pptr() == rhs_p.pptr());
    assert(this->epptr() == rhs_p.epptr());
    assert(this->getloc() == rhs_p.getloc());
  }
};

void test_sfinae_with_nasty_char() {
  using SpBuf = std::basic_spanbuf<nasty_char, nasty_char_traits>;

  static_assert(std::move_constructible<SpBuf>);
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test_sfinae() {
  using SpBuf = std::basic_spanbuf<CharT, TraitsT>;

  static_assert(std::move_constructible<SpBuf>);
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpBuf = std::basic_spanbuf<CharT, TraitsT>;

  {
    // Empty `span`
    {
      // Mode: `ios_base::in`
      {
        SpBuf rhsSpBuf{std::ios_base::in};
        SpBuf spBuf(std::move(rhsSpBuf));
        assert(spBuf.span().data() == nullptr);
        assert(spBuf.span().empty());
        assert(spBuf.span().size() == 0);
      }
      // Mode: `ios_base::out`
      {
        SpBuf rhsSpBuf{std::ios_base::out};
        SpBuf spBuf(std::move(rhsSpBuf));
        assert(spBuf.span().data() == nullptr);
        assert(spBuf.span().empty());
        assert(spBuf.span().size() == 0);
      }
      // Mode: multiple
      {
        SpBuf rhsSpBuf{std::ios_base::out | std::ios_base::in};
        SpBuf spBuf(std::move(rhsSpBuf));
        assert(spBuf.span().data() == nullptr);
        assert(spBuf.span().empty());
        assert(spBuf.span().size() == 0);
      }
    }

    // Non-empty `span`
    {
      CharT arr[4];
      std::span<CharT> sp{arr};

      // Mode: `ios_base::in`
      {
        SpBuf rhsSpBuf{sp, std::ios_base::in};
        SpBuf spBuf(std::move(rhsSpBuf));
        assert(spBuf.span().data() == arr);
        assert(!spBuf.span().empty());
        assert(spBuf.span().size() == 4);
      }
      // Mode `ios_base::out`
      {
        SpBuf rhsSpBuf{sp, std::ios_base::out};
        SpBuf spBuf(std::move(rhsSpBuf));
        assert(spBuf.span().data() == arr);
        assert(spBuf.span().empty());
        assert(spBuf.span().size() == 0);
      }
      // Mode: multiple
      {
        SpBuf rhsSpBuf{sp, std::ios_base::out | std::ios_base::in | std::ios_base::binary};
        SpBuf spBuf(std::move(rhsSpBuf));
        assert(spBuf.span().data() == arr);
        assert(spBuf.span().empty());
        assert(spBuf.span().size() == 0);
      }
    }
  }

  // Check post-conditions
  {
    // Empty `span`
    {
      {
        std::span<CharT> sp;
        TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp);
        TestSpanBuf<CharT, TraitsT> spBuf(std::move(static_cast<SpBuf&>(rhsSpBuf)));
        assert(rhsSpBuf.span().empty());
        assert(spBuf.span().data() == nullptr);
        spBuf.check_postconditions(rhsSpBuf);
      }
      // Mode: `ios_base::in`
      {
        std::span<CharT> sp;
        TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::in);
        TestSpanBuf<CharT, TraitsT> spBuf(std::move(static_cast<SpBuf&>(rhsSpBuf)));
        assert(rhsSpBuf.span().empty());
        assert(spBuf.span().data() == nullptr);
        spBuf.check_postconditions(rhsSpBuf);
      }
      // Mode: `ios_base::out`
      {
        std::span<CharT> sp;
        TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::out);
        TestSpanBuf<CharT, TraitsT> spBuf(std::move(static_cast<SpBuf&>(rhsSpBuf)));
        assert(rhsSpBuf.span().empty());
        assert(spBuf.span().data() == nullptr);
        spBuf.check_postconditions(rhsSpBuf);
      }
      // Mode: multiple
      {
        std::span<CharT> sp;
        TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::out | std::ios_base::in | std::ios_base::binary);
        TestSpanBuf<CharT, TraitsT> spBuf(std::move(static_cast<SpBuf&>(rhsSpBuf)));
        assert(rhsSpBuf.span().empty());
        assert(spBuf.span().data() == nullptr);
        spBuf.check_postconditions(rhsSpBuf);
      }
    }

    // Non-empty `span`
    {
      CharT arr[4];

      {
        std::span<CharT> sp{arr};
        TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp);
        TestSpanBuf<CharT, TraitsT> spBuf(std::move(static_cast<SpBuf&>(rhsSpBuf)));
        assert(rhsSpBuf.span().empty());
        assert(spBuf.span().data() == arr);
        spBuf.check_postconditions(rhsSpBuf);
      }
      // Mode: `ios_base::in`
      {
        std::span<CharT> sp{arr};
        TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::in);
        TestSpanBuf<CharT, TraitsT> spBuf(std::move(static_cast<SpBuf&>(rhsSpBuf)));
        assert(!rhsSpBuf.span().empty());
        assert(spBuf.span().data() == arr);
        spBuf.check_postconditions(rhsSpBuf);
      }
      // Mode: `ios_base::out`
      {
        std::span<CharT> sp{arr};
        TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::out);
        TestSpanBuf<CharT, TraitsT> spBuf(std::move(static_cast<SpBuf&>(rhsSpBuf)));
        assert(rhsSpBuf.span().empty());
        assert(spBuf.span().data() == arr);
        spBuf.check_postconditions(rhsSpBuf);
      }
      // Mode: multiple
      {
        std::span<CharT> sp{arr};
        TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::out | std::ios_base::in | std::ios_base::binary);
        TestSpanBuf<CharT, TraitsT> spBuf(std::move(static_cast<SpBuf&>(rhsSpBuf)));
        assert(rhsSpBuf.span().empty());
        assert(spBuf.span().data() == arr);
        spBuf.check_postconditions(rhsSpBuf);
      }
    }
  }
}

int main(int, char**) {
  test_sfinae_with_nasty_char();
  test_sfinae<char>();
  test_sfinae<char, constexpr_char_traits<char>>();
  test<char>();
  test<char, constexpr_char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
  test<wchar_t, constexpr_char_traits<wchar_t>>();
#endif

  return 0;
}
