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
//     basic_spanbuf& operator=(basic_spanbuf&& rhs);

#include <cassert>
#include <span>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_macros.h"

template <typename CharT, typename TraitsT>
struct TestSpanBuf : std::basic_spanbuf<CharT, TraitsT> {
  using std::basic_spanbuf<CharT, TraitsT>::basic_spanbuf;

  TestSpanBuf(std::basic_spanbuf<CharT, TraitsT>&& rhs_p) : std::basic_spanbuf<CharT, TraitsT>(std::move(rhs_p)) {}

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

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test_postconditions() {
  using SpBuf = std::basic_spanbuf<CharT, TraitsT>;

  // Empty `span`
  {
    // Mode: default (`in` | `out`)
    {
      std::span<CharT> sp;
      TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp);
      TestSpanBuf<CharT, TraitsT> spBuf = std::move(static_cast<SpBuf&>(rhsSpBuf));
      assert(rhsSpBuf.span().data() == nullptr);
      assert(spBuf.span().data() == nullptr);
      spBuf.check_postconditions(rhsSpBuf);
    }
    // Mode: `in`
    {
      std::span<CharT> sp;
      TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::in);
      TestSpanBuf<CharT, TraitsT> spBuf = std::move(static_cast<SpBuf&>(rhsSpBuf));
      assert(rhsSpBuf.span().data() == nullptr);
      assert(spBuf.span().data() == nullptr);
      spBuf.check_postconditions(rhsSpBuf);
    }
    // Mode: `out`
    {
      std::span<CharT> sp;
      TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::out);
      TestSpanBuf<CharT, TraitsT> spBuf = std::move(static_cast<SpBuf&>(rhsSpBuf));
      assert(rhsSpBuf.span().data() == nullptr);
      assert(spBuf.span().data() == nullptr);
      spBuf.check_postconditions(rhsSpBuf);
    }
    // Mode: multiple
    {
      std::span<CharT> sp;
      TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::in | std::ios_base::out | std::ios_base::binary);
      TestSpanBuf<CharT, TraitsT> spBuf = std::move(static_cast<SpBuf&>(rhsSpBuf));
      assert(rhsSpBuf.span().data() == nullptr);
      assert(spBuf.span().data() == nullptr);
      spBuf.check_postconditions(rhsSpBuf);
    }
    // Mode: `ate`
    {
      std::span<CharT> sp;
      TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::out | std::ios_base::ate);
      TestSpanBuf<CharT, TraitsT> spBuf = std::move(static_cast<SpBuf&>(rhsSpBuf));
      assert(rhsSpBuf.span().data() == nullptr);
      assert(spBuf.span().data() == nullptr);
      spBuf.check_postconditions(rhsSpBuf);
    }
  }

  // Non-empty `span`
  {
    CharT arr[4];

    // Mode: default (`in` | `out`)
    {
      std::span<CharT> sp{arr};
      TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp);
      TestSpanBuf<CharT, TraitsT> spBuf = std::move(static_cast<SpBuf&>(rhsSpBuf));
      assert(rhsSpBuf.span().data() == arr);
      assert(spBuf.span().data() == arr);
      spBuf.check_postconditions(rhsSpBuf);
    }
    // Mode: `in`
    {
      std::span<CharT> sp{arr};
      TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::in);
      TestSpanBuf<CharT, TraitsT> spBuf = std::move(static_cast<SpBuf&>(rhsSpBuf));
      assert(rhsSpBuf.span().data() == arr);
      assert(spBuf.span().data() == arr);
      spBuf.check_postconditions(rhsSpBuf);
    }
    // Mode: `out`
    {
      std::span<CharT> sp{arr};
      TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::out);
      TestSpanBuf<CharT, TraitsT> spBuf = std::move(static_cast<SpBuf&>(rhsSpBuf));
      assert(rhsSpBuf.span().data() == arr);
      assert(spBuf.span().data() == arr);
      spBuf.check_postconditions(rhsSpBuf);
    }
    // Mode: multiple
    {
      std::span<CharT> sp{arr};
      TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::in | std::ios_base::out | std::ios_base::binary);
      TestSpanBuf<CharT, TraitsT> spBuf = std::move(static_cast<SpBuf&>(rhsSpBuf));
      assert(rhsSpBuf.span().data() == arr);
      assert(spBuf.span().data() == arr);
      spBuf.check_postconditions(rhsSpBuf);
    }
    // Mode: `ate`
    {
      std::span<CharT> sp{arr};
      TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::out | std::ios_base::ate);
      TestSpanBuf<CharT, TraitsT> spBuf = std::move(static_cast<SpBuf&>(rhsSpBuf));
      assert(rhsSpBuf.span().data() == arr);
      assert(spBuf.span().data() == arr);
      spBuf.check_postconditions(rhsSpBuf);
    }
  }
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpBuf = std::basic_spanbuf<CharT, TraitsT>;

  // Empty `span`
  {
    // Mode: default (`in` | `out`)
    {
      SpBuf rhsSpBuf;
      assert(rhsSpBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);

      SpBuf spBuf = std::move(rhsSpBuf);
      assert(spBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);

      // After move
      assert(rhsSpBuf.span().data() == nullptr);
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);
    }
    // Mode: `in`
    {
      SpBuf rhsSpBuf{std::ios_base::in};
      assert(rhsSpBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);

      SpBuf spBuf = std::move(rhsSpBuf);
      assert(spBuf.span().data() == nullptr);
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);

      // After move
      assert(rhsSpBuf.span().data() == nullptr);
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);
    }
    // Mode: `out`
    {
      SpBuf rhsSpBuf{std::ios_base::out};
      assert(rhsSpBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);

      SpBuf spBuf = std::move(rhsSpBuf);
      assert(spBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);

      // After move
      assert(rhsSpBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);
    }
    // Mode: multiple
    {
      SpBuf rhsSpBuf{std::ios_base::out | std::ios_base::in | std::ios_base::binary};
      assert(rhsSpBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);

      SpBuf spBuf = std::move(rhsSpBuf);
      assert(spBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);

      // After move
      assert(rhsSpBuf.span().data() == nullptr);
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);
    }
    // Mode: `ate`
    {
      SpBuf rhsSpBuf{std::ios_base::ate};
      SpBuf spBuf = std::move(rhsSpBuf);
      assert(spBuf.span().data() == nullptr);
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);

      // After move
      assert(rhsSpBuf.span().data() == nullptr);
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);
    }
  }

  // Non-empty `span`
  {
    CharT arr[4];
    std::span<CharT> sp{arr};

    // Mode: default (`in` | `out`)
    {
      SpBuf rhsSpBuf{sp};
      assert(rhsSpBuf.span().data() == arr);
      // Mode `out` counts read characters
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);

      SpBuf spBuf = std::move(rhsSpBuf);
      assert(spBuf.span().data() == arr);
      // Mode `out` counts read characters
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);

      // After move
      assert(rhsSpBuf.span().data() == arr);
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);
    }
    // Mode: `in`
    {
      SpBuf rhsSpBuf{sp, std::ios_base::in};
      assert(rhsSpBuf.span().data() == arr);
      assert(!rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 4);

      SpBuf spBuf = std::move(rhsSpBuf);
      assert(spBuf.span().data() == arr);
      assert(!spBuf.span().empty());
      assert(spBuf.span().size() == 4);
    }
    // Mode `out`
    {
      SpBuf rhsSpBuf{sp, std::ios_base::out};
      assert(rhsSpBuf.span().data() == arr);
      // Mode `out` counts read characters
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);

      SpBuf spBuf = std::move(rhsSpBuf);
      assert(spBuf.span().data() == arr);
      // Mode `out` counts read characters
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);

      // After move
      assert(rhsSpBuf.span().data() == arr);
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);
    }
    // Mode: multiple
    {
      SpBuf rhsSpBuf{sp, std::ios_base::out | std::ios_base::in | std::ios_base::binary};
      assert(rhsSpBuf.span().data() == arr);
      // Mode `out` counts read characters
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);

      SpBuf spBuf = std::move(rhsSpBuf);
      assert(spBuf.span().data() == arr);
      // Mode `out` counts read characters
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);

      // After move
      assert(rhsSpBuf.span().data() == arr);
      assert(rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 0);
    }
    // Mode: `ate`
    {
      SpBuf rhsSpBuf{sp, std::ios_base::ate};
      assert(rhsSpBuf.span().data() == arr);
      assert(!rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 4);

      SpBuf spBuf = std::move(rhsSpBuf);
      assert(spBuf.span().data() == arr);
      assert(!spBuf.span().empty());
      assert(spBuf.span().size() == 4);

      // After move
      assert(rhsSpBuf.span().data() == arr);
      assert(!rhsSpBuf.span().empty());
      assert(rhsSpBuf.span().size() == 4);
    }
  }
}

int main(int, char**) {
#ifndef TEST_HAS_NO_NASTY_STRING
  test_postconditions<nasty_char, nasty_char_traits>();
#endif
  test_postconditions<char>();
  test_postconditions<char, constexpr_char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_postconditions<wchar_t>();
  test_postconditions<wchar_t, constexpr_char_traits<wchar_t>>();
#endif
#ifndef TEST_HAS_NO_NASTY_STRING
  test<nasty_char, nasty_char_traits>();
#endif
  test<char>();
  test<char, constexpr_char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
  test<wchar_t, constexpr_char_traits<wchar_t>>();
#endif

  return 0;
}
