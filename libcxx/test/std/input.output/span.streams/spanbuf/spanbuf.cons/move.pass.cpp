//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <spanstream>

//   template<class charT, class traits = char_traits<charT>>
//   class basic_spanbuf
//     : public basic_streambuf<charT, traits> {

//     // [spanbuf.cons], constructors
//
//     basic_spanbuf(basic_spanbuf&& rhs);

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
  using SpanBuf = std::basic_spanbuf<CharT, TraitsT>;

  // Empty `span`
  {
    // Mode: default
    {
      std::span<CharT> sp;
      TestSpanBuf<CharT, TraitsT> rhsSpanBuf(sp);
      TestSpanBuf<CharT, TraitsT> spanBuf(std::move(static_cast<SpanBuf&>(rhsSpanBuf)));
      assert(rhsSpanBuf.span().data() == nullptr);
      assert(spanBuf.span().data() == nullptr);
      spanBuf.check_postconditions(rhsSpanBuf);
    }
    // Mode: `in`
    {
      std::span<CharT> sp;
      TestSpanBuf<CharT, TraitsT> rhsSpanBuf(sp, std::ios_base::in);
      TestSpanBuf<CharT, TraitsT> spanBuf(std::move(static_cast<SpanBuf&>(rhsSpanBuf)));
      assert(rhsSpanBuf.span().data() == nullptr);
      assert(spanBuf.span().data() == nullptr);
      spanBuf.check_postconditions(rhsSpanBuf);
    }
    // Mode: `out`
    {
      std::span<CharT> sp;
      TestSpanBuf<CharT, TraitsT> rhsSpanBuf(sp, std::ios_base::out);
      TestSpanBuf<CharT, TraitsT> spanBuf(std::move(static_cast<SpanBuf&>(rhsSpanBuf)));
      assert(rhsSpanBuf.span().data() == nullptr);
      assert(spanBuf.span().data() == nullptr);
      spanBuf.check_postconditions(rhsSpanBuf);
    }
    // Mode: multiple
    {
      std::span<CharT> sp;
      TestSpanBuf<CharT, TraitsT> rhsSpanBuf(sp, std::ios_base::in | std::ios_base::out | std::ios_base::binary);
      TestSpanBuf<CharT, TraitsT> spanBuf(std::move(static_cast<SpanBuf&>(rhsSpanBuf)));
      assert(rhsSpanBuf.span().data() == nullptr);
      assert(spanBuf.span().data() == nullptr);
      spanBuf.check_postconditions(rhsSpanBuf);
    }
    // Mode: `ate`
    {
      std::span<CharT> sp;
      TestSpanBuf<CharT, TraitsT> rhsSpanBuf(sp, std::ios_base::out | std::ios_base::ate);
      TestSpanBuf<CharT, TraitsT> spanBuf(std::move(static_cast<SpanBuf&>(rhsSpanBuf)));
      assert(rhsSpanBuf.span().data() == nullptr);
      assert(spanBuf.span().data() == nullptr);
      spanBuf.check_postconditions(rhsSpanBuf);
    }
  }

  // Non-empty `span`
  {
    CharT arr[4];

    // Mode: default
    {
      std::span<CharT> sp{arr};
      TestSpanBuf<CharT, TraitsT> rhsSpanBuf(sp);
      TestSpanBuf<CharT, TraitsT> spanBuf(std::move(static_cast<SpanBuf&>(rhsSpanBuf)));
      assert(rhsSpanBuf.span().data() == arr);
      assert(spanBuf.span().data() == arr);
      spanBuf.check_postconditions(rhsSpanBuf);
    }
    // Mode: `in`
    {
      std::span<CharT> sp{arr};
      TestSpanBuf<CharT, TraitsT> rhsSpanBuf(sp, std::ios_base::in);
      TestSpanBuf<CharT, TraitsT> spanBuf(std::move(static_cast<SpanBuf&>(rhsSpanBuf)));
      assert(rhsSpanBuf.span().data() == nullptr);
      assert(spanBuf.span().data() == arr);
      // spanBuf.check_postconditions(rhsSpanBuf);
    }
    // Mode: `out`
    {
      std::span<CharT> sp{arr};
      TestSpanBuf<CharT, TraitsT> rhsSpanBuf(sp, std::ios_base::out);
      TestSpanBuf<CharT, TraitsT> spanBuf(std::move(static_cast<SpanBuf&>(rhsSpanBuf)));
      assert(rhsSpanBuf.span().data() == arr);
      assert(spanBuf.span().data() == arr);
      spanBuf.check_postconditions(rhsSpanBuf);
    }
    // Mode: multiple
    {
      std::span<CharT> sp{arr};
      TestSpanBuf<CharT, TraitsT> rhsSpanBuf(sp, std::ios_base::in | std::ios_base::out | std::ios_base::binary);
      TestSpanBuf<CharT, TraitsT> spanBuf(std::move(static_cast<SpanBuf&>(rhsSpanBuf)));
      assert(rhsSpanBuf.span().data() == arr);
      assert(spanBuf.span().data() == arr);
      spanBuf.check_postconditions(rhsSpanBuf);
    }
    // Mode: `ate`
    {
      std::span<CharT> sp{arr};
      TestSpanBuf<CharT, TraitsT> rhsSpanBuf(sp, std::ios_base::out | std::ios_base::ate);
      TestSpanBuf<CharT, TraitsT> spanBuf(std::move(static_cast<SpanBuf&>(rhsSpanBuf)));
      assert(rhsSpanBuf.span().data() == arr);
      assert(spanBuf.span().data() == arr);
      // spanBuf.check_postconditions(rhsSpanBuf);
    }
  }
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpanBuf = std::basic_spanbuf<CharT, TraitsT>;

  // Empty `span`
  {
    // Mode: default (`in` | `out`)
    {
      SpanBuf rhsSpanBuf;
      assert(rhsSpanBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(rhsSpanBuf.span().size() == 0);

      SpanBuf spanBuf{std::move(rhsSpanBuf)};
      assert(spanBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(spanBuf.span().size() == 0);

      // Test after move
      assert(rhsSpanBuf.span().data() == nullptr);
      assert(rhsSpanBuf.span().size() == 0);
    }
    // Mode: `in`
    {
      SpanBuf rhsSpanBuf{std::ios_base::in};
      assert(rhsSpanBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(rhsSpanBuf.span().size() == 0);

      SpanBuf spanBuf{std::move(rhsSpanBuf)};
      assert(spanBuf.span().data() == nullptr);

      assert(spanBuf.span().size() == 0);

      // Test after move
      assert(rhsSpanBuf.span().data() == nullptr);
      assert(rhsSpanBuf.span().size() == 0);
    }
    // Mode: `out`
    {
      SpanBuf rhsSpanBuf{std::ios_base::out};
      assert(rhsSpanBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(rhsSpanBuf.span().size() == 0);
      SpanBuf spanBuf{std::move(rhsSpanBuf)};
      assert(spanBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(spanBuf.span().size() == 0);

      // Test after move
      assert(rhsSpanBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(rhsSpanBuf.span().size() == 0);
    }
    // Mode: `ate`
    {
      SpanBuf rhsSpanBuf{std::ios_base::ate};
      SpanBuf spanBuf{std::move(rhsSpanBuf)};
      assert(spanBuf.span().data() == nullptr);

      assert(spanBuf.span().size() == 0);

      // Test after move
      assert(rhsSpanBuf.span().data() == nullptr);
      assert(rhsSpanBuf.span().size() == 0);
    }
    // Mode: multiple
    {
      SpanBuf rhsSpanBuf{std::ios_base::out | std::ios_base::in | std::ios_base::ate};
      SpanBuf spanBuf{std::move(rhsSpanBuf)};
      assert(spanBuf.span().data() == nullptr);
      assert(spanBuf.span().size() == 0);

      // Test after move
      assert(rhsSpanBuf.span().data() == nullptr);
      assert(rhsSpanBuf.span().size() == 0);
    }
  }

  // Non-empty `span`
  {
    CharT arr[4];
    std::span<CharT> sp{arr};

    // Mode: default (`in` | `out`)
    {
      SpanBuf rhsSpanBuf{sp};
      assert(rhsSpanBuf.span().data() == arr);
      // Mode `out` counts read characters
      assert(rhsSpanBuf.span().size() == 0);

      SpanBuf spanBuf{std::move(rhsSpanBuf)};
      assert(spanBuf.span().data() == arr);
      // Mode `out` counts read characters
      assert(spanBuf.span().size() == 0);

      // Test after move
      assert(rhsSpanBuf.span().data() == arr);
      assert(rhsSpanBuf.span().size() == 0);
    }
    // Mode: `in`
    {
      SpanBuf rhsSpanBuf{sp, std::ios_base::in};
      assert(rhsSpanBuf.span().data() == arr);
      assert(rhsSpanBuf.span().size() == 4);

      SpanBuf spanBuf{std::move(rhsSpanBuf)};
      assert(spanBuf.span().data() == arr);
      assert(spanBuf.span().size() == 4);

      // Test after move
      assert(rhsSpanBuf.span().data() == nullptr);
      assert(rhsSpanBuf.span().size() == 0);
    }
    // Mode `out`
    {
      SpanBuf rhsSpanBuf{sp, std::ios_base::out};
      assert(rhsSpanBuf.span().data() == arr);
      // Mode `out` counts read characters
      assert(rhsSpanBuf.span().size() == 0);

      SpanBuf spanBuf{std::move(rhsSpanBuf)};
      assert(spanBuf.span().data() == arr);
      // Mode `out` counts read characters
      assert(spanBuf.span().size() == 0);

      // Test after move
      assert(rhsSpanBuf.span().data() == arr);
      assert(rhsSpanBuf.span().size() == 0);
    }
    // Mode: `ate`
    {
      SpanBuf rhsSpanBuf{sp, std::ios_base::ate};
      assert(rhsSpanBuf.span().data() == arr);
      assert(rhsSpanBuf.span().size() == 4);

      SpanBuf spanBuf{std::move(rhsSpanBuf)};
      assert(spanBuf.span().data() == arr);
      assert(spanBuf.span().size() == 4);

      // Test after move
      assert(rhsSpanBuf.span().data() == nullptr);
      assert(rhsSpanBuf.span().size() == 0);
    }
    // Mode: `ate`
    {
      SpanBuf rhsSpanBuf{sp, std::ios_base::out | std::ios_base::ate};
      assert(rhsSpanBuf.span().data() == arr);
      assert(rhsSpanBuf.span().size() == 4);

      SpanBuf spanBuf{std::move(rhsSpanBuf)};
      assert(spanBuf.span().data() == arr);
      assert(spanBuf.span().size() == 4);

      // Test after move
      assert(rhsSpanBuf.span().data() == arr);
      assert(rhsSpanBuf.span().size() == 4);
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
