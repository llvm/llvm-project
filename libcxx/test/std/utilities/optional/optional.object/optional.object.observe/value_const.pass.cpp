//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <optional>

// constexpr const T& optional<T>::value() const &;

#include <cassert>
#include <memory>
#include <optional>

#include "test_macros.h"
#if TEST_STD_VER >= 26
#  include "copy_move_types.h"
#endif

using std::optional;
using std::in_place_t;
using std::in_place;
using std::bad_optional_access;

struct X
{
    X() = default;
    X(const X&) = delete;
    constexpr int test() const & {return 3;}
    int test() & {return 4;}
    constexpr int test() const && {return 5;}
    int test() && {return 6;}
};

#if TEST_STD_VER >= 26
constexpr bool test_ref() {
  {
    TracedCopyMove x{};
    const std::optional<TracedCopyMove&> opt(x);
    ASSERT_NOT_NOEXCEPT(opt.value());
    ASSERT_SAME_TYPE(decltype(opt.value()), TracedCopyMove&);

    assert(std::addressof(opt.value()) == std::addressof(x));
    assert(opt->constMove == 0);
    assert(opt->nonConstMove == 0);
    assert(opt->constCopy == 0);
    assert(opt->nonConstCopy == 0);
  }

  {
    TracedCopyMove x{};
    const std::optional<const TracedCopyMove&> opt(x);
    ASSERT_NOT_NOEXCEPT(opt.value());
    ASSERT_SAME_TYPE(decltype(opt.value()), const TracedCopyMove&);

    assert(std::addressof(opt.value()) == std::addressof(x));
    assert(opt->constMove == 0);
    assert(opt->nonConstMove == 0);
    assert(opt->constCopy == 0);
    assert(opt->nonConstCopy == 0);
  }

  return true;
}
#endif

int main(int, char**)
{
    {
        const optional<X> opt; ((void)opt);
        ASSERT_NOT_NOEXCEPT(opt.value());
        ASSERT_SAME_TYPE(decltype(opt.value()), X const&);
    }
    {
        constexpr optional<X> opt(in_place);
        static_assert(opt.value().test() == 3, "");
    }
    {
        const optional<X> opt(in_place);
        assert(opt.value().test() == 3);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        const optional<X> opt;
        try
        {
            (void)opt.value();
            assert(false);
        }
        catch (const bad_optional_access&)
        {
        }
    }
#endif

#if TEST_STD_VER >= 26
    assert(test_ref());
    static_assert(test_ref());
#endif

    return 0;
}
