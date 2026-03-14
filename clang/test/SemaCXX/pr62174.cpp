// RUN: %clang_cc1 -std=c++20 %s -fsyntax-only -verify
// expected-no-diagnostics
namespace lib {
    namespace impl {
        template <class>
        inline constexpr bool test = false;
    }
    using impl::test;
}

struct foo {};

template <>
inline constexpr bool lib::test<foo> = true;

static_assert(lib::test<foo>);
