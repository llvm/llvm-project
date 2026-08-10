// RUN: not %clang_cc1 -fsyntax-only -verify -std=gnu++20 -ferror-limit 19 %s
// Creduced test case for the crash in RemoveNestedImmediateInvocation after compliation errros.

a, class b {                               template < typename c>                 consteval b(c
} template <typename...> using d = b;
auto e(d<>) -> int:;
}
f
}
g() {
                    auto h = "":(::i(e(h))
