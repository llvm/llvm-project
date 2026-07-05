// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify

//--- a.cppm
export module a;

struct Base {
    template <class T>
    friend constexpr auto f(T) { return 0; }
};
export struct A: Base {};

//--- b.cppm
export module b;

import a;

namespace n {

struct B {};

auto b() -> void {
	f(A{});
    f(B{}); // expected-error {{use of undeclared identifier 'f'}}
}

} // namespace n
