// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++23 mod1.cppm -emit-module-interface -o mod1.pcm -fallow-pcm-with-compiler-errors -verify
// RUN: %clang_cc1 -std=c++23 mod2.cppm -emit-module-interface -o mod2.pcm -fmodule-file=mod1=mod1.pcm -verify -fallow-pcm-with-compiler-errors
// RUN: %clang_cc1 -std=c++23 mod3.cppm -emit-module-interface -o mod3.pcm -fmodule-file=mod1=mod1.pcm -fmodule-file=mod2=mod2.pcm -verify -fallow-pcm-with-compiler-errors
// RUN: %clang_cc1 -std=c++23 main.cpp -fmodule-file=mod1=mod1.pcm -fmodule-file=mod2=mod2.pcm -fmodule-file=mod3=mod3.pcm -verify -fallow-pcm-with-compiler-errors -ast-dump-all

//--- mod1.cppm
export module mod1;

export template <unsigned N, unsigned M>
class A {
public:
  constexpr A(const char[], const char[]) {
    auto x = BrokenExpr; // expected-error {{use of undeclared identifier 'BrokenExpr'}}
  }
};

export template<A<1,1> NTTP>
struct B {};

template < unsigned N, unsigned M >
A(const char (&)[N], const char (&)[M]) -> A< 1, 1 >;

//--- mod2.cppm
export module mod2;
import mod1;

struct C: B <A{"a", "b"}> { // expected-error {{non-type template argument is not a constant expression}}
  constexpr C(int a) { }
};

//--- mod3.cppm
// expected-no-diagnostics
export module mod3;
export import mod2;

//--- main.cpp
// expected-no-diagnostics
import mod3; // no crash
