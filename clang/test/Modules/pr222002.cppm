// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/mod.cppm -emit-module-interface -o %t/mod.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fmodule-file=mod=%t/mod.pcm \
// RUN:   -fsyntax-only -verify
//
// Test  again with reduced BMI
// RUN: %clang_cc1 -std=c++20 %t/mod.cppm -emit-reduced-module-interface \
// RUN:   -o %t/mod.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fmodule-file=mod=%t/mod.pcm \
// RUN:   -fsyntax-only -verify

//--- mod.cppm
export module mod;

export template <typename T> struct dependent {
  dependent(int) {}
};

template <typename U> dependent(U) -> dependent<U>;

export template <typename T> struct independent {
  independent(int) {}
};

independent(int) -> independent<int>;

export template <int N> struct nontype {
  nontype(int) {}
};

nontype(int) -> nontype<1>;

template <typename T> struct wrapped {
  wrapped(int) {}
};

template <typename U> wrapped(U) -> wrapped<U>;

export template <bool> void use_it() { wrapped d(1); }

//--- use.cpp
// expected-no-diagnostics
import mod;

void use() {
  dependent d(1);
  independent i(1);
  nontype n(1);
  use_it<true>();
}
