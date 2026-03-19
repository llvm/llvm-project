// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++23 m.cppm -emit-module-interface -o m.pcm -fallow-pcm-with-compiler-errors -verify
// RUN: %clang_cc1 -std=c++23 main.cpp -fmodule-file=m=m.pcm -verify -fallow-pcm-with-compiler-errors -verify

// RUN: %clang_cc1 -std=c++23 m.cppm -fmodules-reduced-bmi -emit-module-interface -o m.pcm -fallow-pcm-with-compiler-errors -verify
// RUN: %clang_cc1 -std=c++23 main.cpp -fmodule-file=m=m.pcm -verify -fallow-pcm-with-compiler-errors -verify

//--- m.cppm
export module m;

export int f() {
  return 0;
}

export struct Foo {
  __Int bar; // expected-error {{unknown type name '__Int'}}
};

//--- main.cpp
// expected-no-diagnostics
import m; // ok

static_assert(__is_same(decltype(f), int())); // ok
