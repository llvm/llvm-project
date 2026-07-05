// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux -emit-module-interface %t/a.cppm -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux -emit-module-interface -fprebuilt-module-path=%t %t/b.cppm -o %t/B.pcm

// Just check that this doesn't crash.

//--- a.cppm
module;

template <typename _Visitor>
void __do_visit(_Visitor &&__visitor) {
  using _V0 = int;
  [](_V0 __v) -> _V0 { return __v; } (1);
}

export module A;

void g() {
  struct Visitor { };
  __do_visit(Visitor());
}

//--- b.cppm
module;

template <typename _Visitor>
void __do_visit(_Visitor &&__visitor) {
  using _V0 = int;

  // Check that we instantiate this lambda's call operator in 'f' below
  // instead of the one in 'a.cppm' here; otherwise, we won't find a
  // corresponding instantiation of the using declaration above.
  [](_V0 __v) -> _V0 { return __v; } (1);
}

export module B;
import A;

void f() {
  __do_visit(1);
}
