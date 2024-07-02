// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++20 std-10-5-ex1-interface.cpp \
// RUN: -DBAD_FWD_DECL  -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 -emit-module-interface std-10-5-ex1-interface.cpp \
// RUN: -o A.pcm

// RUN: %clang_cc1 -std=c++20 std-10-5-ex1-use.cpp -fmodule-file=A=A.pcm \
// RUN:    -fsyntax-only -verify

// Test again with reduced BMI.
// RUN: rm A.pcm
// RUN: %clang_cc1 -std=c++20 std-10-5-ex1-interface.cpp \
// RUN: -DBAD_FWD_DECL  -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface std-10-5-ex1-interface.cpp \
// RUN: -o A.pcm

// RUN: %clang_cc1 -std=c++20 std-10-5-ex1-use.cpp -fmodule-file=A=A.pcm \
// RUN:    -fsyntax-only -verify


//--- std-10-5-ex1-interface.cpp

export module A;
#ifdef BAD_FWD_DECL
export inline void fn_e(); // expected-error {{inline function not defined before the private module fragment}}
                           // expected-note@std-10-5-ex1-interface.cpp:21 {{private module fragment begins here}}
#endif
export inline void ok_fn() {}
export inline void ok_fn2();
#ifdef BAD_FWD_DECL
inline void fn_m(); // expected-error {{inline function not defined before the private module fragment}}
                    // expected-note@std-10-5-ex1-interface.cpp:21 {{private module fragment begins here}}
#endif
static void fn_s();
export struct X;
export void g(X *x) {
  fn_s();
}
export X *factory();
void ok_fn2() {}

module :private;
struct X {};
X *factory() {
  return new X();
}

void fn_e() {}
void fn_m() {}
void fn_s() {}

//--- std-10-5-ex1-use.cpp

import A;

void foo() {
  X x; // expected-error 1+{{missing '#include'; 'X' must be defined before it is used}}
       // expected-note@std-10-5-ex1-interface.cpp:22 1+{{definition here is not reachable}}
  X *p = factory();
}
