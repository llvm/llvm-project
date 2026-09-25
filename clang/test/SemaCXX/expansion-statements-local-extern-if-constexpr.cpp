// RUN: %clang_cc1 %s -I %S/Inputs -std=c++2c -fsyntax-only -verify
// XFAIL: *

// FIXME: This test currently asserts due to a bug not related to expansion statements:
// https://github.com/llvm/llvm-project/issues/223003. Reenable it once that bug is fixed.

int foo8_decl; // #foo8_decl
void foo8() {
  template for (constexpr auto x : {true, false}) { // #foo8_instantiation
    if constexpr (x) {
      extern int foo8_decl;
    } else {
      extern thread_local int foo8_decl; // #mismatched_foo8_decl
      // expected-error@#mismatched_foo8_decl {{thread-local declaration of 'foo8_decl' follows non-thread-local declaration}}
      // expected-note@#foo8_instantiation {{in instantiation of expansion statement requested here}}
      // expected-note@#foo8_decl {{previous definition is here}}
    }
  }
}
