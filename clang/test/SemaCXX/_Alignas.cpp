// RUN: %clang_cc1 %s -fsyntax-only -verify=expected,cpp
// RUN: %clang_cc1 -x c %s -fsyntax-only -verify=expected,c

// Ensure that we correctly parse _Alignas as an extension in C++.
_Alignas(64) int i1;
_Alignas(long long) int i2;
int volatile _Alignas(64) i3; // Test strange ordering

void foo(void) {
  // We previously rejected these valid declarations.
  _Alignas(8) char i4;
  _Alignas(char) char i5;

  (void)(int _Alignas(64))0; // expected-warning {{'_Alignas' attribute ignored when parsing type}}
  // FIXME: C and C++ should both diagnose the same way, as being ignored.
  (void)(_Alignas(64) int)0; // c-error {{expected expression}} \
                                cpp-warning {{'_Alignas' attribute ignored when parsing type}}
}

struct S {
  _Alignas(int) int i;
  _Alignas(64) int j;
};

void bar(_Alignas(8) char c1, char _Alignas(char) c2); // expected-error 2 {{'_Alignas' attribute cannot be applied to a function parameter}}
