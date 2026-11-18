// RUN: %clang_cc1 -fsyntax-only -verify %s
// This test checks that we don't crash when parsing malformed _Atomic types
// with nested sizeof/alignof expressions. See GitHub issue #173886.

int a[100];
int main() {
  a[__alignof__(_Atomic(void) _Atomic double unsigned)]; // expected-error {{cannot combine with previous '_Atomic' declaration specifier}} \
                                                         // expected-error {{'_Atomic' cannot be signed or unsigned}} \
                                                         // expected-error {{_Atomic cannot be applied to incomplete type 'void'}}
}
