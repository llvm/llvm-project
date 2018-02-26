// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -DUNSIGNED -verify -Wsign-compare %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -DSIGNED -verify -Wsign-compare %s
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -DUNSIGNED -DSILENCE -verify %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -DSIGNED -DSILENCE -verify %s

int main() {
  enum A { A_a = 0, A_b = 1 };
  static const int message[] = {0, 1};
  enum A a;

  if (a < 2)
    return 0;

#if defined(SIGNED) && !defined(SILENCE)
  if (a < sizeof(message)/sizeof(message[0])) // expected-warning {{comparison of integers of different signs: 'enum A' and 'unsigned long long'}}
    return 0;
#else
  // expected-no-diagnostics
  if (a < 2U)
    return 0;
  if (a < sizeof(message)/sizeof(message[0]))
    return 0;
#endif
}
