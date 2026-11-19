// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DCHAR_REDECL -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DMODE_T_CHAR -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DMODE_T_CHAR -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DMODE_T_BOOL -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DMODE_T_BITINT9 -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DMODE_T_BITINT10 -verify

// Regression tests for GH244171: calling umask with a parameter type narrower
// than 10 bits must not assert when constructing the 0777 bitmask APInt.

#if defined(CHAR_REDECL)
int umask(char);
void foo(void) {
  umask(0xFFFF); // expected-warning {{implicit conversion from 'int' to 'char' changes value from 65535 to -1}}
}
#elif defined(MODE_T_CHAR)
// expected-no-diagnostics
typedef unsigned char mode_t;
#ifdef __cplusplus
extern "C"
#endif
mode_t umask(mode_t);
void foo(void) {
  umask(0);
  umask(0xFF);
}
#elif defined(MODE_T_BOOL)
// expected-no-diagnostics
typedef _Bool mode_t;
mode_t umask(mode_t);
void foo(void) { umask(1); }
#elif defined(MODE_T_BITINT9)
// expected-no-diagnostics
typedef unsigned _BitInt(9) mode_t;
mode_t umask(mode_t);
void foo(void) { umask(0777); }
#elif defined(MODE_T_BITINT10)
typedef unsigned _BitInt(10) mode_t;
mode_t umask(mode_t);
void foo(void) {
  umask(0777);
  umask(01000); // expected-warning {{'umask' argument sets non-file-permission bits (01000); those bits are ignored}}
}
#endif
