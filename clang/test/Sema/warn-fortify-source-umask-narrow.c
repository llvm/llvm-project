// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DCHAR_REDECL -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DMODE_T_CHAR -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DMODE_T_CHAR -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DMODE_T_BOOL -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DMODE_T_BITINT15 -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DMODE_T_BITINT16 -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DMODE_T_SHORT -verify

// Regression tests for GH244171: bail out when umask is declared with an
// integer parameter narrower than 16 bits (and do not assert constructing the
// 0777 bitmask APInt).

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
#elif defined(MODE_T_BITINT15)
// expected-no-diagnostics
typedef unsigned _BitInt(15) mode_t;
mode_t umask(mode_t);
void foo(void) {
  umask(0777);
  umask(01000);
}
#elif defined(MODE_T_BITINT16)
typedef unsigned _BitInt(16) mode_t;
mode_t umask(mode_t);
void foo(void) {
  umask(0777);
  umask(01000); // expected-warning {{'umask' argument sets non-file-permission bits (01000); those bits are ignored}}
}
#elif defined(MODE_T_SHORT)
typedef unsigned short mode_t;
mode_t umask(mode_t);
void foo(void) {
  umask(0777);
  umask(01000); // expected-warning {{'umask' argument sets non-file-permission bits (01000); those bits are ignored}}
}
#endif
