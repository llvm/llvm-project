// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -verify
// expected-no-diagnostics

// A user declaration of umask whose signature does not match mode_t(mode_t) is
// neither flagged as an incompatible library redeclaration nor recognized as
// the libc umask builtin.

typedef unsigned int mode_t;
extern int umask(int);

void call_user_umask(void) {
  umask(0644);
  umask(0xFFFF);
}
