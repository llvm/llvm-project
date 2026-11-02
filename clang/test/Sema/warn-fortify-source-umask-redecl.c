// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -verify

// A user declaration of umask whose integer type differs from the historical
// unsigned-int spelling -- here a plain int, as in a K&R-era or sloppy
// prototype -- is still recognized as the libc umask and checked. Because
// umask is a LibBuiltin with no prototype to match, the declaration is not
// flagged as an incompatible library redeclaration (it would have been when
// the builtin carried a fixed unsigned-int prototype), and the fortify check
// still fires on an out-of-range constant.

extern int umask(int);

void call_user_umask(void) {
  umask(0644);
  umask(0xFFFF); // expected-warning {{'umask' argument sets non-file-permission bits (0177000); those bits are ignored}}
}
