// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -verify

// umask forward-declared in user code without including <sys/stat.h>. Because
// umask is a LibBuiltin recognized by name (not by system-header origin), the
// fortify check still fires here -- this is the case the earlier system-header
// gate went silent on. umask has no prototype to match against, so the
// declaration is never reported as an incompatible library redeclaration
// regardless of how this target spells mode_t.

typedef unsigned mode_t;
mode_t umask(mode_t);

void call_user_umask(mode_t runtime_mode) {
  umask(0);
  umask(0644);
  umask(01000);   // expected-warning {{'umask' argument sets non-file-permission bits (01000); those bits are ignored}}
  umask(0xFFFF);  // expected-warning {{'umask' argument sets non-file-permission bits (0177000); those bits are ignored}}
  umask(runtime_mode); // no warning, not a constant
}
