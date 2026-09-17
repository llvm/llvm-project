// RUN: %clang_cc1 -triple arm64-apple-macosx11.0 %s -verify

// Regression test for the macOS/BSD case. There mode_t is a 16-bit type, so an
// earlier attempt that gave umask a fixed unsigned-int builtin prototype turned
// the libc's own <sys/stat.h> declaration into an incompatible library
// redeclaration and silently dropped its builtin identity -- the fortify check
// went dead, even for the system header. umask now has no prototype to match
// (it is recognized purely by name), so the check fires regardless of how wide
// the libc makes mode_t.

#include "Inputs/warn-fortify-source-umask-darwin.h"

void call_umask_darwin(mode_t runtime_mode) {
  umask(0);
  umask(0644);
  umask(01000);   // expected-warning {{'umask' argument sets non-file-permission bits (01000); those bits are ignored}}
  umask(0xFFFF);  // expected-warning {{'umask' argument sets non-file-permission bits (0177000); those bits are ignored}}
  umask(runtime_mode); // no warning, not a constant
}
