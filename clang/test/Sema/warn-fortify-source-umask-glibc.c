// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -verify

// Verify the builtin-based fortify check fires on a glibc-style declaration
// of umask whose prototype is written in terms of the internal __mode_t
// typedef (= unsigned int) rather than mode_t directly. Recognition is by
// builtin name, so the libc's particular mode_t spelling does not matter.

#include "Inputs/warn-fortify-source-umask-glibc.h"

void call_umask_glibc(mode_t runtime_mode) {
  umask(0);
  umask(0644);
  umask(01000);   // expected-warning {{'umask' argument sets non-file-permission bits (01000); those bits are ignored}}
  umask(0xFFFF);  // expected-warning {{'umask' argument sets non-file-permission bits (0177000); those bits are ignored}}
  umask(runtime_mode); // no warning, not a constant
}
