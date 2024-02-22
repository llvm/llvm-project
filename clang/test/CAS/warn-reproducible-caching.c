// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t

// No warning in normal invocation.
// RUN: %clang -target x86_64-apple-macos11 -fsyntax-only %t/t.c -isystem %t/sys 2>&1 | FileCheck %s --check-prefix=NOWARN --allow-empty
// NOWARN-NOT: warning

// Warning in cached invocation even if coming from system header. -Werror does not affect it.
// RUN: %clang -target x86_64-apple-macos11 -c %t/t.c -o %t/t.o -isystem %t/sys -Werror \
// RUN:   -fdepscan=inline -Xclang -fcas-path -Xclang %t/cas -Xclang -fcache-compile-job 2>&1 | FileCheck %s --check-prefix=WARN
// WARN: warning: encountered non-reproducible token

// Error if explicitly turned into error.
// RUN: not %clang -target x86_64-apple-macos11 -c %t/t.c -o %t/t.o -isystem %t/sys -Werror=reproducible-caching \
// RUN:   -fdepscan=inline -Xclang -fcas-path -Xclang %t/cas -Xclang -fcache-compile-job 2>&1 | FileCheck %s --check-prefix=ERROR
// ERROR: error: encountered non-reproducible token

//--- t.c
#include <sys.h>

//--- sys/sys.h
const char *foo() {
  return __DATE__;
}
