// Checking crashes against injected binary functions created by patch
// entries pass and debug info turned on. In these cases, we were
// trying to fetch input to output maps on injected functions and
// crashing.

// REQUIRES: system-linux

// RUN: %clang %cflags -no-pie -g %s -fuse-ld=lld -o %t.exe -Wl,-q
// RUN: llvm-bolt -relocs %t.exe -o %t.out --update-debug-sections \
// RUN:   --force-patch

#include <stdio.h>

static void foo() { printf("foo\n"); }

int main() {
  foo();
  return 0;
}
