// This test checks that BOLT refuses to optimize binaries
// compiled with -mbranch-protection=pac-ret, unless the
// --allow-experimental-pacret flag is set.

// REQUIRES: system-linux,bolt-runtime
// RUN: %clangxx --target=aarch64-unknown-linux-gnu \
// RUN: -mbranch-protection=pac-ret -fuse-ld=lld -Wl,-q %s -o %t.exe
// RUN: not --crash llvm-bolt %t.exe -o %t.bolt.exe 2>&1 | FileCheck %s

// CHECK: BOLT-ERROR: set --allow-experimental-pacret to allow processing

extern "C" int printf(const char *, ...);

void bar() { throw 10; }

void foo() {
  try {
    bar();
  } catch (int e) {
    printf("Exception caught: %d\n", e);
  }
}

int main() {
  foo();
  return 0;
}
