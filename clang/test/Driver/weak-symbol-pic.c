// RUN: %clang -fpic -S -target arm-linux-gnueabihf -o - %s | FileCheck %s
// RUN: %clang -fpic -S -target armeb-linux-gnueabihf -o - %s | FileCheck %s

// Test that weak symbols use GOT relocations in PIC mode on ARM.
// This matches GCC behavior where weak symbols require GOT because they
// can be preempted at runtime.

// CHECK: xxx(GOT_PREL)

void __attribute__((__weak__)) xxx(void);

void call_func(void (*f)(void));

int main() {
  call_func(xxx);
  return 0;
}