// This test checks that the data object located in text section
// is properly emitted in the new section.

// RUN: %clang %cflags %s -o %t.exe -Wl,-q
// RUN: llvm-bolt %t.exe -o %t.bolt --lite=0 --use-old-text=0
// RUN: llvm-objdump -j .text -d --disassemble-symbols=arr %t.bolt | \
// RUN:   FileCheck %s

// CHECK: {{.*}} <arr>:

extern void exit(int);

typedef void (*FooPtr)();

void exitOk() { exit(0); }

__attribute__((section(".text"))) const FooPtr arr[] = {exitOk, 0};

int main() {
  arr[0]();
  return -1;
}
