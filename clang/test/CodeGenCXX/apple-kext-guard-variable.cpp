// RUN: %clang -target x86_64-apple-darwin10 -S -o %t.s -Wno-stdlibcxx-not-found -mkernel -Xclang -verify %s

int foo();
void test() {
  static int y = 0;
  static int x = foo(); // expected-error {{this initialization requires a guard variable, which the kernel does not support}}
}
