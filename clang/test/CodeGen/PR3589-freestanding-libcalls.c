// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | grep 'declare i32 @printf' | count 1
// RUN: %clang_cc1 -triple i386-unknown-unknown -O2 -emit-llvm %s -o - | grep 'declare noundef i32 @puts' | count 1
// RUN: %clang_cc1 -triple i386-unknown-unknown -ffreestanding -O2 -emit-llvm %s -o - | not grep 'declare noundef i32 @puts'

int printf(const char *, ...);

void f0(void) {
  printf("hello\n");
}
