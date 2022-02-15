// RUN: %clang_cc1 %s -emit-llvm -o - | grep llvm.global_ctors

void foo(void) __attribute__((constructor));
void foo(void) {
  bar();
}
