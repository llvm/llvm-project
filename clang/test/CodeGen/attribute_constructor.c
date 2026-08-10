// RUN: %clang_cc1 %s -emit-llvm -o - | grep llvm.global_ctors

extern int bar();
void foo(void) __attribute__((constructor));
void foo(void) {
  bar();
}
