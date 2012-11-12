// RUN: %clang_cc1 %s -emit-llvm -o - | grep llvm.global_ctors

void foo() __attribute__((constructor));
void foo() {
  bar();
}
