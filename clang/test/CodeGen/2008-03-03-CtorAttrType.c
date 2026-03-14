// RUN: %clang_cc1 %s -emit-llvm -o - | grep llvm.global_ctors
int __attribute__((constructor)) foo(void) {
  return 0;
}
void __attribute__((constructor)) bar(void) {}

