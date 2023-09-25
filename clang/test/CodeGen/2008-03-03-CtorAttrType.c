// RUN: %clang_cc1 %s -emit-llvm -o - | grep llvm.global_ctors
void __attribute__((constructor)) foo(void) {}
void __attribute__((constructor)) bar(void) {}

