// RUN: %clang_cc1 %s -emit-llvm -o - | grep noalias

void * __attribute__ ((malloc)) foo (void) { return 0; }
