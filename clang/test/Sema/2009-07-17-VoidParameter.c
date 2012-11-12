// RUN: not %clang_cc1 -emit-llvm %s -o -
// PR4214
typedef void vt;
void (*func_ptr)(vt my_vt); // expected-error {argument may not have 'void' type}
