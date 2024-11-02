// RUN: %clang_cc1 -verify -S -emit-llvm -o- %s -isystem %S -DWITH_DECL | FileCheck --check-prefix=CHECK-WITH-DECL %s
// RUN: %clang_cc1 -verify -S -emit-llvm -o- %s -isystem %S -UWITH_DECL | FileCheck --check-prefix=CHECK-NO-DECL %s
// RUN: %clang_cc1 -verify -S -emit-llvm -o- %s -isystem %S -disable-llvm-passes -DWITH_SELF_REFERENCE_DECL | FileCheck --check-prefix=CHECK-SELF-REF-DECL %s
//
// CHECK-WITH-DECL-NOT: @llvm.memcpy
// CHECK-NO-DECL: @llvm.memcpy
// CHECK-SELF-REF-DECL-LABEL: define {{.*}}ptr @memcpy.inline
// CHECK-SELF-REF-DECL:       @llvm.memcpy.{{.*}}(
//
#include <memcpy-nobuiltin.inc>
void test(void *dest, void const *from, size_t n) {
  memcpy(dest, from, n);

  static char buffer[1];
  memcpy(buffer, from, 2); // expected-warning {{'memcpy' will always overflow; destination buffer has size 1, but size argument is 2}}
}
