// RUN: clang -S -emit-llvm -o- %s -isystem . -DWITH_DECL | FileCheck --check-prefix=CHECK-WITH-DECL %s
// RUN: clang -S -emit-llvm -o- %s -isystem . -UWITH_DECL | FileCheck --check-prefix=CHECK-NO-DECL %s
// CHECK-WITH-DECL-NOT: @llvm.memcpy
// CHECK-NO-DECL: @llvm.memcpy
#include <memcpy-nobuitin.inc>
void test(void* dest, void const* from, size_t n) {
  memcpy(dest,  from, n);
}
