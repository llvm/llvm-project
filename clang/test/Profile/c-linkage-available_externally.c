// Make sure instrumentation data from available_externally functions doesn't
// get thrown out and are emitted with the expected linkage.
// RUN: %clang_cc1 -O2 -triple x86_64-apple-macosx10.9 -main-file-name c-linkage-available_externally.c %s -o - -emit-llvm -fprofile-instrument=clang | FileCheck %s

// CHECK: @__profc_foo = linkonce_odr hidden global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
// CHECK: @__profd_foo = linkonce_odr hidden global {{.*}} i64 sub (i64 ptrtoint (ptr @__profc_foo to i64), i64 ptrtoint (ptr @__profd_foo to i64)), {{.*}}, section "__DATA,__llvm_prf_data,regular,live_support", align 8
inline int foo(void) { return 1; }

int main(void) {
  return foo();
}
