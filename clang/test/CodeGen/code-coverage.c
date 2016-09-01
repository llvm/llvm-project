// RUN: %clang_cc1 -emit-llvm -disable-red-zone -femit-coverage-data %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -disable-red-zone -femit-coverage-data -coverage-no-function-names-in-data %s -o - | FileCheck %s --check-prefix WITHOUTNAMES
// RUN: %clang_cc1 -emit-llvm -disable-red-zone -femit-coverage-data -coverage-notes-file=aaa.gcno -coverage-data-file=bbb.gcda %s -o - | FileCheck %s --check-prefix GCOV_FILE_INFO

// <rdar://problem/12843084>

int test1(int a) {
  switch (a % 2) {
  case 0:
    ++a;
  case 1:
    a /= 2;
  }
  return a;
}

// Check that the noredzone flag is set on the generated functions.

// CHECK: void @__llvm_gcov_indirect_counter_increment(i32* %{{.*}}, i64** %{{.*}}) unnamed_addr [[NRZ:#[0-9]+]]

// Inside llvm_gcov_writeout, check that -coverage-no-function-names-in-data
// passes null as the function name.
// CHECK: void @__llvm_gcov_writeout() unnamed_addr [[NRZ]]
// CHECK: call void @llvm_gcda_emit_function({{.*}}, i8* getelementptr {{.*}}, {{.*}})
// WITHOUTNAMES: void @__llvm_gcov_writeout() unnamed_addr
// WITHOUTNAMES: call void @llvm_gcda_emit_function({{.*}}, i8* null, {{.*}})

// CHECK: void @__llvm_gcov_flush() unnamed_addr [[NRZ]]
// CHECK: void @__llvm_gcov_init() unnamed_addr [[NRZ]]

// CHECK: attributes [[NRZ]] = { {{.*}}noredzone{{.*}} }

// GCOV_FILE_INFO: !llvm.gcov = !{![[GCOV:[0-9]+]]}
// GCOV_FILE_INFO: ![[GCOV]] = !{!"aaa.gcno", !"bbb.gcda", !{{[0-9]+}}}
