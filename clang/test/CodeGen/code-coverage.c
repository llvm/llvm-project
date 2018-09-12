// RUN: %clang_cc1 -emit-llvm -disable-red-zone -femit-coverage-data %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -disable-red-zone -femit-coverage-data -coverage-no-function-names-in-data %s -o - | FileCheck %s --check-prefix WITHOUTNAMES
// RUN: %clang_cc1 -emit-llvm -disable-red-zone -femit-coverage-data -coverage-notes-file=aaa.gcno -coverage-data-file=bbb.gcda -dwarf-column-info -debug-info-kind=limited -dwarf-version=4 %s -o - | FileCheck %s --check-prefix GCOV_FILE_INFO

// RUN: %clang_cc1 -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -femit-coverage-data %s 2>&1 | FileCheck --check-prefix=NEWPM %s
// RUN: %clang_cc1 -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -femit-coverage-data -O3 %s 2>&1 | FileCheck --check-prefix=NEWPM-O3 %s

// NEWPM-NOT: Running pass
// NEWPM: Running pass: GCOVProfilerPass

// NEWPM-O3-NOT: Running pass
// NEWPM-O3: Running pass: ForceFunctionAttrsPass
// NEWPM-O3: Running pass: GCOVProfilerPass


int test1(int a) {
  switch (a % 2) {
  case 0:
    ++a;
  case 1:
    a /= 2;
  }
  return a;
}

int test2(int b) {
  return b * 2;
}

// Inside function emission data structure, check that
// -coverage-no-function-names-in-data uses null as the function name.
// CHECK: @__llvm_internal_gcov_emit_function_args.0 = internal unnamed_addr constant
// CHECK-SAME: { i32 {{[0-9]+}}, i8* getelementptr inbounds ({{[^,]*}}, {{[^,]*}}* @
// CHECK-SAME: { i32 {{[0-9]+}}, i8* getelementptr inbounds ({{[^,]*}}, {{[^,]*}}* @
// WITHOUTNAMES: @__llvm_internal_gcov_emit_function_args.0 = internal unnamed_addr constant
// WITHOUTNAMES-NOT: getelementptr inbounds ({{.*}}@
// WITHOUTNAMES-SAME: zeroinitializer,
// WITHOUTNAMES-NOT: getelementptr inbounds ({{.*}}@
// WITHOUTNAMES-SAME: { i32 {{[0-9]+}}, i8* null,

// Check that the noredzone flag is set on the generated functions.

// CHECK: void @__llvm_gcov_writeout() unnamed_addr [[NRZ:#[0-9]+]]
// CHECK: void @__llvm_gcov_flush() unnamed_addr [[NRZ]]
// CHECK: void @__llvm_gcov_init() unnamed_addr [[NRZ]]

// CHECK: attributes [[NRZ]] = { {{.*}}noredzone{{.*}} }

// GCOV_FILE_INFO: !llvm.gcov = !{![[GCOV:[0-9]+]]}
// GCOV_FILE_INFO: ![[GCOV]] = !{!"aaa.gcno", !"bbb.gcda", !{{[0-9]+}}}
