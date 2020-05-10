/// We support coverage versions 4.2 and 4.7.
/// 4.7 (default, compatible with gcov 7) enables cfg_checksum.
// RUN: %clang_cc1 -emit-llvm -disable-red-zone -femit-coverage-data -coverage-version='402*' %s -o - | \
// RUN:   FileCheck --check-prefixes=CHECK,402 %s
// RUN: %clang_cc1 -emit-llvm -disable-red-zone -femit-coverage-data -coverage-version='407*' %s -o - | \
// RUN:   FileCheck --check-prefixes=CHECK,407 %s
// RUN: %clang_cc1 -emit-llvm -disable-red-zone -femit-coverage-data %s -o - | \
// RUN:   FileCheck --check-prefixes=CHECK,407 %s

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

// 402: private unnamed_addr constant [5 x i8] c"*204\00"
// 407: private unnamed_addr constant [5 x i8] c"*704\00"

// CHECK: @__llvm_internal_gcov_emit_function_args.0 = internal unnamed_addr constant [2 x %0]
// CHECK-SAME: [%0 zeroinitializer, %0 { i32 1, i32 0, i32 0 }]

// Check that the noredzone flag is set on the generated functions.

// CHECK: void @__llvm_gcov_writeout() unnamed_addr [[NRZ:#[0-9]+]]
// CHECK: void @__llvm_gcov_flush() unnamed_addr [[NRZ]]
// CHECK: void @__llvm_gcov_init() unnamed_addr [[NRZ]]

// CHECK: attributes [[NRZ]] = { {{.*}}noredzone{{.*}} }

// GCOV_FILE_INFO: !llvm.gcov = !{![[GCOV:[0-9]+]]}
// GCOV_FILE_INFO: ![[GCOV]] = !{!"aaa.gcno", !"bbb.gcda", !{{[0-9]+}}}
