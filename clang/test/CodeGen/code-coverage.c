/// We support coverage versions 3.4, 4.7 and 4.8.
/// 3.4 redesigns the format and changed .da to .gcda
/// 4.7 enables cfg_checksum.
/// 4.8 (default, compatible with gcov 7) emits the exit block the second.
// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -disable-red-zone -coverage-data-file=/dev/null -coverage-version='B21*' %s -o - | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK-CTOR-INIT,1210 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -disable-red-zone -coverage-data-file=/dev/null %s -o - | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK-CTOR-INIT,1110 %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix -emit-llvm -disable-red-zone -coverage-data-file=/dev/null -coverage-version='B21*' %s -o - | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK-RT-INIT,1210 %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix -emit-llvm -disable-red-zone -coverage-data-file=/dev/null %s -o - | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK-RT-INIT,1110 %s

// RUN: %clang_cc1 -emit-llvm -disable-red-zone -coverage-notes-file=aaa.gcno -coverage-data-file=bbb.gcda -debug-info-kind=limited -dwarf-version=4 %s -o - | FileCheck %s --check-prefix GCOV_FILE_INFO

// RUN: %clang_cc1 -emit-llvm-bc -o /dev/null -fdebug-pass-manager -coverage-data-file=/dev/null %s 2>&1 | FileCheck --check-prefix=NEWPM %s
// RUN: %clang_cc1 -emit-llvm-bc -o /dev/null -fdebug-pass-manager -coverage-data-file=/dev/null -O3 %s 2>&1 | FileCheck --check-prefix=NEWPM-O3 %s

// NEWPM: Running pass: VerifierPass
// NEWPM: Running pass: GCOVProfilerPass

// NEWPM-O3: Running pass: VerifierPass
// NEWPM-O3: Running pass: Annotation2MetadataPass
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


// CHECK: @__llvm_internal_gcov_emit_function_args.0 = internal unnamed_addr constant [2 x %emit_function_args_ty]
// CHECK-SAME: [%emit_function_args_ty { i32 0, i32 {{[-0-9]+}}, i32 {{[-0-9]+}} }, %emit_function_args_ty { i32 1, i32 {{[-0-9]+}}, i32 {{[-0-9]+}} }]

// CHECK: @__llvm_internal_gcov_emit_file_info = internal unnamed_addr constant [1 x %file_info]
/// 0x4231312a 'B' '1' '1' '*'
// 1110-SAME: i32 1110520106
/// 0x4232312a 'B' '2' '1' '*'
// 1210-SAME: i32 1110585642

// Check for gcov initialization function pointers.
// CHECK-RT-INIT: @__llvm_covinit_functions = private constant { ptr, ptr } { ptr @__llvm_gcov_writeout, ptr @__llvm_gcov_reset }, section "__llvm_covinit"

// Check that the noredzone flag is set on the generated functions.

// CHECK: void @__llvm_gcov_writeout() unnamed_addr [[NRZ:#[0-9]+]]
// CHECK-CTOR-INIT: void @__llvm_gcov_init() unnamed_addr [[NRZ]]

// CHECK: attributes [[NRZ]] = { {{.*}}noredzone{{.*}} }

// GCOV_FILE_INFO: !llvm.gcov = !{![[GCOV:[0-9]+]]}
// GCOV_FILE_INFO: ![[GCOV]] = !{!"aaa.gcno", !"bbb.gcda", !{{[0-9]+}}}
