/// We support coverage versions 3.4, 4.7 and 4.8.
/// 3.4 redesigns the format and changed .da to .gcda
/// 4.7 enables cfg_checksum.
/// 4.8 (default, compatible with gcov 7) emits the exit block the second.
// RUN: %clang_cc1 -emit-llvm -disable-red-zone -fprofile-arcs -coverage-version='304*' %s -o - | \
// RUN:   FileCheck --check-prefixes=CHECK,304 %s
// RUN: %clang_cc1 -emit-llvm -disable-red-zone -fprofile-arcs -coverage-version='407*' %s -o - | \
// RUN:   FileCheck --check-prefixes=CHECK,407 %s
// RUN: %clang_cc1 -emit-llvm -disable-red-zone -fprofile-arcs %s -o - | \
// RUN:   FileCheck --check-prefixes=CHECK,408 %s

// RUN: %clang_cc1 -emit-llvm -disable-red-zone -fprofile-arcs -coverage-notes-file=aaa.gcno -coverage-data-file=bbb.gcda -debug-info-kind=limited -dwarf-version=4 %s -o - | FileCheck %s --check-prefix GCOV_FILE_INFO

// RUN: %clang_cc1 -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -fprofile-arcs %s 2>&1 | FileCheck --check-prefix=NEWPM %s
// RUN: %clang_cc1 -emit-llvm-bc -o /dev/null -fexperimental-new-pass-manager -fdebug-pass-manager -fprofile-arcs -O3 %s 2>&1 | FileCheck --check-prefix=NEWPM-O3 %s

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


// CHECK: @__llvm_internal_gcov_emit_function_args.0 = internal unnamed_addr constant [2 x %emit_function_args_ty]
// CHECK-SAME: [%emit_function_args_ty { i32 0, i32 {{[-0-9]+}}, i32 {{[-0-9]+}} }, %emit_function_args_ty { i32 1, i32 {{[-0-9]+}}, i32 {{[-0-9]+}} }]

// CHECK: @__llvm_internal_gcov_emit_file_info = internal unnamed_addr constant [1 x %file_info]
/// 0x3330342a '3' '0' '4' '*'
// 304-SAME: i32 858797098
/// 0x3430372a '4' '0' '7' '*'
// 407-SAME: i32 875575082
/// 0x3430382a '4' '0' '8' '*'
// 408-SAME: i32 875575338

// Check that the noredzone flag is set on the generated functions.

// CHECK: void @__llvm_gcov_writeout() unnamed_addr [[NRZ:#[0-9]+]]
// CHECK: void @__llvm_gcov_init() unnamed_addr [[NRZ]]

// CHECK: attributes [[NRZ]] = { {{.*}}noredzone{{.*}} }

// GCOV_FILE_INFO: !llvm.gcov = !{![[GCOV:[0-9]+]]}
// GCOV_FILE_INFO: ![[GCOV]] = !{!"aaa.gcno", !"bbb.gcda", !{{[0-9]+}}}
