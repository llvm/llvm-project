// RUN: echo "; __SEPARATOR__" > %t.sep

// =============================================================================
// Array access without a macro
// =============================================================================

// RUN: %clang_cc1 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN:   -fbounds-safety -fbounds-safety-debug-trap-reasons=basic \
// RUN:   -emit-llvm %s -o %t.ll
//
// RUN: cat %t.ll %t.sep %t.ll > %t.repeated.ll
// RUN: FileCheck %s --check-prefixes=CHECK,BASIC,PLAIN \
// RUN:   --input-file=%t.repeated.ll

// RUN: %clang_cc1 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN:   -fbounds-safety -fbounds-safety-debug-trap-reasons=detailed \
// RUN:   -emit-llvm %s -o %t2.ll
//
// RUN: cat %t2.ll %t.sep %t2.ll > %t2.repeated.ll
// RUN: FileCheck %s --check-prefixes=CHECK,DETAILED,PLAIN \
// RUN:   --input-file=%t2.repeated.ll

// =============================================================================
// Array access through a macro
// =============================================================================

// RUN: %clang_cc1 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN:   -fbounds-safety -fbounds-safety-debug-trap-reasons=basic \
// RUN:   -emit-llvm -DARRAY_ACCESS_THROUGH_MACRO %s -o %t3.ll
//
// RUN: cat %t3.ll %t.sep %t3.ll > %t3.repeated.ll
// RUN: FileCheck %s --check-prefixes=CHECK,BASIC,MACRO \
// RUN:   --input-file=%t3.repeated.ll

// RUN: %clang_cc1 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN:   -fbounds-safety -fbounds-safety-debug-trap-reasons=detailed \
// RUN:   -emit-llvm -DARRAY_ACCESS_THROUGH_MACRO %s -o %t4.ll
//
// RUN: cat %t4.ll %t.sep %t4.ll > %t4.repeated.ll
// RUN: FileCheck %s --check-prefixes=CHECK,DETAILED,MACRO \
// RUN:   --input-file=%t4.repeated.ll


#define ARRAY(__ARR, __IDX) __ARR[__IDX]

int operation(int index) {
  int array[] = {0, 1, 2};
#ifndef ARRAY_ACCESS_THROUGH_MACRO
  return array[index];
#else
  return ARRAY(array, index);
#endif
}

// In first copy of the file

// CHECK-DAG: [[OPT_REMARK:![0-9]+]] = !{!"bounds-safety-check-ptr-le-upper-bound"}

// BASIC-DAG: [[TRAP_SCOPE:![0-9]+]] = distinct !DISubprogram(name: "__clang_trap_msg$Bounds check failed$Dereferencing above bounds", scope: [[FILE_SCOPE:![0-9]+]], file: [[FILE_SCOPE:![0-9]+]], type: {{![0-9]+}}, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: {{![0-9]+}}
// DETAILED-DAG: [[TRAP_SCOPE:![0-9]+]] = distinct !DISubprogram(name: "__clang_trap_msg$Bounds check failed$indexing above upper bound in 'array[index]'", scope: [[FILE_SCOPE:![0-9]+]], file: [[FILE_SCOPE:![0-9]+]], type: {{![0-9]+}}, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: {{![0-9]+}}

// CHECK-DAG: [[TRAP_LOC:![0-9]+]] = !DILocation(line: 0, scope: [[TRAP_SCOPE]], inlinedAt: [[SRC_LOC:![0-9]+]])
// PLAIN-DAG: [[SRC_LOC]] = !DILocation(line: 49, column: 10, scope: {{![0-9]+}})
// MACRO-DAG: [[SRC_LOC]] = !DILocation(line: 51, column: 10, scope: {{![0-9]+}})

// In the detailed mode the address space overflow gets its own trap reason.
// FIXME: Basic mode should probably be making this distinction too.

// DETAILED-DAG: [[TRAP_SCOPE_2:![0-9]+]] = distinct !DISubprogram(name: "__clang_trap_msg$Bounds check failed$indexing overflows address space in 'array[index]'", scope: [[FILE_SCOPE:![0-9]+]], file: [[FILE_SCOPE:![0-9]+]], type: {{![0-9]+}}, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: {{![0-9]+}}
// DETAILED-DAG: [[TRAP_LOC_2:![0-9]+]] = !DILocation(line: 0, scope: [[TRAP_SCOPE_2]], inlinedAt: [[SRC_LOC:![0-9]+]])




// CHECK-LABEL: ; __SEPARATOR__

// In second copy of the file

// CHECK-LABEL: @operation
// Note: We use the `OPT_REMARK` to make sure we match against the correct branch.
// FIXME: The overflow check should have its own opt-remark (rdar://150322607)
// CHECK: br i1 %{{.+}}, label %{{.+}}, label %[[TRAP_LABEL_0:[a-z0-9]+]], !dbg [[SRC_LOC]], !prof !{{[0-9]+}}, !annotation [[OPT_REMARK]]
// CHECK: [[TRAP_LABEL_0]]:
// CHECK-NEXT: call void @llvm.ubsantrap(i8 25) {{.*}}!dbg [[TRAP_LOC]]

// CHECK: br i1 %{{.+}}, label %{{.+}}, label %[[TRAP_LABEL_1:[a-z0-9]+]], !dbg [[SRC_LOC]], !prof !{{[0-9]+}}, !annotation [[OPT_REMARK]]
// CHECK: [[TRAP_LABEL_1]]:
// BASIC-NEXT: call void @llvm.ubsantrap(i8 25) {{.*}}!dbg [[TRAP_LOC]]
// DETAILED-NEXT: call void @llvm.ubsantrap(i8 25) {{.*}}!dbg [[TRAP_LOC_2]]
