
// RUN: %clang_cc1 -DTYPE=__bidi_indexable -O0 -debug-info-kind=standalone -dwarf-version=5 -fbounds-safety -emit-llvm %s -o %t_bidi.ll
// RUN: echo "; __SEPARATOR__" > %t.sep
// RUN: cat %t_bidi.ll %t.sep %t_bidi.ll > %t_bidi.repeated.ll
// RUN: FileCheck %s --input-file=%t_bidi.repeated.ll

// RUN: %clang_cc1 -DTYPE=__indexable -O0 -debug-info-kind=standalone -dwarf-version=5 -fbounds-safety -emit-llvm %s -o %t_idx.ll
// RUN: echo "; __SEPARATOR__" > %t.sep
// RUN: cat %t_idx.ll %t.sep %t_idx.ll > %t_idx.repeated.ll
// RUN: FileCheck %s --input-file=%t_idx.repeated.ll

#include <ptrcheck.h>

#ifndef TYPE
#error TYPE must be defined
#endif

int* __single operation(int* TYPE i) {
    return i;
}
// In first copy of the file

// CHECK-DAG: [[OPT_REMARK:![0-9]+]] = !{!"bounds-safety-check-ptr-le-upper-bound"}
// CHECK-DAG: [[TRAP_SCOPE:![0-9]+]] = distinct !DISubprogram(name: "__clang_trap_msg$Bounds check failed$Pointer above bounds while casting", scope: [[FILE_SCOPE:![0-9]+]], file: [[FILE_SCOPE:![0-9]+]], type: {{![0-9]+}}, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: {{![0-9]+}}
// CHECK-DAG: [[TRAP_LOC:![0-9]+]] = !DILocation(line: 0, scope: [[TRAP_SCOPE]], inlinedAt: [[SRC_LOC:![0-9]+]])
// CHECK-DAG: [[SRC_LOC]] = !DILocation(line: 19, column: 12, scope: {{![0-9]+}})
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
// CHECK-NEXT: call void @llvm.ubsantrap(i8 25) {{.*}}!dbg [[TRAP_LOC]]
