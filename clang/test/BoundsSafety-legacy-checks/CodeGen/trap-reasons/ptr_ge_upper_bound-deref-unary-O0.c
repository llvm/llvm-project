
// RUN: %clang_cc1 -DTYPE=__bidi_indexable -O0 -debug-info-kind=standalone -dwarf-version=5 -fbounds-safety -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -DTYPE=__indexable -O0 -debug-info-kind=standalone -dwarf-version=5 -fbounds-safety -emit-llvm %s -o - | FileCheck %s
#include <ptrcheck.h>

#ifndef TYPE
#error TYPE must be defined
#endif

int operation(int* TYPE i) {
    return *i;
}

// CHECK-LABEL: @operation
// We don't try to match the registers used in the comparison because trying
// to match the IR is very fragile.
// CHECK: [[BRANCH_REG:%[0-9]+]] = icmp ult ptr %{{.+}}, %{{.+}}, !dbg [[LOC:![0-9]+]]
// CHECK-NEXT: br i1 [[BRANCH_REG]], label {{.+}}, label %[[TRAP_LABEL:[a-z0-9]+]], !dbg [[LOC]]
// CHECK: [[TRAP_LABEL]]:
// CHECK-NEXT: call void @llvm.ubsantrap(i8 25) {{.*}}!dbg [[TRAP_LOC:![0-9]+]]

// CHECK-DAG: [[TRAP_LOC]] = !DILocation(line: 0, scope: [[TRAP_SCOPE:![0-9]+]], inlinedAt: [[LOC]])
// CHECK-DAG: [[TRAP_SCOPE]] = distinct !DISubprogram(name: "__clang_trap_msg$Bounds check failed$Dereferencing above bounds", scope: [[FILE_SCOPE:![0-9]+]], file: [[FILE_SCOPE]], type: {{![0-9]+}}, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: {{![0-9]+}}
// CHECK-DAG: [[LOC]] = !DILocation(line: 11, column: 12, scope: {{![0-9]+}})
