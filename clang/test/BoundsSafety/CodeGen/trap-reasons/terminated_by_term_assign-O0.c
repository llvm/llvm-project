
// RUN: %clang_cc1 -O0 -debug-info-kind=standalone -dwarf-version=5 -fbounds-safety -emit-llvm %s -o - | FileCheck %s
#include <ptrcheck.h>

void assign(char* __null_terminated str) {
    str[0] = 'x';
}

// CHECK-LABEL: @assign

// CHECK: %[[CMP_REG:[a-z0-9]+]] = icmp ne i8 %{{.+}}, 0, !dbg [[LOC:![0-9]+]]
// CHECK-NEXT: br i1 %[[CMP_REG]], label %{{.+}}, label %[[TRAP_LABEL:[a-z0-9]+]], !dbg [[LOC]]

// CHECK: [[TRAP_LABEL]]:
// CHECK-NEXT: call void @llvm.ubsantrap(i8 25) {{.*}}!dbg [[TRAP_LOC:![0-9]+]]

// CHECK-DAG: [[TRAP_LOC]] = !DILocation(line: 0, scope: [[TRAP_SCOPE:![0-9]+]], inlinedAt: [[LOC]])
// CHECK-DAG: [[TRAP_SCOPE]] = distinct !DISubprogram(name: "__clang_trap_msg$Bounds check failed$The terminator cannot be assigned", scope: [[FILE_SCOPE:![0-9]+]], file: [[FILE_SCOPE:![0-9]+]], type: {{![0-9]+}}, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: {{![0-9]+}}
