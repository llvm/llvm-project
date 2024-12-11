
// RUN: %clang_cc1 -O0 -debug-info-kind=standalone -dwarf-version=5 -fbounds-safety -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

int *__indexable convert(int *__bidi_indexable p) {
  return p;
}

// CHECK-LABEL: @convert(

// CHECK: [[BRANCH_REG:%[0-9]+]] = icmp uge ptr %{{.+}}, %{{.+}}, !dbg [[LOC:![0-9]+]]
// CHECK: br i1 [[BRANCH_REG]], label {{.+}}, label %[[TRAP_LABEL:[a-z0-9]+]], !dbg [[LOC]]
// CHECK: [[TRAP_LABEL]]:
// CHECK-NEXT: call void @llvm.ubsantrap(i8 25) {{.*}}!dbg [[TRAP_LOC:![0-9]+]]

// CHECK-DAG: [[TRAP_LOC]] = !DILocation(line: 0, scope: [[TRAP_SCOPE:![0-9]+]], inlinedAt: [[LOC]])
// CHECK-DAG: [[TRAP_SCOPE]] = distinct !DISubprogram(name: "__clang_trap_msg$Bounds check failed$Converted __indexable pointer is below bounds", scope: [[FILE_SCOPE:![0-9]+]], file: [[FILE_SCOPE]], type: {{.+}}, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: {{![0-9]+}}
// CHECK-DAG: [[LOC]] = !DILocation(line: 7, column: 10, scope: {{![0-9]+}})
