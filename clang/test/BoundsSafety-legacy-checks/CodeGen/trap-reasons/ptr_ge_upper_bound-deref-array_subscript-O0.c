
// RUN: %clang_cc1 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN:   -fbounds-safety -fbounds-safety-debug-trap-reasons=basic \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefixes=CHECK,BASIC %s

// RUN: %clang_cc1 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN:   -fbounds-safety -fbounds-safety-debug-trap-reasons=detailed \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefixes=CHECK,DETAILED %s

int operation(int index) {
  int array[] = {0, 1, 2};
  return array[index];
}

// CHECK-LABEL: @operation
// We don't try to match the registers used in the comparison because trying
// to match the IR is very fragile.
// CHECK: [[BRANCH_REG:%[0-9]+]] = icmp ult ptr %{{.+}}, %{{.+}}, !dbg [[LOC:![0-9]+]]
// CHECK-NEXT: br i1 [[BRANCH_REG]], label {{.+}}, label %[[TRAP_LABEL:[a-z0-9]+]], !dbg [[LOC]]
// CHECK: [[TRAP_LABEL]]:
// CHECK-NEXT: call void @llvm.ubsantrap(i8 25) {{.*}}!dbg [[TRAP_LOC:![0-9]+]]

// CHECK-DAG: [[TRAP_LOC]] = !DILocation(line: 0, scope: [[TRAP_SCOPE:![0-9]+]], inlinedAt: [[LOC]])

// BASIC-DAG: [[TRAP_SCOPE]] = distinct !DISubprogram(name: "__clang_trap_msg$Bounds check failed$Dereferencing above bounds", scope: [[FILE_SCOPE:![0-9]+]], file: [[FILE_SCOPE]], type: {{![0-9]+}}, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: {{![0-9]+}}
// DETAILED-DAG: [[TRAP_SCOPE]] = distinct !DISubprogram(name: "__clang_trap_msg$Bounds check failed$indexing above upper bound in 'array[index]'", scope: [[FILE_SCOPE:![0-9]+]], file: [[FILE_SCOPE]], type: {{![0-9]+}}, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: {{![0-9]+}}

// CHECK-DAG: [[LOC]] = !DILocation(line: 12, column: 10, scope: {{![0-9]+}})
