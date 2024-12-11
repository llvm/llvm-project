
// RUN: %clang_cc1 -O3 -debug-info-kind=standalone -dwarf-version=5 -fbounds-safety -emit-llvm %s -o - | FileCheck --check-prefixes CHECK,CHECK-FAKE-FRAME  %s

int bad_read(int index) {
  int array[] = {0, 1, 2};
  int array2[] = {1, 5, 8};
  // Placing these two accesses on different lines prevents their debug
  // location from being being merged to have the same line. So they will
  // end up as 0-line locations.
  return array[index] +
    array2[index];
}

// Merging of traps isn't necessarily ideal because it causes debug info to be lost
// and therefore the trap information stored there to be lost too.
// We should investigate preventing merging of traps (rdar://85946510).
// For now just check that when traps get merged we drop the -fbounds-safety trap reason
// to avoid confusing users.

// Check that only one trap is emitted when optimized.
// CHECK--LABEL: @bad_read
// CHECK: call void @llvm.ubsantrap(i8 25) {{.*}}!dbg [[TRAP_LOC:![0-9]+]]
// CHECK-NOT: call void @llvm.ubsantrap(i8 25)

// Check the merged trap as a 0-line location and scoped within the correct function
// !26 = !DILocation(line: 0, scope: !14)
// !14 = distinct !DISubprogram(name: "bad_read", scope: !15, file:
// CHECK-DAG: [[TRAP_LOC]] = !DILocation(line: 0, scope: [[TRAP_SCOPE:![0-9]+]])
// CHECK-DAG: [[TRAP_SCOPE]] = distinct !DISubprogram(name: "bad_read"

// Make sure fake -fbounds-safety traps are not emitted.
// CHECK-FAKE-FRAME-NOT: {{![0-9]+}} = distinct !DISubprogram(name: "Bounds check failed:
