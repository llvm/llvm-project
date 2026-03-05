// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM_NO_DEBUG
// RUN: %clang_cc1 -debug-info-kind=constructor -dwarf-version=4 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM_WITH_DEBUG
int foo(int a, int b) {
  // LLVM_NO_DEBUG-NOT: !dbg

  // LLVM_WITH_DEBUG-LABEL: foo
  // LLVM_WITH_DEBUG: %[[VAR_A:.*]] = load i32, ptr %{{.*}}, align 4, !dbg ![[DI_LOC1:.*]]
  // LLVM_WITH_DEBUG: %[[VAR_B:.*]] = load i32, ptr %{{.*}}, align 4, !dbg ![[DI_LOC2:.*]]
  // LLVM_WITH_DEBUG: %[[VAR_C:.*]] = add nsw i32 %[[VAR_A]], %[[VAR_B]], !dbg ![[DI_LOC1]]
  // LLVM_WITH_DEBUG: store i32 %[[VAR_C]], ptr %{{.*}}, align 4, !dbg ![[DI_LOC3:.*]]

  // LLVM_WITH_DEBUG: ![[DI_LOC3]] = !DILocation(line: [[LINE:.*]], scope: ![[SCOPE:.*]])
  // LLVM_WITH_DEBUG: ![[DI_LOC1]] = !DILocation(line: [[LINE]], column: {{.*}}, scope: ![[SCOPE]])
  // LLVM_WITH_DEBUG: ![[DI_LOC2]] = !DILocation(line: [[LINE]], column: {{.*}}, scope: ![[SCOPE]])
  int c = a + b;
  return c;
}