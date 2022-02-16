// RUN: %clang_cc1 %s -debug-info-kind=line-tables-only -S -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -debug-info-kind=line-directives-only -S -emit-llvm -o - | FileCheck %s
// Checks that clang with "-gline-tables-only" or "-gline-directives-only" emits metadata for
// compile unit, subprogram and file.

int main(void) {
  // CHECK: ret i32 0, !dbg
  return 0;
}

// CHECK: !llvm.dbg.cu = !{!0}
// CHECK: !DICompileUnit(
// CHECK: !DISubprogram(
// CHECK: !DIFile(
