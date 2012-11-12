// RUN: %clang_cc1 %s -gline-tables-only -S -emit-llvm -o - | FileCheck %s
// Checks that clang with "-gline-tables-only" emits metadata for
// compile unit, subprogram and file.

int main() {
  // CHECK: ret i32 0, !dbg
  return 0;
}

// CHECK: !llvm.dbg.cu = !{!0}
// CHECK: DW_TAG_compile_unit
// CHECK: {{.*main.* DW_TAG_subprogram}}
// CHECK: DW_TAG_file_type
