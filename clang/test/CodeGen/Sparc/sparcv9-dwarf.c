// RUN: %clang_cc1 -triple sparcv9-unknown-unknown -emit-llvm %s -o - | FileCheck %s
static unsigned char dwarf_reg_size_table[102+1];

int test(void) {
  __builtin_init_dwarf_reg_size_table(dwarf_reg_size_table);

  return __builtin_dwarf_sp_column();
}

// CHECK-LABEL: define{{.*}} signext i32 @test()
// CHECK:       store i8 8, ptr @dwarf_reg_size_table
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 1), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 2), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 3), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 4), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 5), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 6), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 7), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 8), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 9), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 10), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 11), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 12), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 13), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 14), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 15), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 16), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 17), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 18), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 19), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 20), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 21), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 22), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 23), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 24), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 25), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 26), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 27), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 28), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 29), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 30), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 31), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 32), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 33), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 34), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 35), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 36), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 37), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 38), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 39), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 40), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 41), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 42), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 43), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 44), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 45), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 46), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 47), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 48), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 49), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 50), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 51), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 52), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 53), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 54), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 55), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 56), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 57), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 58), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 59), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 60), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 61), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 62), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 63), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 64), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 65), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 66), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 67), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 68), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 69), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 70), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 71), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 72), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 73), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 74), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 75), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 76), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 77), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 78), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 79), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 80), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 81), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 82), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 83), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 84), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 85), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 86), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds nuw (i8, ptr @dwarf_reg_size_table, i64 87), align 1
// CHECK-NEXT:  ret i32 14
