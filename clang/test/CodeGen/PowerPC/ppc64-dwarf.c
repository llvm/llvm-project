// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,PPC64
static unsigned char dwarf_reg_size_table[1024];

int test(void) {
  __builtin_init_dwarf_reg_size_table(dwarf_reg_size_table);

  return __builtin_dwarf_sp_column();
}

// CHECK-LABEL: define{{.*}} signext i32 @test()
// CHECK:      store i8 8, ptr @dwarf_reg_size_table, align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 1), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 2), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 3), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 4), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 5), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 6), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 7), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 8), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 9), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 10), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 11), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 12), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 13), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 14), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 15), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 16), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 17), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 18), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 19), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 20), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 21), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 22), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 23), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 24), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 25), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 26), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 27), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 28), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 29), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 30), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 31), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 32), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 33), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 34), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 35), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 36), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 37), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 38), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 39), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 40), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 41), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 42), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 43), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 44), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 45), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 46), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 47), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 48), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 49), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 50), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 51), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 52), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 53), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 54), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 55), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 56), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 57), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 58), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 59), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 60), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 61), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 62), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 63), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 64), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 65), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 66), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 67), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 68), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 69), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 70), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 71), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 72), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 73), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 74), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 75), align 1
// CHECK-NEXT: store i8 4, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 76), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 77), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 78), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 79), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 80), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 81), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 82), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 83), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 84), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 85), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 86), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 87), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 88), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 89), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 90), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 91), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 92), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 93), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 94), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 95), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 96), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 97), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 98), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 99), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 100), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 101), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 102), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 103), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 104), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 105), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 106), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 107), align 1
// CHECK-NEXT: store i8 16, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 108), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 109), align 1
// CHECK-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 110), align 1
// PPC64-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 111), align 1
// PPC64-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 112), align 1
// PPC64-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 113), align 1
// PPC64-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 114), align 1
// PPC64-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 115), align 1
// PPC64-NEXT: store i8 8, ptr getelementptr inbounds (i8, ptr @dwarf_reg_size_table, i32 116), align 1
// CHECK-NEXT: ret i32 1
