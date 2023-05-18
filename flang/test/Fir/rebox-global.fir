// RUN: tco %s | FileCheck %s
// RUN: %flang_fc1 -emit-llvm  %s -o - | FileCheck %s

fir.global @x target : i32

fir.global @p : !fir.box<!fir.ptr<i32>> {
  %0 = fir.address_of(@x) : !fir.ref<i32>
  %1 = fir.embox %0 : (!fir.ref<i32>) -> !fir.box<i32>
  %2 = fir.rebox %1 : (!fir.box<i32>) -> !fir.box<!fir.ptr<i32>>
  fir.has_value %2 : !fir.box<!fir.ptr<i32>>
}
// CHECK: @p = global { ptr, i64, i32, i8, i8, i8, i8 } { ptr @x, {{.*}}, i32 {{.*}}, i8 0, i8 9, i8 1, i8 0 }

// Test that sdiv is not generated inside fir.global.
fir.global @char4 target : !fir.char<4,10>
fir.global @pointer_char4_init : !fir.box<!fir.ptr<!fir.char<4,10>>> {
  %0 = fir.address_of(@char4) : !fir.ref<!fir.char<4,10>>
  %1 = fir.embox %0 : (!fir.ref<!fir.char<4,10>>) -> !fir.box<!fir.char<4,10>>
  %2 = fircg.ext_rebox %1 : (!fir.box<!fir.char<4,10>>) -> !fir.box<!fir.ptr<!fir.char<4,10>>>
  fir.has_value %2 : !fir.box<!fir.ptr<!fir.char<4,10>>>
}
// CHECK-LABEL: @pointer_char4_init
// CHECK-SAME: { ptr @char4, i64 ptrtoint (ptr getelementptr ([10 x i32], ptr null, i32 1) to i64), i32 20180515, i8 0, i8 44, i8 1, i8 0 }
