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
