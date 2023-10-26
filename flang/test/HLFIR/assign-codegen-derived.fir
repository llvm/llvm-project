// Test hlfir.assign code generation to FIR of derived type requiring
// or not finalization.

// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

!t_simple = !fir.type<simple{i:i32}>
fir.type_info @simple noinit nodestroy nofinal : !t_simple

func.func @test_simple(%a: !fir.ref<!t_simple>, %b: !fir.ref<!t_simple>) {
  hlfir.assign %b to %a : !fir.ref<!t_simple>, !fir.ref<!t_simple>
  return
}
// CHECK-LABEL: func.func @test_simple(
// CHECK-NOT: Destroy
// CHECK:  %[[VAL_1:.*]] = fir.coordinate_of %{{.*}}, %{{.*}} : (!fir.ref<!fir.type<simple{i:i32}>>, !fir.field) -> !fir.ref<i32>
// CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %{{.*}}, %{{.*}} : (!fir.ref<!fir.type<simple{i:i32}>>, !fir.field) -> !fir.ref<i32>
// CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
// CHECK:  fir.store %[[VAL_4]] to %[[VAL_3]] : !fir.ref<i32>


!t_with_final = !fir.type<with_final{i:i32}>
fir.type_info @with_final noinit : !t_with_final

func.func @test_with_final(%a: !fir.ref<!t_with_final>, %b: !fir.ref<!t_with_final>) {
  hlfir.assign %b to %a : !fir.ref<!t_with_final>, !fir.ref<!t_with_final>
  return
}
// CHECK-LABEL: func.func @test_with_final(
// CHECK:  fir.call @_FortranAAssign
