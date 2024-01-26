// Test fir.boxproc type conversion in the boxed-procedure pass.
// RUN: fir-opt --boxed-procedure %s | FileCheck %s

//CHECK-LABEL:  func.func private @test1(!fir.type<a{x:i32,y:f64}>, () -> ()) -> none
func.func private @test1(!fir.type<a{x:i32, y:f64}>, !fir.boxproc<() -> ()>) -> none

//CHECK-LABEL:  func.func private @test2((!fir.type<a{x:i32,y:f64}>) -> ()) -> none
func.func private @test2(!fir.boxproc<(!fir.type<a{x:i32, y:f64}>) -> ()>) -> none

//CHECK-LABEL:  func.func private @test3(() -> !fir.type<a{x:i32,y:f64}>) -> none
func.func private @test3(!fir.boxproc<() -> (!fir.type<a{x:i32, y:f64}>)>) -> none

//CHECK-LABEL:  func.func private @test5(((i32) -> f32) -> ())
func.func private @test5(!fir.boxproc<(!fir.boxproc<(i32) -> (f32)>) -> ()>)

// CHECK-LABEL:   func.func @proc_pointer_component(
// CHECK-SAME:                                      %[[VAL_0:.*]]: (!fir.ref<i32>) -> f32,
// CHECK-SAME:                                      %[[VAL_1:.*]]: !fir.ref<i32>) {

func.func @proc_pointer_component(%arg0 : !fir.boxproc<(!fir.ref<i32>) -> f32>, %arg1: !fir.ref<i32>) {
  %0 = fir.alloca !fir.type<_QFtestTmatrix{element:!fir.array<2x2xf32>,solve:!fir.boxproc<() -> ()>}>
  %1 = fir.field_index solve, !fir.type<_QFtestTmatrix{element:!fir.array<2x2xf32>,solve:!fir.boxproc<() -> ()>}>
  %2 = fir.coordinate_of %0, %1 : (!fir.ref<!fir.type<_QFtestTmatrix{element:!fir.array<2x2xf32>,solve:!fir.boxproc<() -> ()>}>>, !fir.field) -> !fir.ref<!fir.boxproc<() -> ()>>
  %3 = fir.convert %arg0 : (!fir.boxproc<(!fir.ref<i32>) -> f32>) ->  !fir.boxproc<() -> ()>
  fir.store %3 to %2 : !fir.ref<!fir.boxproc<() -> ()>>
  %4 = fir.field_index solve, !fir.type<_QFtestTmatrix{element:!fir.array<2x2xf32>,solve:!fir.boxproc<() -> ()>}>
  %5 = fir.coordinate_of %0, %4 : (!fir.ref<!fir.type<_QFtestTmatrix{element:!fir.array<2x2xf32>,solve:!fir.boxproc<() -> ()>}>>, !fir.field) -> !fir.ref<!fir.boxproc<() -> ()>>
  %6 = fir.load %5 : !fir.ref<!fir.boxproc<() -> ()>>
  %7 = fir.convert %6 : (!fir.boxproc<() -> ()>) -> !fir.boxproc<(!fir.ref<i32>) -> f32>
  %8 = fir.box_addr %7 : (!fir.boxproc<(!fir.ref<i32>) -> f32>) -> ((!fir.ref<i32>) -> f32)
  %9 = fir.call %8(%arg1) : (!fir.ref<i32>) -> f32
  return

// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.type<_QFtestTmatrixUnboxProc{element:!fir.array<2x2xf32>,solve:() -> ()}>
// CHECK:           %[[VAL_3:.*]] = fir.field_index solve, !fir.type<_QFtestTmatrixUnboxProc{element:!fir.array<2x2xf32>,solve:() -> ()}>
// CHECK:           %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_3]] : (!fir.ref<!fir.type<_QFtestTmatrixUnboxProc{element:!fir.array<2x2xf32>,solve:() -> ()}>>, !fir.field) -> !fir.ref<() -> ()>
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_0]] : ((!fir.ref<i32>) -> f32) -> (() -> ())
// CHECK:           fir.store %[[VAL_5]] to %[[VAL_4]] : !fir.ref<() -> ()>
// CHECK:           %[[VAL_6:.*]] = fir.field_index solve, !fir.type<_QFtestTmatrixUnboxProc{element:!fir.array<2x2xf32>,solve:() -> ()}>
// CHECK:           %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_6]] : (!fir.ref<!fir.type<_QFtestTmatrixUnboxProc{element:!fir.array<2x2xf32>,solve:() -> ()}>>, !fir.field) -> !fir.ref<() -> ()>
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<() -> ()>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (() -> ()) -> ((!fir.ref<i32>) -> f32)
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : ((!fir.ref<i32>) -> f32) -> ((!fir.ref<i32>) -> f32)
// CHECK:           %[[VAL_11:.*]] = fir.call %[[VAL_10]](%[[VAL_1]]) : (!fir.ref<i32>) -> f32

}

// CHECK-LABEL: fir.global internal @ProcedurePointer : (!fir.ref<i32>) -> f32 {
fir.global internal @ProcedurePointer : !fir.boxproc<(!fir.ref<i32>) -> f32> {
  %0 = fir.zero_bits (!fir.ref<i32>) -> f32
  %1 = fir.emboxproc %0 : ((!fir.ref<i32>) -> f32) -> !fir.boxproc<(!fir.ref<i32>) -> f32>
  fir.has_value %1 : !fir.boxproc<(!fir.ref<i32>) -> f32>

// CHECK: %[[VAL_0:.*]] = fir.zero_bits (!fir.ref<i32>) -> f32
// CHECK: %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : ((!fir.ref<i32>) -> f32) -> ((!fir.ref<i32>) -> f32)
// CHECK: fir.has_value %[[VAL_1]] : (!fir.ref<i32>) -> f32
}

// CHECK-LABEL: func.func @proc_pointer_global() {
func.func @proc_pointer_global() {
  %0 = fir.address_of(@ProcedurePointer) : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>
  %1 = fir.load %0 : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> f32>>
  return

// CHECK: %[[VAL_0:.*]] = fir.address_of(@ProcedurePointer) : !fir.ref<(!fir.ref<i32>) -> f32>
// CHECK: %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<(!fir.ref<i32>) -> f32>
}

func.func @block_arg_rewrite(%arg0: !fir.ref<!fir.type<t{p:!fir.boxproc<() -> ()>}>>,  %cdt : i1) {
  cf.cond_br %cdt, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  cf.br ^bb3(%arg0 : !fir.ref<!fir.type<t{p:!fir.boxproc<() -> ()>}>>)
^bb2:  // pred: ^bb0
  %8 = fir.absent !fir.ref<!fir.type<t{p:!fir.boxproc<() -> ()>}>>
  cf.br ^bb3(%8 : !fir.ref<!fir.type<t{p:!fir.boxproc<() -> ()>}>>)
^bb3(%9: !fir.ref<!fir.type<t{p:!fir.boxproc<() -> ()>}>>):  // 2 preds: ^bb1, ^bb2
  cf.br ^bb4
^bb4:  // pred: ^bb3
  fir.call @dosomething(%9) : (!fir.ref<!fir.type<t{p:!fir.boxproc<() -> ()>}>>) -> ()
  return
}
// CHECK-LABEL:   func.func @block_arg_rewrite(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.ref<!fir.type<tUnboxProc{p:() -> ()}>>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: i1) {
// CHECK:           cf.cond_br %[[VAL_1]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb3(%[[VAL_0]] : !fir.ref<!fir.type<tUnboxProc{p:() -> ()}>>)                           
// CHECK:         ^bb2:
// CHECK:           %[[VAL_2:.*]] = fir.absent !fir.ref<!fir.type<tUnboxProc{p:() -> ()}>>                         
// CHECK:           cf.br ^bb3(%[[VAL_2]] : !fir.ref<!fir.type<tUnboxProc{p:() -> ()}>>)                           
// CHECK:         ^bb3(%[[VAL_3:.*]]: !fir.ref<!fir.type<tUnboxProc{p:() -> ()}>>):
// CHECK:           cf.br ^bb4
// CHECK:         ^bb4:
// CHECK:           fir.call @dosomething(%[[VAL_3]]) : (!fir.ref<!fir.type<tUnboxProc{p:() -> ()}>>) -> ()

func.func private @dosomething(!fir.ref<!fir.type<t{p:!fir.boxproc<() -> ()>}>>)

!t2 = !fir.type<t2{i:!fir.boxproc<() -> ()>}>
func.func @box_addr_test(%arg0: !fir.box<!t2>) -> !fir.ref<!t2> {
  %3 = fir.box_addr %arg0 : (!fir.box<!t2>) -> !fir.ref<!t2>
  return %3 : !fir.ref<!t2>
}
// CHECK: func.func @box_addr_test(
// CHECK:   fir.box_addr %{{.*}} : (!fir.box<!fir.type<t2UnboxProc{i:() -> ()}>>) -> !fir.ref<!fir.type<t2UnboxProc{i:() -> ()}>>
