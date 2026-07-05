// Test code generation of hlfir.forall, hlfir.forall_index, and hlfir.forall_mask.
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

func.func @test_simple(%x: !fir.ref<!fir.array<10xi32>>) {
  hlfir.forall lb {
    %c1 = arith.constant 1 : index
    hlfir.yield %c1 : index
  } ub {
    %c10 = arith.constant 10 : index
    hlfir.yield %c10 : index
  }  (%i: index) {
    hlfir.region_assign {
      %c42 = arith.constant 42 : i32
      hlfir.yield %c42 : i32
    } to {
      %2 = hlfir.designate %x (%i)  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
      hlfir.yield %2 : !fir.ref<i32>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_simple(
// CHECK-SAME:                           %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 42 : i32
// CHECK:           fir.do_loop %[[VAL_5:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_3]] {
// CHECK:             %[[VAL_6:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_5]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_4]] to %[[VAL_6]] : i32, !fir.ref<i32>
// CHECK:           }

func.func @test_index(%x: !fir.ref<!fir.array<10xi32>>) {
  hlfir.forall lb {
    %c1 = arith.constant 1 : index
    hlfir.yield %c1 : index
  } ub {
    %c10 = arith.constant 10 : index
    hlfir.yield %c10 : index
  }  (%arg1: i32) {
    %i = hlfir.forall_index "i" %arg1 : (i32) -> !fir.ref<i32>
    hlfir.region_assign {
      %i_load = fir.load %i : !fir.ref<i32>
      hlfir.yield %i_load : i32
    } to {
      %2 = hlfir.designate %x (%arg1)  : (!fir.ref<!fir.array<10xi32>>, i32) -> !fir.ref<i32>
      hlfir.yield %2 : !fir.ref<i32>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_index(
// CHECK-SAME:                          %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) {
// CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "i"}
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_5:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] {
// CHECK:             %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (index) -> i32
// CHECK:             fir.store %[[VAL_6]] to %[[VAL_1]] : !fir.ref<i32>
// CHECK:             %[[VAL_7:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
// CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_6]])  : (!fir.ref<!fir.array<10xi32>>, i32) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_7]] to %[[VAL_8]] : i32, !fir.ref<i32>
// CHECK:           }


func.func @split_schedule(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.box<!fir.array<?xf32>>, %arg2: !fir.box<!fir.array<?x?xf32>>) {
  %c11 = arith.constant 11 : i64
  %c10 = arith.constant 10 : i64
  %c1 = arith.constant 1 : i64
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "y"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %2:2 = hlfir.declare %arg2 {uniq_name = "z"} : (!fir.box<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  hlfir.forall lb {
    hlfir.yield %c1 : i64
  } ub {
    hlfir.yield %c10 : i64
  }  (%arg3: i64) {
    hlfir.region_assign {
      %3 = hlfir.designate %1#0 (%arg3)  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
      %4 = fir.load %3 : !fir.ref<f32>
      hlfir.yield %4 : f32
    } to {
      %3 = hlfir.designate %0#0 (%arg3)  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
      hlfir.yield %3 : !fir.ref<f32>
    }
    hlfir.forall lb {
      hlfir.yield %c1 : i64
    } ub {
      hlfir.yield %c10 : i64
    }  (%arg4: i64) {
      hlfir.region_assign {
        %3 = arith.subi %c11, %arg3 : i64
        %4 = hlfir.designate %0#0 (%3)  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
        %5 = fir.load %4 : !fir.ref<f32>
        hlfir.yield %5 : f32
      } to {
        %3 = hlfir.designate %2#0 (%arg3, %arg4)  : (!fir.box<!fir.array<?x?xf32>>, i64, i64) -> !fir.ref<f32>
        hlfir.yield %3 : !fir.ref<f32>
      }
    }
  }
  return
}
// CHECK-LABEL:   func.func @split_schedule(
// CHECK:           %[[VAL_3:.*]] = arith.constant 11 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "y"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "z"} : (!fir.box<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_12:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_11]] {
// CHECK:             %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (index) -> i64
// CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_13]])  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
// CHECK:             %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<f32>
// CHECK:             %[[VAL_16:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_13]])  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
// CHECK:             hlfir.assign %[[VAL_15]] to %[[VAL_16]] : f32, !fir.ref<f32>
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
// CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
// CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
// CHECK:           %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_20:.*]] = %[[VAL_17]] to %[[VAL_18]] step %[[VAL_19]] {
// CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (index) -> i64
// CHECK:             fir.do_loop %[[VAL_25:.*]] = %[[VAL_22]] to %[[VAL_23]] step %[[VAL_24]] {
// CHECK:               %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (index) -> i64
// CHECK:               %[[VAL_27:.*]] = arith.subi %[[VAL_3]], %[[VAL_21]] : i64
// CHECK:               %[[VAL_28:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_27]])  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
// CHECK:               %[[VAL_29:.*]] = fir.load %[[VAL_28]] : !fir.ref<f32>
// CHECK:               %[[VAL_30:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_21]], %[[VAL_26]])  : (!fir.box<!fir.array<?x?xf32>>, i64, i64) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_29]] to %[[VAL_30]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }

func.func @test_mask(%arg0: !fir.box<!fir.array<?x?xf32>>, %arg1: !fir.box<!fir.array<?x?xf32>>, %arg2: !fir.box<!fir.array<?x!fir.logical<4>>>) {
  %c10 = arith.constant 10 : i64
  %c1 = arith.constant 1 : i64
  %0:2 = hlfir.declare %arg2 {uniq_name = "mask"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %1:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "y"} : (!fir.box<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  hlfir.forall lb {
    hlfir.yield %c1 : i64
  } ub {
    hlfir.yield %c10 : i64
  }  (%arg3: i64) {
    hlfir.forall_mask {
      %3 = hlfir.designate %0#0 (%arg3)  : (!fir.box<!fir.array<?x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
      %4 = fir.load %3 : !fir.ref<!fir.logical<4>>
      %5 = fir.convert %4 : (!fir.logical<4>) -> i1
      hlfir.yield %5 : i1
    } do {
      hlfir.forall lb {
        hlfir.yield %c1 : i64
      } ub {
        hlfir.yield %arg3 : i64
      }  (%arg4: i64) {
        hlfir.region_assign {
          %3 = hlfir.designate %2#0 (%arg3, %arg4)  : (!fir.box<!fir.array<?x?xf32>>, i64, i64) -> !fir.ref<f32>
          %4 = fir.load %3 : !fir.ref<f32>
          hlfir.yield %4 : f32
        } to {
          %3 = hlfir.designate %1#0 (%arg3, %arg4)  : (!fir.box<!fir.array<?x?xf32>>, i64, i64) -> !fir.ref<f32>
          hlfir.yield %3 : !fir.ref<f32>
        }
      }
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_mask(
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "mask"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "x"} : (!fir.box<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "y"} : (!fir.box<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
// CHECK:           %[[VAL_10:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
// CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_10]] {
// CHECK:             %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (index) -> i64
// CHECK:             %[[VAL_13:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_12]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.logical<4>) -> i1
// CHECK:             fir.if %[[VAL_15]] {
// CHECK:               %[[VAL_17:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
// CHECK:               fir.do_loop %[[VAL_19:.*]] = %[[VAL_16]] to %[[VAL_17]] step %[[VAL_18]] {
// CHECK:                 %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (index) -> i64
// CHECK:                 %[[VAL_21:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_12]], %[[VAL_20]])  : (!fir.box<!fir.array<?x?xf32>>, i64, i64) -> !fir.ref<f32>
// CHECK:                 %[[VAL_22:.*]] = fir.load %[[VAL_21]] : !fir.ref<f32>
// CHECK:                 %[[VAL_23:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_12]], %[[VAL_20]])  : (!fir.box<!fir.array<?x?xf32>>, i64, i64) -> !fir.ref<f32>
// CHECK:                 hlfir.assign %[[VAL_22]] to %[[VAL_23]] : f32, !fir.ref<f32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
