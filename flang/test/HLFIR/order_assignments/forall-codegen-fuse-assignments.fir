// Test code generation of hlfir.forall when assignment fusing is enabled
// and possible.
// RUN: fir-opt %s --lower-hlfir-ordered-assignments=fuse-assignments=true | FileCheck %s --check-prefix=FUSE
// RUN: fir-opt %s --lower-hlfir-ordered-assignments=fuse-assignments=false | FileCheck %s --check-prefix=NOFUSE

func.func @test_assignment_fusing(%x: !fir.ref<!fir.array<10xi32>>, %y : !fir.box<!fir.array<?xi32>>) {
  %c42 = arith.constant 42 : i32
  hlfir.forall lb {
    %c1 = arith.constant 1 : index
    hlfir.yield %c1 : index
  } ub {
    %c10 = arith.constant 10 : index
    hlfir.yield %c10 : index
  }  (%i: index) {
    hlfir.region_assign {
      hlfir.yield %c42 : i32
    } to {
      %2 = hlfir.designate %x (%i)  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
      hlfir.yield %2 : !fir.ref<i32>
    }
    hlfir.region_assign {
      hlfir.yield %c42 : i32
    } to {
      %2 = hlfir.designate %y (%i)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      hlfir.yield %2 : !fir.ref<i32>
    }
  }
  return
}
// FUSE-LABEL:   func.func @test_assignment_fusing(
// FUSE-SAME:                                      %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>,
// FUSE-SAME:                                      %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>) {
// FUSE:           %[[VAL_2:.*]] = arith.constant 42 : i32
// FUSE:           %[[VAL_3:.*]] = arith.constant 1 : index
// FUSE:           %[[VAL_4:.*]] = arith.constant 10 : index
// FUSE:           %[[VAL_5:.*]] = arith.constant 1 : index
// FUSE:           fir.do_loop %[[VAL_6:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_5]] {
// FUSE-NEXT:         %[[VAL_7:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_6]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
// FUSE-NEXT:         hlfir.assign %[[VAL_2]] to %[[VAL_7]] : i32, !fir.ref<i32>
// FUSE-NEXT:         %[[VAL_8:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_6]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// FUSE-NEXT:         hlfir.assign %[[VAL_2]] to %[[VAL_8]] : i32, !fir.ref<i32>
// FUSE-NEXT:      }
// FUSE-NEXT:      return

// NOFUSE-LABEL: func.func @test_assignment_fusing(
// NOFUSE:         fir.do_loop
// NOFUSE:           hlfir.assign
// NOFUSE:         }
// NOFUSE:         fir.do_loop
// NOFUSE:           hlfir.assign
// NOFUSE:         }
