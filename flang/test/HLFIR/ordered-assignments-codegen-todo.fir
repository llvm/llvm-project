// Just test that Ordered assignment pass TODOs are properly reported.
// RUN: %not_todo_cmd fir-opt --lower-hlfir-ordered-assignments %s 2>&1 | FileCheck %s


// CHECK: not yet implemented: FORALL construct or statement in HLFIR

func.func @forall_todo(%arg0: !fir.ref<!fir.array<10xf32>>, %arg1: !fir.ref<!fir.array<10xf32>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  hlfir.forall lb {
    hlfir.yield %c1 : index
  } ub {
    hlfir.yield %c10 : index
  }  (%arg2: i64) {
    hlfir.region_assign {
      %1 = hlfir.designate %arg1 (%arg2)  : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
      hlfir.yield %1 : !fir.ref<f32>
    } to {
      %1 = hlfir.designate %arg0 (%arg2)  : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
      hlfir.yield %1 : !fir.ref<f32>
    }
  }
  return
}
