// RUN: tco %s | FileCheck %s

// the fir.embox in the init region is turned into an alloca for the box. Test
// that CodeGen.cpp knows where to place an alloca when it is inside of an
// omp.reduction.declare

// regretably this has to be nonsense IR because we need the subsequent patches
// to process anything useful

omp.reduction.declare @test_reduction : !fir.ref<!fir.box<i32>> init {
^bb0(%arg0: !fir.ref<!fir.box<i32>>):
  %0 = fir.alloca !fir.box<i32>
  %1 = fir.alloca i32
  %2 = fir.embox %1 : (!fir.ref<i32>) -> !fir.box<i32>

  // use the embox for something so it isn't removed
  fir.store %2 to %0 : !fir.ref<!fir.box<i32>>

  omp.yield(%0 : !fir.ref<!fir.box<i32>>)
} combiner {
^bb0(%arg0: !fir.ref<!fir.box<i32>>, %arg1: !fir.ref<!fir.box<i32>>):
  %0 = fir.undefined !fir.ref<!fir.box<i32>>
  omp.yield(%0 : !fir.ref<!fir.box<i32>>)
}

func.func @_QQmain() attributes {fir.bindc_name = "reduce"} {
  %4 = fir.alloca !fir.box<i32>
  omp.parallel byref reduction(@test_reduction %4 -> %arg0 : !fir.ref<!fir.box<i32>>) {
    omp.terminator
  }
  return
}

// basically we are testing that there isn't a crash
// CHECK-LABEL: define void @_QQmain
// CHECK-NEXT:    alloca { ptr, i64, i32, i8, i8, i8, i8 }, i64 1, align 8
