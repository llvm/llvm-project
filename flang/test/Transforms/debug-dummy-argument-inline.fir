// Tests that inlined calls arguments are not mistaken for the arguments of the
// procedure where the calls were inlined.
// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s -o - | FileCheck %s

// CHECK: #di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "i", file = #di_file, line = 6, arg = 1, type = #di_basic_type>
// CHECK: #di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram, name = "i", file = #di_file, line = 1, type = #di_basic_type>

func.func @foo_(%arg0: !fir.ref<i32> {fir.bindc_name = "i"} loc("debug-dummy-argument-inline.f90":5:1)) attributes {fir.internal_name = "_QPfoo"} {
  %0 = fir.undefined !fir.dscope loc(#loc5)
  %1 = fircg.ext_declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFfooEi"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc6)
  %2 = fir.undefined !fir.dscope loc(#loc11)
  %3 = fircg.ext_declare %1 dummy_scope %2 arg 1 {uniq_name = "_QFbarEi"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc12)
  fir.call @buzz_(%3) fastmath<contract> : (!fir.ref<i32>) -> () loc(#loc13)
  %4 = fir.undefined !fir.dscope loc(#loc14)
  %5 = fircg.ext_declare %1 dummy_scope %4 arg 1 {uniq_name = "_QFbarEi"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc15)
  fir.call @buzz_(%5) fastmath<contract> : (!fir.ref<i32>) -> () loc(#loc16)
  return loc(#loc9)
} loc(#loc5)
func.func private @buzz_(!fir.ref<i32>) attributes {fir.internal_name = "_QPbuzz"} loc(#loc10)
#loc = loc("debug-dummy-argument-inline.f90":0:0)
#loc1 = loc("debug-dummy-argument-inline.f90":1:1)
#loc2 = loc("debug-dummy-argument-inline.f90":2:14)
#loc3 = loc("debug-dummy-argument-inline.f90":3:3)
#loc4 = loc("debug-dummy-argument-inline.f90":4:1)
#loc5 = loc("debug-dummy-argument-inline.f90":5:1)
#loc6 = loc("debug-dummy-argument-inline.f90":6:14)
#loc7 = loc("debug-dummy-argument-inline.f90":7:3)
#loc8 = loc("debug-dummy-argument-inline.f90":8:3)
#loc9 = loc("debug-dummy-argument-inline.f90":9:1)
#loc10 = loc("debug-dummy-argument-inline.f90":3:8)
#loc11 = loc(callsite(#loc1 at #loc7))
#loc12 = loc(callsite(#loc2 at #loc7))
#loc13 = loc(callsite(#loc3 at #loc7))
#loc14 = loc(callsite(#loc1 at #loc8))
#loc15 = loc(callsite(#loc2 at #loc8))
#loc16 = loc(callsite(#loc3 at #loc8))
