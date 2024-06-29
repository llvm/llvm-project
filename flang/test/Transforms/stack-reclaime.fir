// RUN: fir-opt --split-input-file --stack-reclaim %s | FileCheck %s

func.func @alloca_in_loop(%lb : index, %ub : index, %step : index, %b : i1, %addr : !fir.ref<index>) {
  fir.do_loop %iv = %lb to %ub step %step unordered {
    %0 = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
  }
  return
}

// CHECK-LABEL: func.func @alloca_in_loop
// CHECK: fir.do_loop
// CHECK: %[[STACKPTR:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK: %{{.*}} = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
// CHECK: llvm.intr.stackrestore %0 : !llvm.ptr
