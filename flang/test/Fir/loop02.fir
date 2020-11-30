// RUN: tco --always-execute-loop-body %s | FileCheck %s

// CHECK-LABEL: @x
func @x(%addr : !fir.ref<index>) {
  %bound = constant 452 : index
  %step = constant 1 : index
  // CHECK: %[[phi:.*]] = phi i64 [{{.*}}], [ 1,
  // CHECK: = icmp sgt i64 %[[phi]], 0
  fir.do_loop %iv = %bound to %bound step %step {
    // CHECK: call void @y(i64* %
    fir.call @y(%addr) : (!fir.ref<index>) -> ()
  }
  // CHECK: ret void
  return
}

func private @y(%addr : !fir.ref<index>)
