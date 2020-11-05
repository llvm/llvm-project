// RUN: tco %s | FileCheck %s

// CHECK-LABEL: @x
func @x(%lb : index, %ub : index, %step : index, %b : i1, %addr : !fir.ref<index>) {
  // CHECK: [[LOOP:[0-9]+]]:
  // CHECK: %[[COND:.*]] = icmp sgt i64 %{{.*}}, 0
  // CHECK: br i1 %[[COND]]
  fir.do_loop %iv = %lb to %ub step %step unordered {
    // expect following conditional blocks to get fused
    // CHECK: select i1 %
    fir.if %b {
      // CHECK: store i64
      fir.store %iv to %addr : !fir.ref<index>
    } else {
      %zero = constant 0 : index
      fir.store %zero to %addr : !fir.ref<index>
    }
    // CHECK: br label %[[LOOP]]
  }
  // CHECK: ret void
  return
}

func @f2() -> i1

// CHECK-LABEL: @x2
func @x2(%lo : index, %up : index, %ok : i1) {
  %c1 = constant 1 : index
  // CHECK-DAG: %[[count:.*]] = phi i64
  // CHECK-DAG: %[[exit:.*]] = phi i1
  // CHECK: %[[cond:.*]] = icmp slt i64 %[[count]], %
  // CHECK: %[[and:.*]] = and i1 %[[exit]], %[[cond]]
  // CHECK: br i1 %[[and]]
  %unused = fir.iterate_while (%i = %lo to %up step %c1) and (%ok1 = %ok) {
    %ok2 = fir.call @f2() : () -> i1
    fir.result %ok2 : i1
  }
  // CHECK: ret
  return
}

func @f3(i16)

// do_loop with an extra loop-carried value
// CHECK-LABEL: @x3
func @x3(%lo : index, %up : index) -> i1 {
  %c1 = constant 1 : index
  %ok1 = constant true
  // CHECK-DAG: %[[ok:.*]] = phi i1
  // CHECK-DAG: %[[count:.*]] = phi i64
  // CHECK: = icmp sgt i64 %[[count]], 0
  %ok2 = fir.do_loop %i = %lo to %up step %c1 iter_args(%j = %ok1) -> i1 {
    %ok = fir.call @f2() : () -> i1
    fir.result %ok : i1
    // CHECK: = sub i64 %[[count]], 1
  }
  // CHECK: ret i1 %[[ok]]
  return %ok2 : i1
}

// iterate_while with an extra loop-carried value
// CHECK-LABEL: @y3
func @y3(%lo : index, %up : index) -> i1 {
  %c1 = constant 1 : index
  %ok1 = constant true
  // CHECK: %[[ok4:.*]] = call i1 @f2()
  %ok4 = fir.call @f2() : () -> i1
  // CHECK-DAG: %[[count:.*]] = phi i64
  // CHECK-DAG: %[[ok3:.*]] = phi i1 {{.*}}[ true
  // CHECK-DAG: %[[j:.*]] = phi i1 {{.*}}[ %[[ok4]]
  // CHECK: %[[prev:.*]] = icmp slt i64 %[[count]],
  // CHECK: = and i1 %[[ok3]], %[[prev]]
  %ok2:2 = fir.iterate_while (%i = %lo to %up step %c1) and (%ok3 = %ok1) iter_args(%j = %ok4) -> i1 {
    %ok = fir.call @f2() : () -> i1
    fir.result %ok3, %ok : i1, i1
    // CHECK: = add i64 %[[count]], 1
  }
  // CHECK: %[[result:.*]] = and i1 %[[ok3]], %[[j]]
  %andok = and %ok2#0, %ok2#1 : i1
  // CHECK: ret i1 %[[result]]
  return %andok : i1
}

func @f4(i32) -> i1

// do_loop that returns the final value of the induction
// CHECK-LABEL: @x4
// CHECK-SAME: (i64 %[[lo:.*]],
func @x4(%lo : index, %up : index) -> index {
  %c1 = constant 1 : index
  // CHECK: %[[top:.*]] = add i64
  // CHECK-DAG: %[[i:.*]] = phi i64 {{.*}}[ %[[lo]],
  // CHECK-DAG: %[[count:.*]] = phi i64 {{.*}}[ %[[top]],
  // CHECK: icmp sgt i64 %[[count]],
  %v = fir.do_loop %i = %lo to %up step %c1 -> index {
    // CHECK: trunc i64 %[[i]] to i32
    %i1 = fir.convert %i : (index) -> i32
    // CHECK: call i1 @f4
    %ok = fir.call @f4(%i1) : (i32) -> i1
    fir.result %i : index
  }
  // CHECK: ret i64 %[[i]]
  return %v : index
}

// iterate_while that returns the final value of both inductions
// CHECK-LABEL: @y4
func @y4(%lo : index, %up : index) -> index {
  %c1 = constant 1 : index
  %ok1 = constant true
  // CHECK-DAG: %[[i:.*]] = phi i64 [
  // CHECK-DAG: %[[ok2:.*]] = phi i1 [
  // CHECK: icmp slt i64 %[[i]]
  // CHECK: and i1
  %v:2 = fir.iterate_while (%i = %lo to %up step %c1) and (%ok2 = %ok1) -> (index, i1) {
    %i1 = fir.convert %i : (index) -> i32
    // CHECK: call i1 @f4
    %ok = fir.call @f4(%i1) : (i32) -> i1
    fir.result %i, %ok : index, i1
  }
  // CHECK: ret i64 %[[i]]
  return %v#0 : index
}

// do_loop that returns the final induction value
// and an extra loop-carried value
// CHECK-LABEL: @x5
// CHECK-SAME: (i64 %[[lo:.*]],
func @x5(%lo : index, %up : index) -> index {
  %c1 = constant 1 : index
  // CHECK: %[[top:.*]] = add i64
  %s1 = constant 42 : i16
  // CHECK-DAG: %[[i:.*]] = phi i64 {{.*}}[ %[[lo]],
  // CHECK-DAG: %[[count:.*]] = phi i64 {{.*}}[ %[[top]],
  // CHECK-DAG: %[[s:.*]] = phi i16
  // CHECK: icmp sgt i64 %[[count]]
  %v:2 = fir.do_loop %i = %lo to %up step %c1 iter_args(%s = %s1) -> (index, i16) {
    // CHECK: call i1 @f2
    %ok = fir.call @f2() : () -> i1
    %s2 = fir.convert %ok : (i1) -> i16
    fir.result %i, %s2 : index, i16
    // CHECK: add i64 %[[i]], 1
    // CHECK: sub i64 %[[count]], 1
  }
  // CHECK: call void @f3
  fir.call @f3(%v#1) : (i16) -> ()
  // CHECK: ret i64 %[[i]]
  return %v#0 : index
}

// iterate_while that returns the both induction values
// and an extra loop-carried value
// CHECK-LABEL: @y5
// CHECK-SAME: (i64 %[[lo:.*]],
func @y5(%lo : index, %up : index) -> index {
  %c1 = constant 1 : index
  %s1 = constant 42 : i16
  %ok1 = constant true
  // CHECK-DAG: %[[i:.*]] = phi i64 {{.*}}[ %[[lo]],
  // CHECK-DAG: %[[ok2:.*]] = phi i1 {{.*}}[ true, 
  // CHECK-DAG: %[[s:.*]] = phi i16 {{.*}}[ 42,
  // CHECK: icmp slt i64 %[[i]]
  // CHECK: and i1 %[[ok2]]
  %v:3 = fir.iterate_while (%i = %lo to %up step %c1) and (%ok2 = %ok1) iter_args(%s = %s1) -> (index, i1, i16) {
    // CHECK: call i1 @f2
    %ok = fir.call @f2() : () -> i1
    %s2 = fir.convert %ok : (i1) -> i16
    fir.result %i, %ok, %s2 : index, i1, i16
    // CHECK: add i64 %[[i]], 1
  }
  // CHECK: br i1 %[[ok2]],
  fir.if %v#1 {
    %arg = constant 0 : i32
    // CHECK: call i1 @f4
    %ok4 = fir.call @f4(%arg) : (i32) -> i1
  }
  // CHECK: call void @f3(i16 %[[s]])
  fir.call @f3(%v#2) : (i16) -> ()
  // CHECK: ret i64 %[[i]]
  return %v#0 : index
}
