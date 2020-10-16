// RUN: tco %s | FileCheck %s

// CHECK-LINE: define void @f1
// CHECK: (float* %[[A:[^,]*]], {{.*}}, float %[[F:.*]])
func @f1(%a : !fir.ref<!fir.array<?x?xf32>>, %n : index, %m : index, %o : index, %p : index, %f : f32) {
  %c1 = constant 1 : index
  %s = fir.shape_shift %o, %n, %p, %m : (index, index, index, index) -> !fir.shapeshift<2>
  %vIn = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  // CHECK: = icmp sgt
  %r = fir.do_loop %j = %p to %m step %c1 iter_args(%v1 = %vIn) -> !fir.array<?x?xf32> {
    // CHECK: = icmp sgt
    %r = fir.do_loop %i = %o to %n step %c1 iter_args(%v = %v1) -> !fir.array<?x?xf32> {
      // CHECK: %[[AOFF:.*]] = getelementptr float, float* %[[A]], i64
      // CHECK: store float %[[F]], float* %[[AOFF]]
      %r = fir.array_update %v, %f, %i, %j : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
      fir.result %r : !fir.array<?x?xf32>
    }
    fir.result %r : !fir.array<?x?xf32>
  }
  fir.array_merge_store %vIn, %r to %a : !fir.ref<!fir.array<?x?xf32>>
  // CHECK: ret void
  return
}

// CHECK-LINE: define void @f2
// CHECK: (float* %[[A:[^,]*]], {{.*}}, float %[[F:.*]])
func @f2(%a : !fir.ref<!fir.array<?x?xf32>>, %b : !fir.ref<!fir.array<?x?xf32>>, %n : index, %m : index, %o : index, %p : index, %f : f32) {
  %c1 = constant 1 : index
  %s = fir.shape_shift %o, %n, %p, %m : (index, index, index, index) -> !fir.shapeshift<2>
  %vIn = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  %wIn = fir.array_load %b(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  // CHECK: = icmp sgt
  %r = fir.do_loop %j = %p to %m step %c1 iter_args(%v1 = %vIn) -> !fir.array<?x?xf32> {
    // CHECK: = icmp sgt
    %r = fir.do_loop %i = %o to %n step %c1 iter_args(%v = %v1) -> !fir.array<?x?xf32> {
      %x = fir.array_fetch %wIn, %i, %j : (!fir.array<?x?xf32>, index, index) -> f32
      %y = addf %x, %f : f32
      // CHECK: %[[AOFF:.*]] = getelementptr float, float* %[[A]], i64
      %r = fir.array_update %v, %y, %i, %j : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
      fir.result %r : !fir.array<?x?xf32>
    }
    fir.result %r : !fir.array<?x?xf32>
  }
  fir.array_merge_store %vIn, %r to %a : !fir.ref<!fir.array<?x?xf32>>
  // CHECK: ret void
  return
}

// CHECK-LINE: define void @f3
// CHECK: (float* %[[A:[^,]*]], {{.*}}, float %[[F:.*]])
func @f3(%a : !fir.ref<!fir.array<?x?xf32>>, %b : !fir.ref<!fir.array<?x?xf32>>, %n : index, %m : index, %o : index, %p : index, %f : f32) {
  %c1 = constant 1 : index
  %s = fir.shape_shift %o, %n, %p, %m : (index, index, index, index) -> !fir.shapeshift<2>
  %vIn = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  %wIn = fir.array_load %b(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  // CHECK: = icmp sgt
  %r = fir.do_loop %j = %p to %m step %c1 iter_args(%v1 = %vIn) -> !fir.array<?x?xf32> {
    // CHECK: = icmp sgt
    %r = fir.do_loop %i = %o to %n step %c1 iter_args(%v = %v1) -> !fir.array<?x?xf32> {
      %x = fir.array_fetch %wIn, %i, %j : (!fir.array<?x?xf32>, index, index) -> f32
      %y = addf %x, %f : f32
      // CHECK: %[[AOFF:.*]] = getelementptr float, float* %[[A]], i64
      %i2 = addi %i, %c1 : index
      %r = fir.array_update %v, %y, %i2, %j : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
      fir.result %r : !fir.array<?x?xf32>
    }
    fir.result %r : !fir.array<?x?xf32>
  }
  fir.array_merge_store %vIn, %r to %a : !fir.ref<!fir.array<?x?xf32>>
  // CHECK: ret void
  return
}

// CHECK-LINE: define void @f4
// CHECK: (float* %[[A:[^,]*]], {{.*}}, float %[[F:.*]])
func @f4(%a : !fir.ref<!fir.array<?x?xf32>>, %b : !fir.ref<!fir.array<?x?xf32>>, %n : index, %m : index, %o : index, %p : index, %f : f32) {
  %c1 = constant 1 : index
  %s = fir.shape_shift %o, %n, %p, %m : (index, index, index, index) -> !fir.shapeshift<2>
  %vIn = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  %wIn = fir.array_load %b(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  // CHECK: = icmp sgt
  %r = fir.do_loop %j = %p to %m step %c1 iter_args(%v1 = %vIn) -> !fir.array<?x?xf32> {
    // CHECK: = icmp sgt
    %r = fir.do_loop %i = %o to %n step %c1 iter_args(%v = %v1) -> !fir.array<?x?xf32> {
      %x2 = fir.array_fetch %vIn, %i, %j : (!fir.array<?x?xf32>, index, index) -> f32
      %x = fir.array_fetch %wIn, %i, %j : (!fir.array<?x?xf32>, index, index) -> f32
      %y = addf %x, %f : f32
      %y2 = addf %y, %x2 : f32
      // CHECK: %[[AOFF:.*]] = getelementptr float, float* %[[A]], i64
      %i2 = addi %i, %c1 : index
      %r = fir.array_update %v, %y2, %i2, %j : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
      fir.result %r : !fir.array<?x?xf32>
    }
    fir.result %r : !fir.array<?x?xf32>
  }
  fir.array_merge_store %vIn, %r to %a : !fir.ref<!fir.array<?x?xf32>>
  // CHECK: ret void
  return
}

