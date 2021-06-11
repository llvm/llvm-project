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
  fir.array_merge_store %vIn, %r to %a : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.ref<!fir.array<?x?xf32>>
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
  fir.array_merge_store %vIn, %r to %a : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.ref<!fir.array<?x?xf32>>
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
  fir.array_merge_store %vIn, %r to %a : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.ref<!fir.array<?x?xf32>>
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
  fir.array_merge_store %vIn, %r to %a : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.ref<!fir.array<?x?xf32>>
  // CHECK: ret void
  return
}

// Array expression assignment with potentially non contiguous arrays (e.g.
// `a = b + f`, with and v assumed shapes.
// Tests that the stride from the descriptor is used.
// CHECK-LINE: define void @f6
// CHECK: ({ float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[A:[^,]*]], { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[B:[^,]*]], float %[[F:.*]])
func @f5(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.box<!fir.array<?xf32>>, %arg2: f32) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  %1 = subi %0#1, %c1 : index
  %2 = fir.array_load %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  %3 = fir.array_load %arg1 : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  // CHECK: icmp sgt
  %4 = fir.do_loop %arg3 = %c0 to %1 step %c1 iter_args(%arg4 = %2) -> (!fir.array<?xf32>) {
    // CHECK: %[[B_STRIDE_GEP:.*]] = getelementptr {{.*}}* %[[B]], i32 0, i32 7, i64 0, i32 2
    // CHECK: %[[B_STRIDE:.*]] = load i64, i64* %[[B_STRIDE_GEP]]
    // CHECK: %[[B_DIM_OFFSET:.*]] = mul i64 %{{.*}}, %[[B_STRIDE]]
    // CHECK: %[[B_OFFSET:.*]] =  add i64 %[[B_DIM_OFFSET]], 0
    // CHECK: %[[B_BASE_GEP:.*]] = getelementptr {{.*}}* %1, i32 0, i32 0
    // CHECK: %[[B_BASE:.*]] = load float*, float** %[[B_BASE_GEP]]
    // CHECK: %[[B_VOID_BASE:.*]] = bitcast float* %[[B_BASE]] to i8*
    // CHECK: %[[B_VOID_ADDR:.*]] = getelementptr i8, i8* %[[B_VOID_BASE]], i64 %[[B_OFFSET]]
    // CHECK: %[[B_ADDR:.*]] = bitcast i8* %[[B_VOID_ADDR]] to float*
    // CHECK: %[[B_VAL:.*]] = load float, float* %[[B_ADDR]]
    // CHECK: fadd float %[[B_VAL]], %[[F]]
    %5 = fir.array_fetch %3, %arg3 : (!fir.array<?xf32>, index) -> f32
    %6 = addf %5, %arg2 : f32
    %7 = fir.array_update %arg4, %6, %arg3 : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
    fir.result %7 : !fir.array<?xf32>
  }
  fir.array_merge_store %2, %4 to %arg0 : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.array<?xf32>>
  // CHECK: ret void
  return
}


// Overlapping array expression assignment with a potentially non
// contiguous array (e.g. `a(2:10:1) = a(1:9:1) + f`, with a assumed shape).
// Test that a temp is created.
// CHECK-LINE: define void @f6
// CHECK: ({ float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[A:[^,]*]], float %[[F:.*]])
func @f6(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: f32) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c9 = constant 9 : index
  %c10 = constant 10 : index

  // CHECK: %[[EXT_GEP:.*]] = getelementptr {{.*}} %[[A]], i32 0, i32 7, i64 0, i32 1
  // CHECK: %[[EXTENT:.*]] = load i64, i64* %[[EXT_GEP]]
  // CHECK: %[[SIZE:.*]] = mul i64 4, %[[EXTENT]]
  // CHECK: %[[MALLOC:.*]] = call i8* @malloc(i64 %[[SIZE]])
  // CHECK: %[[TMP:.*]] = bitcast i8* %[[MALLOC]] to float*

  %1 = fir.slice %c2, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.array_load %arg0 [%1] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.array<?xf32>
  %3 = fir.slice %c1, %c9, %c1 : (index, index, index) -> !fir.slice<1>
  %4 = fir.array_load %arg0 [%3] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.array<?xf32>
  %5 = fir.do_loop %arg2 = %c0 to %c9 step %c1 iter_args(%arg3 = %2) -> (!fir.array<?xf32>) {
    %6 = fir.array_fetch %4, %arg2 : (!fir.array<?xf32>, index) -> f32
    %7 = addf %6, %arg1 : f32
    %8 = fir.array_update %arg3, %7, %arg2 : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
    fir.result %8 : !fir.array<?xf32>
  }
  fir.array_merge_store %2, %5 to %arg0[%1] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.array<?xf32>>, !fir.slice<1>
  // CHECK: ret void
  return
}

// Non contiguous array with lower bounds (x = y(100), with y(4:))
// Test array_coor offset computation.
// CHECK-LABEL:  define void @f7(
// CHECK: float* %[[X:[^,]*]], { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[Y:.*]])
func @f7(%arg0: !fir.ref<f32>, %arg1: !fir.box<!fir.array<?xf32>>) {
  %c4 = constant 4 : index
  %c100 = constant 100 : index
  %0 = fir.shift %c4 : (index) -> !fir.shift<1>
  // CHECK: %[[STRIDE_GEP:.*]] = getelementptr {{.*}}* %[[Y]], i32 0, i32 7, i64 0, i32 2
  // CHECK: %[[STRIDE:.*]] = load i64, i64* %[[STRIDE_GEP]]
  // CHECK: mul i64 96, %[[STRIDE]]
  %1 = fir.array_coor %arg1(%0) %c100 : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, index) -> !fir.ref<f32>
  %2 = fir.load %1 : !fir.ref<f32>
  fir.store %2 to %arg0 : !fir.ref<f32>
  return
} 
