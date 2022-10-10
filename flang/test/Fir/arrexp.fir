// RUN: tco %s | FileCheck %s

// CHECK-LINE: define void @f1
// CHECK: (ptr %[[A:[^,]*]], {{.*}}, float %[[F:.*]])
func.func @f1(%a : !fir.ref<!fir.array<?x?xf32>>, %n : index, %m : index, %o : index, %p : index, %f : f32) {
  %c1 = arith.constant 1 : index
  %s = fir.shape_shift %o, %n, %p, %m : (index, index, index, index) -> !fir.shapeshift<2>
  %vIn = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  // CHECK: = icmp sgt
  %r = fir.do_loop %j = %p to %m step %c1 iter_args(%v1 = %vIn) -> !fir.array<?x?xf32> {
    // CHECK: = icmp sgt
    %r = fir.do_loop %i = %o to %n step %c1 iter_args(%v = %v1) -> !fir.array<?x?xf32> {
      // CHECK: %[[AOFF:.*]] = getelementptr float, ptr %[[A]], i64
      // CHECK: store float %[[F]], ptr %[[AOFF]]
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
// CHECK: (ptr %[[A:[^,]*]], {{.*}}, float %[[F:.*]])
func.func @f2(%a : !fir.ref<!fir.array<?x?xf32>>, %b : !fir.ref<!fir.array<?x?xf32>>, %n : index, %m : index, %o : index, %p : index, %f : f32) {
  %c1 = arith.constant 1 : index
  %s = fir.shape_shift %o, %n, %p, %m : (index, index, index, index) -> !fir.shapeshift<2>
  %vIn = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  %wIn = fir.array_load %b(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  // CHECK: = icmp sgt
  %r = fir.do_loop %j = %p to %m step %c1 iter_args(%v1 = %vIn) -> !fir.array<?x?xf32> {
    // CHECK: = icmp sgt
    %r = fir.do_loop %i = %o to %n step %c1 iter_args(%v = %v1) -> !fir.array<?x?xf32> {
      %x = fir.array_fetch %wIn, %i, %j : (!fir.array<?x?xf32>, index, index) -> f32
      %y = arith.addf %x, %f : f32
      // CHECK: %[[AOFF:.*]] = getelementptr float, ptr %[[A]], i64
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
// CHECK: (ptr %[[A:[^,]*]], {{.*}}, float %[[F:.*]])
func.func @f3(%a : !fir.ref<!fir.array<?x?xf32>>, %b : !fir.ref<!fir.array<?x?xf32>>, %n : index, %m : index, %o : index, %p : index, %f : f32) {
  %c1 = arith.constant 1 : index
  %s = fir.shape_shift %o, %n, %p, %m : (index, index, index, index) -> !fir.shapeshift<2>
  %vIn = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  %wIn = fir.array_load %b(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  // CHECK: = icmp sgt
  %r = fir.do_loop %j = %p to %m step %c1 iter_args(%v1 = %vIn) -> !fir.array<?x?xf32> {
    // CHECK: = icmp sgt
    %r = fir.do_loop %i = %o to %n step %c1 iter_args(%v = %v1) -> !fir.array<?x?xf32> {
      %x = fir.array_fetch %wIn, %i, %j : (!fir.array<?x?xf32>, index, index) -> f32
      %y = arith.addf %x, %f : f32
      // CHECK: %[[AOFF:.*]] = getelementptr float, ptr %[[A]], i64
      %i2 = arith.addi %i, %c1 : index
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
// CHECK: (ptr %[[A:[^,]*]], {{.*}}, float %[[F:.*]])
func.func @f4(%a : !fir.ref<!fir.array<?x?xf32>>, %b : !fir.ref<!fir.array<?x?xf32>>, %n : index, %m : index, %o : index, %p : index, %f : f32) {
  %c1 = arith.constant 1 : index
  %s = fir.shape_shift %o, %n, %p, %m : (index, index, index, index) -> !fir.shapeshift<2>
  %vIn = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  %wIn = fir.array_load %b(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  // CHECK: = icmp sgt
  %r = fir.do_loop %j = %p to %m step %c1 iter_args(%v1 = %vIn) -> !fir.array<?x?xf32> {
    // CHECK: = icmp sgt
    %r = fir.do_loop %i = %o to %n step %c1 iter_args(%v = %v1) -> !fir.array<?x?xf32> {
      %x2 = fir.array_fetch %vIn, %i, %j : (!fir.array<?x?xf32>, index, index) -> f32
      %x = fir.array_fetch %wIn, %i, %j : (!fir.array<?x?xf32>, index, index) -> f32
      %y = arith.addf %x, %f : f32
      %y2 = arith.addf %y, %x2 : f32
      // CHECK: %[[AOFF:.*]] = getelementptr float, ptr %[[A]], i64
      %i2 = arith.addi %i, %c1 : index
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
// CHECK-LINE: define void @f5
// CHECK: (ptr %[[A:.*]], ptr %[[B:.*]], float %[[F:.*]])
func.func @f5(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.box<!fir.array<?xf32>>, %arg2: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  %1 = arith.subi %0#1, %c1 : index
  %2 = fir.array_load %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  %3 = fir.array_load %arg1 : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  // CHECK: icmp sgt
  %4 = fir.do_loop %arg3 = %c0 to %1 step %c1 iter_args(%arg4 = %2) -> (!fir.array<?xf32>) {
    // CHECK: %[[B_STRIDE_GEP:.*]] = getelementptr {{.*}}, ptr %[[B]], i32 0, i32 7, i64 0, i32 2
    // CHECK: %[[B_STRIDE:.*]] = load i64, ptr %[[B_STRIDE_GEP]]
    // CHECK: %[[B_DIM_OFFSET:.*]] = mul i64 %{{.*}}, %[[B_STRIDE]]
    // CHECK: %[[B_OFFSET:.*]] =  add i64 %[[B_DIM_OFFSET]], 0
    // CHECK: %[[B_BASE_GEP:.*]] = getelementptr {{.*}}, ptr %{{.*}}, i32 0, i32 0
    // CHECK: %[[B_BASE:.*]] = load ptr, ptr %[[B_BASE_GEP]]
    // CHECK: %[[B_VOID_ADDR:.*]] = getelementptr i8, ptr %[[B_BASE]], i64 %[[B_OFFSET]]
    // CHECK: %[[B_VAL:.*]] = load float, ptr %[[B_VOID_ADDR]]
    // CHECK: fadd float %[[B_VAL]], %[[F]]
    %5 = fir.array_fetch %3, %arg3 : (!fir.array<?xf32>, index) -> f32
    %6 = arith.addf %5, %arg2 : f32
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
// CHECK: (ptr %[[A:[^,]*]], float %[[F:.*]])
func.func @f6(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index

  // CHECK: %[[EXT_GEP:.*]] = getelementptr {{.*}} %[[A]], i32 0, i32 7, i64 0, i32 1
  // CHECK: %[[EXTENT:.*]] = load i64, ptr %[[EXT_GEP]]
  // CHECK: %[[SIZE:.*]] = mul i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), %[[EXTENT]]
  // CHECK: %[[MALLOC:.*]] = call ptr @malloc(i64 %[[SIZE]])
  %1 = fir.slice %c2, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.array_load %arg0 [%1] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.array<?xf32>
  %3 = fir.slice %c1, %c9, %c1 : (index, index, index) -> !fir.slice<1>
  %4 = fir.array_load %arg0 [%3] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.array<?xf32>
  %5 = fir.do_loop %arg2 = %c0 to %c9 step %c1 iter_args(%arg3 = %2) -> (!fir.array<?xf32>) {
    %6 = fir.array_fetch %4, %arg2 : (!fir.array<?xf32>, index) -> f32
    %7 = arith.addf %6, %arg1 : f32
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
// CHECK: ptr %[[X:[^,]*]], ptr %[[Y:.*]])
func.func @f7(%arg0: !fir.ref<f32>, %arg1: !fir.box<!fir.array<?xf32>>) {
  %c4 = arith.constant 4 : index
  %c100 = arith.constant 100 : index
  %0 = fir.shift %c4 : (index) -> !fir.shift<1>
  // CHECK: %[[STRIDE_GEP:.*]] = getelementptr {{.*}}, ptr %[[Y]], i32 0, i32 7, i64 0, i32 2
  // CHECK: %[[STRIDE:.*]] = load i64, ptr %[[STRIDE_GEP]]
  // CHECK: mul i64 96, %[[STRIDE]]
  %1 = fir.array_coor %arg1(%0) %c100 : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, index) -> !fir.ref<f32>
  %2 = fir.load %1 : !fir.ref<f32>
  fir.store %2 to %arg0 : !fir.ref<f32>
  return
} 

// Test A(:, :)%x reference codegen with A constant shape.
// CHECK-LABEL:  define void @f8(
// CHECK-SAME: ptr %[[A:.*]], i32 %[[I:.*]])
func.func @f8(%a : !fir.ref<!fir.array<2x2x!fir.type<t{i:i32}>>>, %i : i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %1 = fir.field_index i, !fir.type<t{i:i32}>
  %2 = fir.shape %c2, %c2 : (index, index) -> !fir.shape<2>
  %3 = fir.slice %c1, %c2, %c1, %c1, %c2, %c1 path %1 : (index, index, index, index, index, index, !fir.field) -> !fir.slice<2>
  // CHECK: %[[GEP:.*]] = getelementptr %t, ptr %[[A]], i64 0, i32 0
  %4 = fir.array_coor %a(%2) [%3] %c1, %c1 : (!fir.ref<!fir.array<2x2x!fir.type<t{i:i32}>>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
  // CHECK: store i32 %[[I]], ptr %[[GEP]], align 4
  fir.store %i to %4 : !fir.ref<i32>
  return
}

// Test casts in in array_coor offset computation when type parameters are not i64
// CHECK-LABEL: define ptr @f9(
// CHECK-SAME: i32 %[[I:.*]], i64 %{{.*}}, i64 %{{.*}}, ptr %[[C:.*]])
func.func @f9(%i: i32, %e : i64, %j: i64, %c: !fir.ref<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>> {
  %s = fir.shape %e, %e : (i64, i64) -> !fir.shape<2>
  // CHECK: %[[CAST:.*]] = sext i32 %[[I]] to i64
  // CHECK: %[[OFFSET:.*]] = mul i64 %{{.*}}, %[[CAST]]
  // CHECK: getelementptr i8, ptr %[[C]], i64 %[[OFFSET]]
  %a = fir.array_coor %c(%s) %j, %j typeparams %i : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, i64, i64, i32) -> !fir.ref<!fir.char<1,?>>
  return %a :  !fir.ref<!fir.char<1,?>>
}
