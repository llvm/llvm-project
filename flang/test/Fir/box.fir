// RUN: tco -o - %s | FileCheck %s

// Global box initialization (test must come first because llvm globals are emitted first).
// CHECK-LABEL: @globalx = internal global { i32*, i64, i32, i8, i8, i8, i8 } { i32* null, i64 4, i32 20180515, i8 0, i8 9, i8 2, i8 0 }
fir.global internal @globalx : !fir.box<!fir.heap<i32>> {
  %c0 = constant 0 : index
  %0 = fir.convert %c0 : (index) -> !fir.heap<i32>
  %1 = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.has_value %1 : !fir.box<!fir.heap<i32>>
}

// CHECK-LABEL: @globaly = internal global { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { float* null, i64 4, i32 20180515, i8 1, i8 25, i8 2, i8 0,{{.*}}[3 x i64] [i64 1, i64 0, i64 4]
fir.global internal @globaly : !fir.box<!fir.heap<!fir.array<?xf32>>> {
  %c0 = constant 0 : index
  %0 = fir.convert %c0 : (index) -> !fir.heap<!fir.array<?xf32>>
  %1 = fir.shape %c0 : (index) -> !fir.shape<1>
  %2 = fir.embox %0(%1) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xf32>>>
}

// CHECK-LABEL: declare void @g({ float*, i64, i32, i8, i8, i8, i8 }*)
func private @g(%b : !fir.box<f32>)
// CHECK-LABEL: declare void @ga({ float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }*)
func private @ga(%b : !fir.box<!fir.array<?xf32>>)

// CHECK-LABEL: define void @f
// CHECK: (float* %[[ARG:.*]])
func @f(%a : !fir.ref<f32>) {
  // CHECK: %[[DESC:.*]] = alloca { float*, i64, i32, i8, i8, i8, i8 }
  // CHECK: %[[INS0:.*]] = insertvalue {{.*}} undef, float* %[[ARG]], 0
  // CHECK: %[[INS1:.*]] = insertvalue {{.*}} %[[INS0]], i64 4, 1
  // CHECK: %[[INS2:.*]] = insertvalue {{.*}} %[[INS1]], i32 {{.*}}, 2
  // CHECK: %[[INS3:.*]] = insertvalue {{.*}} %[[INS2]], i8 0, 3
  // CHECK: %[[INS4:.*]] = insertvalue {{.*}} %[[INS3]], i8 25, 4
  // CHECK: %[[INS5:.*]] = insertvalue {{.*}} %[[INS4]], i8 0, 5
  // CHECK: %[[INS6:.*]] = insertvalue {{.*}} %[[INS5]], i8 0, 6
  // CHECK: store {{.*}} %[[INS6]], {{.*}}* %[[DESC]]
  %b = fir.embox %a : (!fir.ref<f32>) -> !fir.box<f32>

  // CHECK: call void @g({{.*}} %[[DESC]])
  fir.call @g(%b) : (!fir.box<f32>) -> ()
  // CHECK: ret void
  return
}

// CHECK-LABEL: define void @fa
// CHECK: ([100 x float]* %[[ARG:.*]])
func @fa(%a : !fir.ref<!fir.array<100xf32>>) {
  %c = fir.convert %a : (!fir.ref<!fir.array<100xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %c1 = constant 1 : index
  %c100 = constant 100 : index
  %d = fir.shape %c100 : (index) -> !fir.shape<1>
  // CHECK: %[[INS70:.*]] = insertvalue {{.*}} %{{.*}}, i64 0, 7, 0, 0
  // CHECK: %[[INS71:.*]] = insertvalue {{.*}} %[[INS70]], i64 100, 7, 0, 1
  // CHECK: %[[INS72:.*]] = insertvalue {{.*}} %[[INS71]], i64 4, 7, 0, 2
  %b = fir.embox %c(%d) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  // CHECK: call void @ga(
  fir.call @ga(%b) : (!fir.box<!fir.array<?xf32>>) -> ()
  // CHECK: ret void
  return
}

// Boxing of a scalar character of dynamic length
// CHECK-LABEL: define { i8*, i64, i32, i8, i8, i8, i8 }* @b1(
// CHECK-SAME: i8* %[[arg0:.*]], i64 %[[arg1:.*]])
func @b1(%arg0 : !fir.ref<!fir.char<1,?>>, %arg1 : index) -> !fir.box<!fir.char<1,?>> {
  // CHECK: insertvalue {{.*}} i8* %[[arg0]], 0
  // CHECK: insertvalue {{.*}} i64 %[[arg1]], 1
  // CHECK: insertvalue {{.*}} i32 20180515, 2
  %x = fir.embox %arg0 typeparams %arg1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  return %x : !fir.box<!fir.char<1,?>>
}

// Boxing of a dynamic array of character with static length (5)
// CHECK-LABEL: define { [5 x i8]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* @b2(
// CHECK-SAME: [5 x i8]* %[[arg0:.*]], i64 %[[arg1:.*]])
func @b2(%arg0 : !fir.ref<!fir.array<?x!fir.char<1,5>>>, %arg1 : index) -> !fir.box<!fir.array<?x!fir.char<1,5>>> {
  %1 = fir.shape %arg1 : (index) -> !fir.shape<1>
  // CHECK: insertvalue {{.*}} [5 x i8]* %[[arg0]], 0
  // CHECK: insertvalue {{.*}} i64 5, 1
  // CHECK: insertvalue {{.*}} i32 20180515, 2
  // CHECK: insertvalue {{.*}} i64 %[[arg1]], 7, 0, 1
  // CHECK: insertvalue {{.*}} i64 5, 7, 0, 2
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<?x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.char<1,5>>>
  return %2 : !fir.box<!fir.array<?x!fir.char<1,5>>>
}

// Boxing of a dynamic array of character of dynamic length
// CHECK-LABEL: define { i8*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* @b3(
// CHECK-SAME: i8* %[[arg0:.*]], i64 %[[arg1:.*]], i64 %[[arg2:.*]])
func @b3(%arg0 : !fir.ref<!fir.array<?x!fir.char<1,?>>>, %arg1 : index, %arg2 : index) -> !fir.box<!fir.array<?x!fir.char<1,?>>> {
  %1 = fir.shape %arg2 : (index) -> !fir.shape<1>
  // CHECK: insertvalue {{.*}} i8* %[[arg0]], 0
  // CHECK: insertvalue {{.*}} i64 %[[arg1]], 1
  // CHECK: insertvalue {{.*}} i32 20180515, 2
  // CHECK: insertvalue {{.*}} i64 %[[arg2]], 7, 0, 1
  // CHECK: insertvalue {{.*}} i64 %[[arg1]], 7, 0, 2
  %2 = fir.embox %arg0(%1) typeparams %arg1 : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  return %2 : !fir.box<!fir.array<?x!fir.char<1,?>>>
}

// Boxing of a static array of character of dynamic length
// CHECK-LABEL: define { i8*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* @b4(
// CHECK-SAME: i8* %[[arg0:.*]], i64 %[[arg1:.*]])
func @b4(%arg0 : !fir.ref<!fir.array<7x!fir.char<1,?>>>, %arg1 : index) -> !fir.box<!fir.array<7x!fir.char<1,?>>> {
  %c_7 = constant 7 : index
  %1 = fir.shape %c_7 : (index) -> !fir.shape<1>
  // CHECK: insertvalue {{.*}} i8* %[[arg0]], 0
  // CHECK: insertvalue {{.*}} i64 %[[arg1]], 1
  // CHECK: insertvalue {{.*}} i32 20180515, 2
  // CHECK: insertvalue {{.*}} i64 7, 7, 0, 1
  // CHECK: insertvalue {{.*}} i64 %[[arg1]], 7, 0, 2
  %x = fir.embox %arg0(%1) typeparams %arg1 : (!fir.ref<!fir.array<7x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<7x!fir.char<1,?>>>
  return %x : !fir.box<!fir.array<7x!fir.char<1,?>>>
}

// Storing a fir.box into a fir.ref<fir.box> (modifying descriptors).
// CHECK-LABEL: define void @b5(
// CHECK-SAME: { float*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }* %[[arg0:.*]], { float*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }* %[[arg1:.*]])
func @b5(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, %arg1 : !fir.box<!fir.heap<!fir.array<?x?xf32>>>) {
  fir.store %arg1 to %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  // CHECK: %[[boxLoad:.*]] = load { float*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }, { float*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }* %[[arg1]]
  // CHECK: store { float*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] } %[[boxLoad]], { float*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }* %[[arg0]]
  return
}
