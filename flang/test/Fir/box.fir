// RUN: tco -o - %s | FileCheck %s

// Global box initialization (test must come first because llvm globals are emitted first).
// CHECK-LABEL: @globalx = internal global { ptr, i64, i32, i8, i8, i8, i8 } { ptr null, i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), i32 20240719, i8 0, i8 9, i8 2, i8 0 }
fir.global internal @globalx : !fir.box<!fir.heap<i32>> {
  %c0 = arith.constant 0 : index
  %0 = fir.convert %c0 : (index) -> !fir.heap<i32>
  %1 = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.has_value %1 : !fir.box<!fir.heap<i32>>
}

// CHECK-LABEL: @globaly = internal global { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { ptr null, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i32 20240719, i8 1, i8 27, i8 2, i8 0,{{.*}}[3 x i64] [i64 1, i64 0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)]
fir.global internal @globaly : !fir.box<!fir.heap<!fir.array<?xf32>>> {
  %c0 = arith.constant 0 : index
  %0 = fir.convert %c0 : (index) -> !fir.heap<!fir.array<?xf32>>
  %1 = fir.shape %c0 : (index) -> !fir.shape<1>
  %2 = fir.embox %0(%1) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xf32>>>
}

// CHECK-LABEL: declare void @g(ptr)
func.func private @g(%b : !fir.box<f32>)
// CHECK-LABEL: declare void @ga(ptr)
func.func private @ga(%b : !fir.box<!fir.array<?xf32>>)

// CHECK-LABEL: define void @f
// CHECK: (ptr captures(none) %[[ARG:.*]])
func.func @f(%a : !fir.ref<f32>) {
  // CHECK: %[[DESC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }
  // CHECK: %[[INS0:.*]] = insertvalue {{.*}} { ptr undef, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i32 20240719, i8 0, i8 27, i8 0, i8 0 }, ptr %[[ARG]], 0
  // CHECK: store {{.*}} %[[INS0]], {{.*}} %[[DESC]]
  %b = fir.embox %a : (!fir.ref<f32>) -> !fir.box<f32>

  // CHECK: call void @g({{.*}} %[[DESC]])
  fir.call @g(%b) : (!fir.box<f32>) -> ()
  // CHECK: ret void
  return
}

// CHECK-LABEL: define void @fa
// CHECK: (ptr captures(none) %[[ARG:.*]])
func.func @fa(%a : !fir.ref<!fir.array<100xf32>>) {
  %c = fir.convert %a : (!fir.ref<!fir.array<100xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %d = fir.shape %c100 : (index) -> !fir.shape<1>
  // CHECK: %[[INS70:.*]] = insertvalue {{.*}} { ptr undef, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i32 20240719, i8 1, i8 27, i8 0, i8 0, {{.*}} }, ptr %{{.*}}, 0
  %b = fir.embox %c(%d) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  // CHECK: call void @ga(
  fir.call @ga(%b) : (!fir.box<!fir.array<?xf32>>) -> ()
  // CHECK: ret void
  return
}

// Boxing of a scalar character of dynamic length
// CHECK-LABEL: define void @b1(
// CHECK-SAME: ptr captures(none) %[[res:.*]], ptr captures(none) %[[arg0:.*]], i64 %[[arg1:.*]])
func.func @b1(%arg0 : !fir.ref<!fir.char<1,?>>, %arg1 : index) -> !fir.box<!fir.char<1,?>> {
  // CHECK: %[[alloca:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }
  // CHECK: %[[size:.*]] = mul i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64), %[[arg1]]
  // CHECK: insertvalue {{.*}} undef, i64 %[[size]], 1
  // CHECK: insertvalue {{.*}} i32 20240719, 2
  // CHECK: insertvalue {{.*}} ptr %[[arg0]], 0
  %x = fir.embox %arg0 typeparams %arg1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  // CHECK: store {{.*}}, ptr %[[alloca]]
  // CHECK: call void @llvm.memcpy.p0.p0.i32(ptr %[[res]], ptr %[[alloca]], i32 24, i1 false)
  return %x : !fir.box<!fir.char<1,?>>
}

// Boxing of a dynamic array of character with static length (5)
// CHECK-LABEL: define void @b2(
// CHECK-SAME: ptr captures(none) %[[res]],
// CHECK-SAME: ptr captures(none) %[[arg0:.*]], i64 %[[arg1:.*]])
func.func @b2(%arg0 : !fir.ref<!fir.array<?x!fir.char<1,5>>>, %arg1 : index) -> !fir.box<!fir.array<?x!fir.char<1,5>>> {
  %1 = fir.shape %arg1 : (index) -> !fir.shape<1>
  // CHECK: %[[alloca:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }
  // CHECK: insertvalue {{.*}} { ptr undef, i64 ptrtoint (ptr getelementptr ([5 x i8], ptr null, i32 1) to i64), i32 20240719, i8 1, i8 40, i8 0, i8 0, {{.*}} }, i64 %[[arg1]], 7, 0, 1
  // CHECK: insertvalue {{.*}} %{{.*}}, i64 ptrtoint (ptr getelementptr ([5 x i8], ptr null, i32 1) to i64), 7, 0, 2
  // CHECK: insertvalue {{.*}} ptr %[[arg0]], 0
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<?x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.char<1,5>>>
  // CHECK: store {{.*}}, ptr %[[alloca]]
  // CHECK: call void @llvm.memcpy.p0.p0.i32(ptr %[[res]], ptr %[[alloca]], i32 48, i1 false)
  return %2 : !fir.box<!fir.array<?x!fir.char<1,5>>>
}

// Boxing of a dynamic array of character of dynamic length
// CHECK-LABEL: define void @b3(
// CHECK-SAME: ptr captures(none) %[[res:.*]], ptr captures(none) %[[arg0:.*]], i64 %[[arg1:.*]], i64 %[[arg2:.*]])
func.func @b3(%arg0 : !fir.ref<!fir.array<?x!fir.char<1,?>>>, %arg1 : index, %arg2 : index) -> !fir.box<!fir.array<?x!fir.char<1,?>>> {
  %1 = fir.shape %arg2 : (index) -> !fir.shape<1>
  // CHECK: %[[alloca:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }
  // CHECK: %[[size:.*]] = mul i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64), %[[arg1]]
  // CHECK: insertvalue {{.*}} i64 %[[size]], 1
  // CHECK: insertvalue {{.*}} i32 20240719, 2
  // CHECK: insertvalue {{.*}} i64 %[[arg2]], 7, 0, 1
  // CHECK: insertvalue {{.*}} i64 %[[size]], 7, 0, 2
  // CHECK: insertvalue {{.*}} ptr %[[arg0]], 0
  %2 = fir.embox %arg0(%1) typeparams %arg1 : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  // CHECK: store {{.*}}, ptr %[[alloca]]
  // CHECK: call void @llvm.memcpy.p0.p0.i32(ptr %[[res]], ptr %[[alloca]], i32 48, i1 false)
  return %2 : !fir.box<!fir.array<?x!fir.char<1,?>>>
}

// Boxing of a static array of character of dynamic length
// CHECK-LABEL: define void @b4(
// CHECK-SAME: ptr captures(none) %[[res:.*]], ptr captures(none) %[[arg0:.*]], i64 %[[arg1:.*]])
func.func @b4(%arg0 : !fir.ref<!fir.array<7x!fir.char<1,?>>>, %arg1 : index) -> !fir.box<!fir.array<7x!fir.char<1,?>>> {
  %c_7 = arith.constant 7 : index
  %1 = fir.shape %c_7 : (index) -> !fir.shape<1>
  // CHECK: %[[alloca:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }
  // CHECK:   %[[size:.*]] = mul i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64), %[[arg1]]
  // CHECK: insertvalue {{.*}} i64 %[[size]], 1
  // CHECK: insertvalue {{.*}} i32 20240719, 2
  // CHECK: insertvalue {{.*}} i64 7, 7, 0, 1
  // CHECK: insertvalue {{.*}} i64 %[[size]], 7, 0, 2
  // CHECK: insertvalue {{.*}} ptr %[[arg0]], 0
  %x = fir.embox %arg0(%1) typeparams %arg1 : (!fir.ref<!fir.array<7x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<7x!fir.char<1,?>>>
  // CHECK: store {{.*}}, ptr %[[alloca]]
  // CHECK: call void @llvm.memcpy.p0.p0.i32(ptr %[[res]], ptr %[[alloca]], i32 48, i1 false)
  return %x : !fir.box<!fir.array<7x!fir.char<1,?>>>
}

// Storing a fir.box into a fir.ref<fir.box> (modifying descriptors).
// CHECK-LABEL: define void @b5(
// CHECK-SAME: ptr captures(none) %[[arg0:.*]], ptr %[[arg1:.*]])
func.func @b5(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, %arg1 : !fir.box<!fir.heap<!fir.array<?x?xf32>>>) {
  fir.store %arg1 to %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  // CHECK: call void @llvm.memcpy.p0.p0.i32(ptr %0, ptr %1, i32 72, i1 false)
  return
}

func.func private @callee6(!fir.box<none>) -> i32

// CHECK-LABEL: define i32 @box6(
// CHECK-SAME: ptr captures(none) %[[ARG0:.*]], i64 %[[ARG1:.*]], i64 %[[ARG2:.*]])
func.func @box6(%0 : !fir.ref<!fir.array<?x?x?x?xf32>>, %1 : index, %2 : index) -> i32 {
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  %c30 = arith.constant 30 : index
  %c6 = arith.constant 6 : index
  %shape = fir.shape %c100, %c50, %c30, %c6 : (index, index, index, index) -> !fir.shape<4>
  %3 = fir.undefined index
  %c41 = arith.constant 41 : index
  %c2 = arith.constant 2 : index
  %c24 = arith.constant 24 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index

  // CHECK: %[[i:.*]] = sub i64 %[[ARG1]], 1
  // CHECK: %[[i100:.*]] = mul i64 %[[i]], 100
  // CHECK: %[[i100p40:.*]] = add i64 %[[i100]], 40
  // CHECK: %[[diff:.*]] = sub i64 %[[ARG2]], %[[ARG1]]
  // CHECK: %[[dp2:.*]] = add i64 %[[diff]], 2
  // CHECK: %[[sdp2:.*]] = sdiv i64 %[[dp2]], 2
  // CHECK: %[[cmp:.*]] = icmp sgt i64 %[[sdp2]], 0
  // CHECK: %[[extent:.*]] = select i1 %[[cmp]], i64 %[[sdp2]], i64 0
  // CHECK: insertvalue { ptr, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] } { ptr undef, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i32 20240719, i8 2, i8 27, i8 0, i8 0, [2 x [3 x i64]] [{{\[}}3 x i64] [i64 1, i64 undef, i64 undef], [3 x i64] undef] }, i64 %[[extent]], 7, 0, 1
  // CHECK: insertvalue { ptr, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] } %{{.*}}, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 200), 7, 0, 2
  // CHECK: %[[op25:.*]] = add i64 25000, %[[i100p40]]
  // CHECK: insertvalue { ptr, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] } %{{.*}}, i64 1, 7, 1, 0
  // CHECK: insertvalue { ptr, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] } %{{.*}}, i64 4, 7, 1, 1
  // CHECK: insertvalue { ptr, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] } %{{.*}}, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 30000), 7, 1, 2
  // CHECK: %[[op300:.*]] = add i64 300000, %[[op25]]
  // CHECK: %[[ptr:.*]] = getelementptr float, ptr %[[ARG0]], i64 %[[op300]]
  // CHECK: insertvalue { ptr, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] } %{{.*}}, ptr %[[ptr]], 0
  // CHECK: store { ptr, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] } %{{.*}}, ptr %[[mem:[0-9]+]]


  %slice = fir.slice %c41, %3, %3, %1, %2, %c2, %c6, %c24, %c6, %c3, %3, %3 : (index, index, index, index, index, index, index, index, index, index, index, index) -> !fir.slice<4>
  %box = fir.embox %0(%shape)[%slice] : (!fir.ref<!fir.array<?x?x?x?xf32>>, !fir.shape<4>, !fir.slice<4>) -> !fir.box<!fir.array<?x?x?x?xf32>>
  %nonebox = fir.convert %box : (!fir.box<!fir.array<?x?x?x?xf32>>) -> !fir.box<none>
  // CHECK: %[[call:.*]] = call i32 @callee6(ptr %[[mem]])
  %rv = fir.call @callee6(%nonebox) : (!fir.box<none>) -> i32
  // CHECK: ret i32 %[[call]]
  return %rv : i32
}
