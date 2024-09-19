// RUN: tco %s | FileCheck %s
// RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s

// Test applying slice on fir.box
//   subroutine foo(x)
//     real :: x(3:, 4:)
//     call bar(x(5, 6:80:3))
//   end subroutine

func.func private @bar1(!fir.box<!fir.array<?xf32>>)
// CHECK-LABEL: define void @test_rebox_1(
// CHECK-SAME: ptr %[[INBOX:.*]])
func.func @test_rebox_1(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  // CHECK: %[[OUTBOX_ALLOC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c80 = arith.constant 80 : index
  %undef = fir.undefined index
  %0 = fir.slice %c5, %undef, %undef, %c6, %c80, %c3 : (index, index, index, index, index, index) -> !fir.slice<2>
  %1 = fir.shift %c3, %c4 : (index, index) -> !fir.shift<2>

  // CHECK: %[[EXTRA_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }, ptr %[[INBOX]], i32 0, i32 6
  // CHECK: %[[EXTRA:.*]] = load i8, ptr %[[EXTRA_GEP]]
  // CHECK: %[[EXTRA_WITH_ADDENDUM_CORRECTION:.*]] = and i8 %[[EXTRA]]
  // CHECK: %[[OUTBOX0:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { ptr undef, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i32 20240719, i8 1, i8 27, i8 0, i8 undef, [1 x [3 x i64]] undef }, i8 %[[EXTRA_WITH_ADDENDUM_CORRECTION]], 6
  // CHECK: %[[INSTRIDE_0_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }, ptr %[[INBOX]], i32 0, i32 7, i32 0, i32 2
  // CHECK: %[[INSTRIDE_0:.*]] = load i64, ptr %[[INSTRIDE_0_GEP]]
  // CHECK: %[[INSTRIDE_1_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }, ptr %[[INBOX]], i32 0, i32 7, i32 1, i32 2
  // CHECK: %[[INSTRIDE_1:.*]] = load i64, ptr %[[INSTRIDE_1_GEP]]
  // CHECK: %[[INBASE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }, ptr %[[INBOX]], i32 0, i32 0
  // CHECK: %[[INBASE:.*]] = load ptr, ptr %[[INBASE_GEP]]
  // CHECK: %[[OFFSET_0:.*]] = mul i64 2, %[[INSTRIDE_0]]
  // CHECK: %[[VOIDBASE0:.*]] = getelementptr i8, ptr %[[INBASE]], i64 %[[OFFSET_0]]
  // CHECK: %[[OFFSET_1:.*]] = mul i64 2, %[[INSTRIDE_1]]
  // CHECK: %[[VOIDBASE1:.*]] = getelementptr i8, ptr %[[VOIDBASE0]], i64 %[[OFFSET_1]]
  // CHECK: %[[OUTSTRIDE0:.*]] = mul i64 3, %[[INSTRIDE_1]]
  // CHECK: %[[OUTBOX1:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %{{.*}}, i64 %[[OUTSTRIDE0]], 7, 0, 2
  // CHECK: %[[OUTBOX2:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[OUTBOX1]], ptr %[[VOIDBASE1]], 0
  // CHECK: store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[OUTBOX2]], ptr %[[OUTBOX_ALLOC]], align 8
  %2 = fir.rebox %arg0(%1) [%0] : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>, !fir.slice<2>) -> !fir.box<!fir.array<?xf32>>
  // CHECK: call void @bar1(ptr %[[OUTBOX_ALLOC]])
  fir.call @bar1(%2) : (!fir.box<!fir.array<?xf32>>) -> ()
  return
}

// Test that character length is propagated in rebox
//   subroutine foo(x)
//     character(*) :: x(:, :)
//     call bar(x(4:30:1, 4:30:1))
//   end subroutine

func.func private @bar_rebox_test2(!fir.box<!fir.array<?x?x!fir.char<1,?>>>)
// CHECK-LABEL: define void @test_rebox_2(
// CHECK-SAME: ptr %[[INBOX:.*]])
func.func @test_rebox_2(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c30 = arith.constant 30 : index
  %0 = fir.slice %c4, %c30, %c1, %c4, %c30, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  // CHECK: %[[OUTBOX:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }
  // CHECK: %[[LEN_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }, ptr %[[INBOX]], i32 0, i32 1
  // CHECK: %[[LEN:.*]] = load i64, ptr %[[LEN_GEP]]
  // CHECK: %[[SIZE:.*]] = mul i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64), %[[LEN]]
  // CHECK: insertvalue { ptr, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] } undef, i64 %[[SIZE]], 1

  %1 = fir.rebox %arg0 [%0]  : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.slice<2>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  fir.call @bar_rebox_test2(%1) : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> ()
  return
}


// Test setting a new shape on a fir.box
//   subroutine foo(x)
//     real :: x(:)
//     real, pointer(:, :, :), p
//     p(2:5, 3:7, 4:9) => x
//     call bar(p)
//   end subroutine

func.func private @bar_rebox_test3(!fir.box<!fir.array<?x?x?xf32>>)
// CHECK-LABEL: define void @test_rebox_3(
// CHECK-SAME: ptr %[[INBOX:.*]])
func.func @test_rebox_3(%arg0: !fir.box<!fir.array<?xf32>>) {
  // CHECK: %[[OUTBOX_ALLOC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] }
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %1 = fir.shape_shift %c2, %c3, %c3, %c4, %c4, %c5 : (index, index, index, index, index, index) -> !fir.shapeshift<3>
  // CHECK: %[[INSTRIDE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[INBOX]], i32 0, i32 7, i32 0, i32 2
  // CHECK: %[[INSTRIDE:.*]] = load i64, ptr %[[INSTRIDE_GEP]]
  // CHECK: %[[INBASE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[INBOX]], i32 0, i32 0
  // CHECK: %[[INBASE:.*]] = load ptr, ptr %[[INBASE_GEP]]
  // CHECK: %[[OUTSTRIDE1:.*]] = mul i64 3, %[[INSTRIDE]]
  // CHECK: %[[OUTSTRIDE2:.*]] = mul i64 4, %[[OUTSTRIDE1]]
  // CHECK: %[[OUTBOX0:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %{{.*}}, i64 %[[INSTRIDE]], 7, 0, 2
  // CHECK: %[[OUTBOX1:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX0]], i64 3, 7, 1, 0
  // CHECK: %[[OUTBOX2:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX1]], i64 4, 7, 1, 1
  // CHECK: %[[OUTBOX3:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX2]], i64 %[[OUTSTRIDE1]], 7, 1, 2
  // CHECK: %[[OUTBOX4:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX3]], i64 4, 7, 2, 0
  // CHECK: %[[OUTBOX5:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX4]], i64 5, 7, 2, 1
  // CHECK: %[[OUTBOX6:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX5]], i64 %[[OUTSTRIDE2]], 7, 2, 2
  // CHECK: %[[OUTBOX7:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX6]], ptr %[[INBASE]], 0
  // CHECK: store { ptr, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX7]], ptr %[[OUTBOX_ALLOC]]
  %2 = fir.rebox %arg0(%1) : (!fir.box<!fir.array<?xf32>>, !fir.shapeshift<3>) -> !fir.box<!fir.array<?x?x?xf32>>
  // CHECK: call void @bar_rebox_test3(ptr %[[OUTBOX_ALLOC]])
  fir.call @bar_rebox_test3(%2) : (!fir.box<!fir.array<?x?x?xf32>>) -> ()
  return
}


// Test reboxing of character entities where the input has dynamic length and the output has compile
// time constant length.

// CHECK-LABEL: define void @test_rebox_4(
// CHECK-SAME: ptr %[[INPUT:.*]])
func.func @test_rebox_4(%arg0: !fir.box<!fir.array<?x!fir.char<1,?>>>) {
  // CHECK: %[[NEWBOX_STORAGE:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }
  // CHECK: %[[EXTENT_GEP:.*]] = getelementptr {{{.*}}}, ptr %[[INPUT]], i32 0, i32 7, i32 0, i32 1
  // CHECK: %[[EXTENT:.*]] = load i64, ptr %[[EXTENT_GEP]]
  // CHECK: %[[STRIDE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[INPUT]], i32 0, i32 7, i32 0, i32 2
  // CHECK: %[[STRIDE:.*]] = load i64, ptr %[[STRIDE_GEP]]
  // CHECK: %[[BASE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[INPUT]], i32 0, i32 0
  // CHECK: %[[BASE:.*]] = load ptr, ptr %[[BASE_GEP]]
  // CHECK: %[[NEWBOX1:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %{{.*}}, i64 %[[EXTENT]], 7, 0, 1
  // CHECK: %[[NEWBOX2:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[NEWBOX1]], i64 %[[STRIDE]], 7, 0, 2
  // CHECK: %[[NEWBOX3:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[NEWBOX2]], ptr %[[BASE]], 0
  // CHECK: store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[NEWBOX3]], ptr %[[NEWBOX_STORAGE]]
  // CHECK: call void @bar_test_rebox_4(ptr %[[NEWBOX_STORAGE]])

  %1 = fir.rebox %arg0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
  fir.call @bar_test_rebox_4(%1) : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>) -> ()
  return
}
func.func private @bar_test_rebox_4(!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>)

// Testing complex part slice reboxing
//   subroutine test_cmplx_2(a)
//     complex :: a(:)
//     call bar1(a%re)
//   end subroutine

// CHECK-LABEL: define void @test_cmplx_1(
// CHECK-SAME: ptr %[[INBOX:.*]])
func.func @test_cmplx_1(%arg0: !fir.box<!fir.array<?x!fir.complex<4>>>) {
  // CHECK: %[[OUTBOX_ALLOC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?x!fir.complex<4>>>, index) -> (index, index, index)
  %1 = fir.slice %c1, %0#1, %c1 path %c1_i32 : (index, index, index, i32) -> !fir.slice<1>
  %2 = fir.rebox %arg0 [%1] : (!fir.box<!fir.array<?x!fir.complex<4>>>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  // CHECK: %[[INSTRIDE_0_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[INBOX]], i32 0, i32 7, i64 0, i32 1
  // CHECK: %[[INSTRIDE_0:.*]] = load i64, ptr %[[INSTRIDE_0_GEP]]
  // CHECK: %[[INSTRIDE_1_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[INBOX]], i32 0, i32 7, i32 0, i32 2
  // CHECK: %[[INSTRIDE_1:.*]] = load i64, ptr %[[INSTRIDE_1_GEP]]
  // CHECK: %[[FRONT_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[INBOX]], i32 0, i32 0
  // CHECK: %[[FRONT_PTR:.*]] = load ptr, ptr %[[FRONT_GEP]]
  // CHECK: %[[FIELD_OFFSET_GEP:.*]] = getelementptr { float, float }, ptr %[[FRONT_PTR]], i64 0, i32 0
  // CHECK: %[[FRONT_OFFSET:.*]] = mul i64 0, %[[INSTRIDE_1]]
  // CHECK: %[[OFFSET_GEP:.*]] = getelementptr i8, ptr %[[FIELD_OFFSET_GEP]], i64 %[[FRONT_OFFSET]]
  // CHECK: %[[SUB_1:.*]] = sub i64 %[[INSTRIDE_0]], 1
  // CHECK: %[[ADD_1:.*]] = add i64 %[[SUB_1]], 1
  // CHECK: %[[DIV_1:.*]] = sdiv i64 %[[ADD_1]], 1
  // CHECK: %[[CHECK_NONZERO:.*]] = icmp sgt i64 %[[DIV_1]], 0
  // CHECK: %[[CHECKED_BOUND:.*]] = select i1 %[[CHECK_NONZERO]], i64 %[[DIV_1]], i64 0
  // CHECK: %[[STRIDE:.*]] = mul i64 1, %[[INSTRIDE_1]]
  // CHECK: %[[VAL_BUILD_1:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %{{.*}}, i64 %[[CHECKED_BOUND]], 7, 0, 1
  // CHECK: %[[VAL_BUILD_2:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[VAL_BUILD_1]], i64 %[[STRIDE]], 7, 0, 2
  // CHECK: %[[VAL_BUILD_3:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[VAL_BUILD_2]], ptr %[[OFFSET_GEP]], 0
  // CHECK: store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[VAL_BUILD_3]], ptr %[[OUTBOX_ALLOC]]
  fir.call @bar1(%2) : (!fir.box<!fir.array<?xf32>>) -> ()
  // CHECK: call void @bar1(ptr %[[OUTBOX_ALLOC]])
  return
}

// Testing triple on complex part slice
//   subroutine test_cmplx_2(a)
//     complex :: a(:)
//     call bar1(a(7:60:5)%im)
//   end subroutine

// CHECK-LABEL: define void @test_cmplx_2(
// CHECK-SAME: ptr %[[INBOX:.*]])
func.func @test_cmplx_2(%arg0: !fir.box<!fir.array<?x!fir.complex<4>>>) {
  // CHECK: %[[OUTBOX_ALLOC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }
  %c7 = arith.constant 7 : index
  %c5 = arith.constant 5 : index
  %c60 = arith.constant 60 : index
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.slice %c7, %c60, %c5 path %c1_i32 : (index, index, index, i32) -> !fir.slice<1>
  %1 = fir.rebox %arg0 [%0] : (!fir.box<!fir.array<?x!fir.complex<4>>>, !fir.slice<1>) -> !fir.box<!fir.array<11xf32>>
  %2 = fir.convert %1 : (!fir.box<!fir.array<11xf32>>) -> !fir.box<!fir.array<?xf32>>
  // CHECK: %[[INSTRIDE_0_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[INBOX]], i32 0, i32 7, i32 0, i32 2
  // CHECK: %[[INSTRIDE_0:.*]] = load i64, ptr %[[INSTRIDE_0_GEP]]
  // CHECK: %[[FRONT_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[INBOX]], i32 0, i32 0
  // CHECK: %[[FRONT_PTR:.*]] = load ptr, ptr %[[FRONT_GEP]]
  // CHECK: %[[FIELD_OFFSET_GEP:.*]] = getelementptr { float, float }, ptr %[[FRONT_PTR]], i64 0, i32 1
  // CHECK: %[[FRONT_OFFSET:.*]] = mul i64 6, %[[INSTRIDE_0]]
  // CHECK: %[[OFFSET_GEP:.*]] = getelementptr i8, ptr %[[FIELD_OFFSET_GEP]], i64 %[[FRONT_OFFSET]]
  // CHECK: %[[STRIDE:.*]] = mul i64 5, %[[INSTRIDE_0]]
  // CHECK: %[[VAL_BUILD_1:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %{{.*}}, i64 %[[STRIDE]], 7, 0, 2
  // CHECK: %[[VAL_BUILD_2:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[VAL_BUILD_1]], ptr %[[OFFSET_GEP]], 0
  // CHECK: store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[VAL_BUILD_2]], ptr %[[OUTBOX_ALLOC]]
  fir.call @bar1(%2) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
  // CHECK: call void @bar1(ptr %[[OUTBOX_ALLOC]])
  return
}

// Test reboxing of unlimited polymorphic.

func.func @rebox_unlimited_polymorphic_box(%arg0 : !fir.class<!fir.array<?xnone>>, %arg1 : !fir.box<!fir.array<?xnone>>, %arg0r : !fir.ref<!fir.class<!fir.array<?xnone>>>, %arg1r : !fir.ref<!fir.box<!fir.array<?xnone>>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %1 = fir.slice %c1, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.rebox %arg0 [%1] : (!fir.class<!fir.array<?xnone>>, !fir.slice<1>) -> !fir.class<!fir.array<?xnone>>
  %3 = fir.rebox %arg1 [%1] : (!fir.box<!fir.array<?xnone>>, !fir.slice<1>) -> !fir.box<!fir.array<?xnone>>
  fir.store %2 to %arg0r : !fir.ref<!fir.class<!fir.array<?xnone>>>
  fir.store %3 to %arg1r : !fir.ref<!fir.box<!fir.array<?xnone>>>
  return
}
// CHECK-LABEL: define void @rebox_unlimited_polymorphic_box
// CHECK:  %[[VAL_16:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }
// CHECK:  %[[VAL_17:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }
// ...
// CHECK:  store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %{{.*}}, ptr %[[VAL_17]]
// ...
// CHECK:  store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %{{.*}}, ptr %[[VAL_16]]
