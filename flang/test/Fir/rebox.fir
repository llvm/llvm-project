// RUN: tco %s | FileCheck %s

// Test applying slice on fir.box
//   subroutine foo(x)
//     real :: x(3:, 4:)
//     call bar(x(5, 6:80:3))
//   end subroutine

func private @bar1(!fir.box<!fir.array<?xf32>>)
// CHECK-LABEL: define void @test_rebox_1(
// CHECK-SAME: { float*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }* %[[INBOX:.*]])
func @test_rebox_1(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  // CHECK: %[[OUTBOX_ALLOC:.*]] = alloca { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c80 = arith.constant 80 : index
  %undef = fir.undefined index
  %0 = fir.slice %c5, %undef, %undef, %c6, %c80, %c3 : (index, index, index, index, index, index) -> !fir.slice<2>
  %1 = fir.shift %c3, %c4 : (index, index) -> !fir.shift<2>

  // CHECK: %[[INSTRIDE_0_GEP:.*]] = getelementptr { float*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }, { float*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }* %[[INBOX]], i32 0, i32 7, i64 0, i32 2
  // CHECK: %[[INSTRIDE_0:.]] = load i64, i64* %[[INSTRIDE_0_GEP]]
  // CHECK: %[[INSTRIDE_1_GEP:.*]] = getelementptr { float*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }, { float*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }* %[[INBOX]], i32 0, i32 7, i64 1, i32 2
  // CHECK: %[[INSTRIDE_1:.*]] = load i64, i64* %[[INSTRIDE_1_GEP]]
  // CHECK: %[[INBASE_GEP:.*]] = getelementptr { float*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }, { float*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }* %[[INBOX]], i32 0, i32 0
  // CHECK: %[[INBASE:.*]] = load float*, float** %[[INBASE_GEP]]
  // CHECK: %[[VOIDBASE:.*]] = bitcast float* %[[INBASE]] to i8*
  // CHECK: %[[OFFSET_0:.*]] = mul i64 2, %[[INSTRIDE_0]]
  // CHECK: %[[VOIDBASE0:.*]] = getelementptr i8, i8* %[[VOIDBASE]], i64 %[[OFFSET_0]]
  // CHECK: %[[OFFSET_1:.*]] = mul i64 2, %[[INSTRIDE_1]]
  // CHECK: %[[VOIDBASE1:.*]] = getelementptr i8, i8* %[[VOIDBASE0]], i64 %[[OFFSET_1]]
  // CHECK: %[[OUTSTRIDE0:.*]] = mul i64 3, %[[INSTRIDE_1]]
  // CHECK: %[[OUTBOX0:.*]] = insertvalue { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { float* undef, i64 4, i32 {{.*}}, i8 1, i8 27, i8 0, i8 0, [1 x [3 x i64]] [{{.*}} [i64 1, i64 25, i64 undef]] }, i64 %[[OUTSTRIDE0]], 7, 0, 2
  // CHECK: %[[OUTBASE:.*]] = bitcast i8* %[[VOIDBASE1]] to float*
  // CHECK: %[[OUTBOX1:.*]] = insertvalue { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[OUTBOX0]], float* %[[OUTBASE]], 0
  // CHECK: store { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[OUTBOX1]], { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[OUTBOX_ALLOC]], align 8
  %2 = fir.rebox %arg0(%1) [%0] : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>, !fir.slice<2>) -> !fir.box<!fir.array<?xf32>>
  // CHECK: call void @bar1({ float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[OUTBOX_ALLOC]])
  fir.call @bar1(%2) : (!fir.box<!fir.array<?xf32>>) -> ()
  return
}

// Test that character length is propagated in rebox
//   subroutine foo(x)
//     character(*) :: x(:, :)
//     call bar(x(4:30:1, 4:30:1))
//   end subroutine

func private @bar_rebox_test2(!fir.box<!fir.array<?x?x!fir.char<1,?>>>)
// CHECK-LABEL: define void @test_rebox_2(
// CHECK-SAME: { i8*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }* %[[INBOX:.*]])
func @test_rebox_2(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c30 = arith.constant 30 : index
  %0 = fir.slice %c4, %c30, %c1, %c4, %c30, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  // CHECK: %[[OUTBOX:.*]] = alloca { i8*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }
  // CHECK: %[[LEN_GEP:.*]] = getelementptr { i8*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }, { i8*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] }* %[[INBOX]], i32 0, i32 1
  // CHECK: %[[LEN:.*]] = load i64, i64* %[[LEN_GEP]]
  // CHECK: insertvalue { i8*, i64, i32, i8, i8, i8, i8, [2 x [3 x i64]] } undef, i64 %[[LEN]], 1

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

func private @bar_rebox_test3(!fir.box<!fir.array<?x?x?xf32>>)
// CHECK-LABEL: define void @test_rebox_3(
// CHECK-SAME: { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[INBOX:.*]])
func @test_rebox_3(%arg0: !fir.box<!fir.array<?xf32>>) {
  // CHECK: %[[OUTBOX_ALLOC:.*]] = alloca { float*, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] }
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %1 = fir.shape_shift %c2, %c3, %c3, %c4, %c4, %c5 : (index, index, index, index, index, index) -> !fir.shapeshift<3>
  // CHECK: %[[INSTRIDE_GEP:.*]] = getelementptr { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[INBOX]], i32 0, i32 7, i64 0, i32 2
  // CHECK: %[[INSTRIDE:.*]] = load i64, i64* %[[INSTRIDE_GEP]]
  // CHECK: %[[INBASE_GEP:.*]] = getelementptr { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[INBOX]], i32 0, i32 0
  // CHECK: %[[INBASE:.*]] = load float*, float** %[[INBASE_GEP]]
  // CHECK: %[[VOIDBASE:.*]] = bitcast float* %[[INBASE]] to i8*
  // CHECK: %[[OUTSTRIDE1:.*]] = mul i64 3, %[[INSTRIDE]]
  // CHECK: %[[OUTSTRIDE2:.*]] = mul i64 4, %[[OUTSTRIDE1]]
  // CHECK: %[[OUTBOX0:.*]] = insertvalue { float*, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } { float* undef, i64 4, i32 {{.*}}, i8 3, i8 27, i8 0, i8 0, [3 x [3 x i64]] [{{.*}} [i64 2, i64 3, i64 undef], [3 x i64] undef, [3 x i64] undef] }, i64 %[[INSTRIDE]], 7, 0, 2
  // CHECK: %[[OUTBOX1:.*]] = insertvalue { float*, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX0]], i64 3, 7, 1, 0
  // CHECK: %[[OUTBOX2:.*]] = insertvalue { float*, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX1]], i64 4, 7, 1, 1
  // CHECK: %[[OUTBOX3:.*]] = insertvalue { float*, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX2]], i64 %[[OUTSTRIDE1]], 7, 1, 2
  // CHECK: %[[OUTBOX4:.*]] = insertvalue { float*, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX3]], i64 4, 7, 2, 0
  // CHECK: %[[OUTBOX5:.*]] = insertvalue { float*, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX4]], i64 5, 7, 2, 1
  // CHECK: %[[OUTBOX6:.*]] = insertvalue { float*, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX5]], i64 %[[OUTSTRIDE2]], 7, 2, 2
  // CHECK: %[[OUTBASE:.*]] = bitcast i8* %[[VOIDBASE]] to float*
  // CHECK: %[[OUTBOX7:.*]] = insertvalue { float*, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX6]], float* %[[OUTBASE]], 0
  // CHECK: store { float*, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] } %[[OUTBOX7]], { float*, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] }* %[[OUTBOX_ALLOC]]
  %2 = fir.rebox %arg0(%1) : (!fir.box<!fir.array<?xf32>>, !fir.shapeshift<3>) -> !fir.box<!fir.array<?x?x?xf32>>
  // CHECK: call void @bar_rebox_test3({ float*, i64, i32, i8, i8, i8, i8, [3 x [3 x i64]] }* %[[OUTBOX_ALLOC]])
  fir.call @bar_rebox_test3(%2) : (!fir.box<!fir.array<?x?x?xf32>>) -> ()
  return
}


// Test reboxing of character entities where the input has dynamic length and the output has compile
// time constant length.

// CHECK-LABEL: define void @test_rebox_4(
// CHECK-SAME: { i8*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[INPUT:.*]])
func @test_rebox_4(%arg0: !fir.box<!fir.array<?x!fir.char<1,?>>>) {
  // CHECK: %[[NEWBOX_STORAGE:.*]] = alloca { [10 x i8]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }
  // CHECK: %[[EXTENT_GEP:.*]] = getelementptr {{{.*}}}, { i8*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[INPUT]], i32 0, i32 7, i64 0, i32 1
  // CHECK: %[[EXTENT:.*]] = load i64, i64* %[[EXTENT_GEP]]
  // CHECK: %[[STRIDE_GEP:.*]] = getelementptr { i8*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { i8*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[INPUT]], i32 0, i32 7, i64 0, i32 2
  // CHECK: %[[STRIDE:.*]] = load i64, i64* %[[STRIDE_GEP]]
  // CHECK: %[[BASE_GEP:.*]] = getelementptr { i8*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, { i8*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[INPUT]], i32 0, i32 0
  // CHECK: %[[BASE:.*]] = load i8*, i8** %[[BASE_GEP]]
  // CHECK: %[[NEWBOX1:.*]] = insertvalue {{{.*}}} { [10 x i8]* undef, i64 10, i32 20180515, i8 1, i8 40, i8 1, i8 0, [1 x [3 x i64]] [{{.*}} [i64 1, i64 undef, i64 undef]] }, i64 %[[EXTENT]], 7, 0, 1
  // CHECK: %[[NEWBOX2:.*]] = insertvalue { [10 x i8]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[NEWBOX1]], i64 %[[STRIDE]], 7, 0, 2
  // CHECK: %[[BASE_CAST:.*]] = bitcast i8* %12 to [10 x i8]*
  // CHECK: %[[NEWBOX3:.*]] = insertvalue { [10 x i8]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[NEWBOX2]], [10 x i8]* %[[BASE_CAST]], 0
  // CHECK: store { [10 x i8]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } %[[NEWBOX3]], { [10 x i8]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[NEWBOX_STORAGE]]
  // CHECK: call void @bar_test_rebox_4({ [10 x i8]*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %[[NEWBOX_STORAGE]])

  %1 = fir.rebox %arg0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
  fir.call @bar_test_rebox_4(%1) : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>) -> ()
  return
}
func private @bar_test_rebox_4(!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>)
