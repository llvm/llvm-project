// RUN: tco -o - %s | FileCheck %s

// CHECK-LABEL: declare void @g({ float*, i64, i32, i8, i8, i8, i8 }*)
func @g(%b : !fir.box<f32>)
// CHECK-LABEL: declare void @ga({ float*, i64, i32, i8, i8, i8, i8 }*)
func @ga(%b : !fir.box<!fir.array<?xf32>>)

// CHECK-LABEL: define void @f
// CHECK: (float* %[[ARG:.*]])
func @f(%a : !fir.ref<f32>) {
  // CHECK: %[[GEP0:.*]] = getelementptr {{.*}}, i32 0, i32 0
  // CHECK: store float* %[[ARG]], float** %[[GEP0]]
  // CHECK: %[[GEP1:.*]] = getelementptr {{.*}}, i32 0, i32 1
  // CHECK: store i64 {{.*}}, i64* %[[GEP1]]
  // CHECK: %[[GEP2:.*]] = getelementptr {{.*}}, i32 0, i32 2
  // CHECK: store i32 {{.*}}, i32* %[[GEP2]]
  // CHECK: %[[GEP3:.*]] = getelementptr {{.*}}, i32 0, i32 3
  // CHECK: store i8 {{.*}}, i8* %[[GEP3]]
  // CHECK: %[[GEP4:.*]] = getelementptr {{.*}}, i32 0, i32 4
  // CHECK: store i8 {{.*}}, i8* %[[GEP4]]
  // CHECK: %[[GEP5:.*]] = getelementptr {{.*}}, i32 0, i32 5
  // CHECK: store i8 {{.*}}, i8* %[[GEP5]]
  // CHECK: %[[GEP6:.*]] = getelementptr {{.*}}, i32 0, i32 6
  // CHECK: store i8 {{.*}}, i8* %[[GEP6]]
  %b = fir.embox %a : (!fir.ref<f32>) -> !fir.box<f32>

  // CHECK: call void @g(
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
  // CHECK: %[[GEP70:.*]] = getelementptr {{.*}}, i32 0, i32 7, i32 0, i32 0
  // CHECK: store i64 {{.*}}, i64* %[[GEP70]]
  // CHECK: %[[GEP71:.*]] = getelementptr {{.*}}, i32 0, i32 7, i32 0, i32 1
  // CHECK: store i64 {{.*}}, i64* %[[GEP71]]
  // CHECK: %[[GEP72:.*]] = getelementptr {{.*}}, i32 0, i32 7, i32 0, i32 2
  // CHECK: store i64 {{.*}}, i64* %[[GEP72]]
  %b = fir.embox %c(%d) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  // CHECK: call void @ga(
  fir.call @ga(%b) : (!fir.box<!fir.array<?xf32>>) -> ()
  // CHECK: ret void
  return
}
