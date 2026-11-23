// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s -check-prefixes=CHECK

// CHECK: [[BufA:@.*]] = private unnamed_addr constant [2 x i8] c"A\00", align 1
// CHECK: [[BufB:@.*]] = private unnamed_addr constant [2 x i8] c"B\00", align 1

// one-dimensional array
RWBuffer<float> A[2] : register(u10, space1);

// multi-dimensional array
[[vk::binding(13)]] 
RWBuffer<float> B[2][2] : register(u13, space0);

void useArray(RWBuffer<float> LocalArg[2]) {
}

void useMultiArray(RWBuffer<float> LocalArg[2][2]) {
}

// CHECK-LABEL: case1
// CHECK: %LocalOne = alloca [2 x %"class.hlsl::RWBuffer"], align 4
// CHECK: [[ArrayTmp:%.*]] = alloca [2 x %"class.hlsl::RWBuffer"], align 4

// - Initialize all resources of the global array into a local temporary.
// CHECK: [[ArrayPtr0:%.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr [[ArrayTmp]], i32 0, i32 0
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr0]], i32 noundef 10, i32 noundef 1, i32 noundef 2, i32 noundef 0, ptr noundef [[BufA]])
// CHECK-NEXT: [[ArrayPtr1:%.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr [[ArrayTmp]], i32 0, i32 1
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr1]], i32 noundef 10, i32 noundef 1, i32 noundef 2, i32 noundef 1, ptr noundef [[BufA]])

// - Copy from temporary to local array (in a loop)
// CHECK: %arrayinit.begin = getelementptr inbounds [2 x %"class.hlsl::RWBuffer"], ptr %LocalOne, i32 0, i32 0
// CHECK: arrayinit.body:
// CHECK: call void @hlsl::RWBuffer<float>::RWBuffer(hlsl::RWBuffer<float> const&)(ptr {{.*}}, ptr {{.*}})
// CHECK: arrayinit.end:
// CHECK: ret void
void case1() {
  // local one-dimensional array initialized with global array
  RWBuffer<float> LocalOne[2] = A;
}

// CHECK-LABEL: case2
// CHECK: %LocalTwo = alloca [2 x %"class.hlsl::RWBuffer"], align 4

// - Local array is first initialized to poison using the default constructor (in a loop)
//
// CHECK: %array.begin = getelementptr inbounds [2 x %"class.hlsl::RWBuffer"], ptr %LocalTwo, i32 0, i32 0
// CHECK: arrayctor.loop
// CHECK: call void @hlsl::RWBuffer<float>::RWBuffer()(ptr {{.*}})
// CHECK: arrayctor.cont:

// - Initialize individual resource elements directly into the local array
// CHECK: [[ArrayPtr0:%.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr %LocalTwo, i32 0, i32 0
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr0]], i32 noundef 10, i32 noundef 1, i32 noundef 2, i32 noundef 0, ptr noundef [[BufA]])
// CHECK-NEXT: [[ArrayPtr1:%.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr %LocalTwo, i32 0, i32 1
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr1]], i32 noundef 10, i32 noundef 1, i32 noundef 2, i32 noundef 1, ptr noundef [[BufA]])
// CHECK: ret void
void case2() {
  // local one-dimensional array initialized with assignment
  RWBuffer<float> LocalTwo[2];
  LocalTwo = A;
}

// CHECK-LABEL: case3
// CHECK: [[AggTmp:%.*]] = alloca [2 x %"class.hlsl::RWBuffer"], align 4
// CHECK-NEXT: [[ArrayPtr0:%.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr [[AggTmp]], i32 0, i32 0
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr0]], i32 noundef 10, i32 noundef 1, i32 noundef 2, i32 noundef 0, ptr noundef [[BufA]])
// CHECK-NEXT: [[ArrayPtr1:%.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr [[AggTmp]], i32 0, i32 1
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr1]], i32 noundef 10, i32 noundef 1, i32 noundef 2, i32 noundef 1, ptr noundef [[BufA]])
// CHECK-NEXT: call void @useArray(hlsl::RWBuffer<float> [2])(ptr {{.*}} [[AggTmp]])
void case3() {
  // resource array as function argument
  useArray(A);
}

// CHECK-LABEL: case4
// CHECK: %LocalThree = alloca [2 x [2 x %"class.hlsl::RWBuffer"]], align 4
// CHECK: [[TmpResArray:%.*]] = alloca [2 x [2 x %"class.hlsl::RWBuffer"]], align 4

// - Initialize all resources of the global array into a local temporary.
// CHECK: [[ArrayPtr00:%.*]] = getelementptr [2 x [2 x %"class.hlsl::RWBuffer"]], ptr [[TmpResArray]], i32 0, i32 0, i32 0
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr00]], i32 noundef 13, i32 noundef 0, i32 noundef 4, i32 noundef 0, ptr noundef [[BufB]])
// CHECK-NEXT: [[ArrayPtr01:%.*]] = getelementptr [2 x [2 x %"class.hlsl::RWBuffer"]], ptr [[TmpResArray]], i32 0, i32 0, i32 1
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr01]], i32 noundef 13, i32 noundef 0, i32 noundef 4, i32 noundef 1, ptr noundef [[BufB]])
// CHECK-NEXT: [[ArrayPtr10:%.*]] = getelementptr [2 x [2 x %"class.hlsl::RWBuffer"]], ptr [[TmpResArray]], i32 0, i32 1, i32 0
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr10]], i32 noundef 13, i32 noundef 0, i32 noundef 4, i32 noundef 2, ptr noundef [[BufB]])
// CHECK-NEXT: [[ArrayPtr11:%.*]] = getelementptr [2 x [2 x %"class.hlsl::RWBuffer"]], ptr [[TmpResArray]], i32 0, i32 1, i32 1
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr11]], i32 noundef 13, i32 noundef 0, i32 noundef 4, i32 noundef 3, ptr noundef [[BufB]])

// - Copy from temporary to local array (in a double loop)
// CHECK: %arrayinit.begin = getelementptr inbounds [2 x [2 x %"class.hlsl::RWBuffer"]], ptr %LocalThree, i32 0, i32 0
// CHECK: arrayinit.body:
// CHECK: arrayinit.body2:
// CHECK: call void @hlsl::RWBuffer<float>::RWBuffer(hlsl::RWBuffer<float> const&)(ptr {{.*}}, ptr {{.*}})
// CHECK: arrayinit.end:
// CHECK: arrayinit.end{{[0-9]+}}:
// CHECK: ret void
void case4() {
  // local multi-dimensional array initialized with global array
  RWBuffer<float> LocalThree[2][2] = B;
}

// CHECK-LABEL: case5
// CHECK: %LocalFour = alloca [2 x [2 x %"class.hlsl::RWBuffer"]], align 4

// - Local array is first initialized to poison using the default constructor (in a loop)
//
// CHECK: %array.begin = getelementptr inbounds [2 x [2 x %"class.hlsl::RWBuffer"]], ptr %LocalFour, i32 0, i32 0, i32 0
// CHECK: arrayctor.loop
// CHECK: call void @hlsl::RWBuffer<float>::RWBuffer()(ptr {{.*}})
// CHECK: arrayctor.cont:

// - Initialize individual resource elements directly into the local array
// CHECK: [[ArrayPtr00:%.*]] = getelementptr [2 x [2 x %"class.hlsl::RWBuffer"]], ptr %LocalFour, i32 0, i32 0, i32 0
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr00]], i32 noundef 13, i32 noundef 0, i32 noundef 4, i32 noundef 0, ptr noundef [[BufB]])
// CHECK-NEXT: [[ArrayPtr01:%.*]] = getelementptr [2 x [2 x %"class.hlsl::RWBuffer"]], ptr %LocalFour, i32 0, i32 0, i32 1
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr01]], i32 noundef 13, i32 noundef 0, i32 noundef 4, i32 noundef 1, ptr noundef [[BufB]])
// CHECK-NEXT: [[ArrayPtr10:%.*]] = getelementptr [2 x [2 x %"class.hlsl::RWBuffer"]], ptr %LocalFour, i32 0, i32 1, i32 0
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr10]], i32 noundef 13, i32 noundef 0, i32 noundef 4, i32 noundef 2, ptr noundef [[BufB]])
// CHECK-NEXT: [[ArrayPtr11:%.*]] = getelementptr [2 x [2 x %"class.hlsl::RWBuffer"]], ptr %LocalFour, i32 0, i32 1, i32 1
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr11]], i32 noundef 13, i32 noundef 0, i32 noundef 4, i32 noundef 3, ptr noundef [[BufB]])
// CHECK: ret void
void case5() {
  // local multi-dimensional array initialized with assignment
  RWBuffer<float> LocalFour[2][2];
  LocalFour = B;
}

// CHECK-LABEL: case6
// CHECK: [[AggTmp:%.*]] = alloca [2 x [2 x %"class.hlsl::RWBuffer"]], align 4
// CHECK: [[ArrayPtr00:%.*]] = getelementptr [2 x [2 x %"class.hlsl::RWBuffer"]], ptr [[AggTmp]], i32 0, i32 0, i32 0
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr00]], i32 noundef 13, i32 noundef 0, i32 noundef 4, i32 noundef 0, ptr noundef [[BufB]])
// CHECK-NEXT: [[ArrayPtr01:%.*]] = getelementptr [2 x [2 x %"class.hlsl::RWBuffer"]], ptr [[AggTmp]], i32 0, i32 0, i32 1
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr01]], i32 noundef 13, i32 noundef 0, i32 noundef 4, i32 noundef 1, ptr noundef [[BufB]])
// CHECK-NEXT: [[ArrayPtr10:%.*]] = getelementptr [2 x [2 x %"class.hlsl::RWBuffer"]], ptr [[AggTmp]], i32 0, i32 1, i32 0
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr10]], i32 noundef 13, i32 noundef 0, i32 noundef 4, i32 noundef 2, ptr noundef [[BufB]])
// CHECK-NEXT: [[ArrayPtr11:%.*]] = getelementptr [2 x [2 x %"class.hlsl::RWBuffer"]], ptr [[AggTmp]], i32 0, i32 1, i32 1
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr11]], i32 noundef 13, i32 noundef 0, i32 noundef 4, i32 noundef 3, ptr noundef [[BufB]])
// CHECK-NEXT: call void @useMultiArray(hlsl::RWBuffer<float> [2][2])(ptr noundef byval([2 x [2 x %"class.hlsl::RWBuffer"]]) align 4 %agg.tmp)
// CHECK: ret void
void case6() {
  // resource array as function argument
  useMultiArray(B);
}

// CHECK-LABEL: case7
// CHECK: [[AggTmp:%.*]] = alloca [2 x %"class.hlsl::RWBuffer"], align 4
// CHECK: [[Tmp:%.*]] = alloca [2 x %"class.hlsl::RWBuffer"], align 4
// CHECK-NEXT: [[ArrayPtr0:%.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr [[Tmp]], i32 0, i32 0
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr0]], i32 noundef 13, i32 noundef 0, i32 noundef 4, i32 noundef 2, ptr noundef [[BufB]])
// CHECK-NEXT: [[ArrayPtr1:%.*]] = getelementptr [2 x %"class.hlsl::RWBuffer"], ptr [[Tmp]], i32 0, i32 1
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} [[ArrayPtr1]], i32 noundef 13, i32 noundef 0, i32 noundef 4, i32 noundef 3, ptr noundef [[BufB]])
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[AggTmp]], ptr align 4 [[Tmp]], i32 8, i1 false)
// CHECK-NEXT: call void @useArray(hlsl::RWBuffer<float> [2])(ptr noundef byval([2 x %"class.hlsl::RWBuffer"]) align 4 [[AggTmp]])
void case7() {
  // subset of multi-dimensional resource array as function argument
  useArray(B[1]);
}
