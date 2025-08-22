// RUN: %clang_cc1 -DCK1 -verify -Wno-vla -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -DCK1 -verify -Wno-vla -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK: @.offload_sizes = private unnamed_addr constant [8 x i64] [i64 4, i64 16, i64 8, i64 4, i64 8, i64 4, i64 0, i64 4]
// CHECK: @.offload_maptypes = private unnamed_addr constant [8 x i64] [i64 [[#0x43]], i64 [[#0x43]], i64 [[#0x4000]], i64 [[#0x3]], i64 [[#0x4000]], i64 [[#0x43]], i64 [[#0x43]], i64 [[#0x43]]]
// CHECK: @.offload_sizes.1 = private unnamed_addr constant [8 x i64] [i64 0, i64 4, i64 4, i64 0, i64 16, i64 8, i64 4, i64 8]
// CHECK: @.offload_maptypes.2 = private unnamed_addr constant [8 x i64] [i64 [[#0x0]], i64 [[#0x1000000000043]], i64 [[#0x1000000000053]], i64 [[#0x1000000000043]], i64 [[#0x43]], i64 [[#0x4000]], i64 [[#0x3]], i64 [[#0x4000]]]
struct S {
  int a = 0;
  int *ptr = &a;
  int &ref = a;
  int arr[4];
  S() {}
  void foo() {
    //  &this[0], &this->a,             sizeof(this[0].(a-to-arr[a]), ALLOC
    //  &this[0], &this->a,             sizeof(a),            TO | FROM | RETURN_PARAM | MEMBER_OF(1)
    //  &this[0], &ref_ptee(this->ref), sizeof(this->ref[0]), TO | FROM | PTR_AND_OBJ | RETURN_PARAM | MEMBER_OF(1)
    //  &this[0], &this->arr[0],        4 * sizeof(arr[0]),   TO | FROM | RETURN_PARAM | MEMBER_OF(1)
    //  &ptr[0],  &ptr[3],              4 * sizeof(ptr[0],    TO | FROM | RETURN_PARAM
    //  &ptr,     &ptr[3],              sizeof(void*),        ATTACH
    //  &ptr[0],  &ptr[0],              1 * sizeof(ptr[0],    TO | FROM
    //  &ptr,     &ptr[0],              sizeof(void*),        ATTACH
    #pragma omp target data map(tofrom: a, ptr [3:4], ref, ptr[0], arr[:a]) use_device_addr(a, ptr [3:4], ref, ptr[0], arr[:a])
      ++a, ++*ptr, ++ref, ++arr[0];
  }
};

int main() {
  float a = 0;
  float *ptr = &a;
  float &ref = a;
  float arr[4];
  float vla[(int)a];
  S s;
  s.foo();
  //  &a,             &a,             sizeof(a),             TO | FROM | RETURN_PARAM
  //  &ptr[0],        &ptr[3],        4 * sizeof(ptr[3]),    TO | FROM | RETURN_PARAM
  //  &ptr,           &ptr[3],        sizeof(void*),         ATTACH
  //  &ptr[0],        &ptr[0],        sizeof(ptr[0]),        TO | FROM
  //  &ptr,           &ptr[0],        sizeof(void*),         ATTACH
  //  &ref_ptee(ref), &ref_ptee(ref), sizeof(ref_ptee(ref)), TO | FROM | RETURN_PARAM
  //  &arr,           &arr[0],        a * sizeof(arr[0]),    TO | FROM | RETURN_PARAM
  //  &vla,           &vla[0],        sizeof(vla[0]),        TO | FROM | RETURN_PARAM
  #pragma omp target data map(tofrom: a, ptr [3:4], ref, ptr[0], arr[:(int)a], vla[0]) use_device_addr(a, ptr [3:4], ref, ptr[0], arr[:(int)a], vla[0])
    ++a, ++*ptr, ++ref, ++arr[0], ++vla[0];
  return a;
}


// CHECK-LABEL: define dso_local noundef signext i32 @main(
// CHECK-SAME: ) {{.*}} {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[A:%.*]] = alloca float, align 4
// CHECK-NEXT:    [[PTR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[REF:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[ARR:%.*]] = alloca [4 x float], align 4
// CHECK-NEXT:    [[SAVED_STACK:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[__VLA_EXPR0:%.*]] = alloca i64, align 8
// CHECK-NEXT:    [[S:%.*]] = alloca [[STRUCT_S:%.*]], align 8
// CHECK-NEXT:    [[DOTOFFLOAD_BASEPTRS:%.*]] = alloca [8 x ptr], align 8
// CHECK-NEXT:    [[DOTOFFLOAD_PTRS:%.*]] = alloca [8 x ptr], align 8
// CHECK-NEXT:    [[DOTOFFLOAD_MAPPERS:%.*]] = alloca [8 x ptr], align 8
// CHECK-NEXT:    [[DOTOFFLOAD_SIZES:%.*]] = alloca [8 x i64], align 8
// CHECK-NEXT:    [[TMP:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store i32 0, ptr [[RETVAL]], align 4
// CHECK-NEXT:    store float 0.000000e+00, ptr [[A]], align 4
// CHECK-NEXT:    store ptr [[A]], ptr [[PTR]], align 8
// CHECK-NEXT:    store ptr [[A]], ptr [[REF]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load float, ptr [[A]], align 4
// CHECK-NEXT:    [[CONV:%.*]] = fptosi float [[TMP0]] to i32
// CHECK-NEXT:    [[TMP1:%.*]] = zext i32 [[CONV]] to i64
// CHECK-NEXT:    [[TMP2:%.*]] = call ptr @llvm.stacksave.p0()
// CHECK-NEXT:    store ptr [[TMP2]], ptr [[SAVED_STACK]], align 8
// CHECK-NEXT:    [[VLA:%.*]] = alloca float, i64 [[TMP1]], align 4
// CHECK-NEXT:    store i64 [[TMP1]], ptr [[__VLA_EXPR0]], align 8
// CHECK-NEXT:    call void @_ZN1SC1Ev(ptr noundef nonnull align 8 dereferenceable(40) [[S]])
// CHECK-NEXT:    call void @_ZN1S3fooEv(ptr noundef nonnull align 8 dereferenceable(40) [[S]])
// CHECK-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[PTR]], align 8
// CHECK-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[PTR]], align 8
// CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw float, ptr [[TMP4]], i64 3
// CHECK-NEXT:    [[TMP5:%.*]] = load ptr, ptr [[PTR]], align 8
// CHECK-NEXT:    [[TMP6:%.*]] = load ptr, ptr [[PTR]], align 8
// CHECK-NEXT:    [[ARRAYIDX1:%.*]] = getelementptr inbounds float, ptr [[TMP6]], i64 0
// CHECK-NEXT:    [[TMP7:%.*]] = load ptr, ptr [[REF]], align 8, !nonnull [[META3:![0-9]+]], !align [[META4:![0-9]+]]
// CHECK-NEXT:    [[TMP8:%.*]] = load ptr, ptr [[REF]], align 8, !nonnull [[META3]], !align [[META4]]
// CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds nuw [4 x float], ptr [[ARR]], i64 0, i64 0
// CHECK-NEXT:    [[TMP9:%.*]] = load float, ptr [[A]], align 4
// CHECK-NEXT:    [[CONV3:%.*]] = fptosi float [[TMP9]] to i32
// CHECK-NEXT:    [[CONV4:%.*]] = sext i32 [[CONV3]] to i64
// CHECK-NEXT:    [[TMP10:%.*]] = mul nuw i64 [[CONV4]], 4
// CHECK-NEXT:    [[ARRAYIDX5:%.*]] = getelementptr inbounds float, ptr [[VLA]], i64 0
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[DOTOFFLOAD_SIZES]], ptr align 8 @.offload_sizes, i64 64, i1 false)
// CHECK-NEXT:    [[TMP11:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK-NEXT:    store ptr [[A]], ptr [[TMP11]], align 8
// CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 0
// CHECK-NEXT:    store ptr [[A]], ptr [[TMP12]], align 8
// CHECK-NEXT:    [[TMP13:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 0
// CHECK-NEXT:    store ptr null, ptr [[TMP13]], align 8
// CHECK-NEXT:    [[TMP14:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 1
// CHECK-NEXT:    store ptr [[TMP3]], ptr [[TMP14]], align 8
// CHECK-NEXT:    [[TMP15:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 1
// CHECK-NEXT:    store ptr [[ARRAYIDX]], ptr [[TMP15]], align 8
// CHECK-NEXT:    [[TMP16:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 1
// CHECK-NEXT:    store ptr null, ptr [[TMP16]], align 8
// CHECK-NEXT:    [[TMP17:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 2
// CHECK-NEXT:    store ptr [[PTR]], ptr [[TMP17]], align 8
// CHECK-NEXT:    [[TMP18:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 2
// CHECK-NEXT:    store ptr [[ARRAYIDX]], ptr [[TMP18]], align 8
// CHECK-NEXT:    [[TMP19:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 2
// CHECK-NEXT:    store ptr null, ptr [[TMP19]], align 8
// CHECK-NEXT:    [[TMP20:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 3
// CHECK-NEXT:    store ptr [[TMP5]], ptr [[TMP20]], align 8
// CHECK-NEXT:    [[TMP21:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 3
// CHECK-NEXT:    store ptr [[ARRAYIDX1]], ptr [[TMP21]], align 8
// CHECK-NEXT:    [[TMP22:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 3
// CHECK-NEXT:    store ptr null, ptr [[TMP22]], align 8
// CHECK-NEXT:    [[TMP23:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 4
// CHECK-NEXT:    store ptr [[PTR]], ptr [[TMP23]], align 8
// CHECK-NEXT:    [[TMP24:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 4
// CHECK-NEXT:    store ptr [[ARRAYIDX1]], ptr [[TMP24]], align 8
// CHECK-NEXT:    [[TMP25:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 4
// CHECK-NEXT:    store ptr null, ptr [[TMP25]], align 8
// CHECK-NEXT:    [[TMP26:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 5
// CHECK-NEXT:    store ptr [[TMP7]], ptr [[TMP26]], align 8
// CHECK-NEXT:    [[TMP27:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 5
// CHECK-NEXT:    store ptr [[TMP8]], ptr [[TMP27]], align 8
// CHECK-NEXT:    [[TMP28:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 5
// CHECK-NEXT:    store ptr null, ptr [[TMP28]], align 8
// CHECK-NEXT:    [[TMP29:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 6
// CHECK-NEXT:    store ptr [[ARR]], ptr [[TMP29]], align 8
// CHECK-NEXT:    [[TMP30:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 6
// CHECK-NEXT:    store ptr [[ARRAYIDX2]], ptr [[TMP30]], align 8
// CHECK-NEXT:    [[TMP31:%.*]] = getelementptr inbounds [8 x i64], ptr [[DOTOFFLOAD_SIZES]], i32 0, i32 6
// CHECK-NEXT:    store i64 [[TMP10]], ptr [[TMP31]], align 8
// CHECK-NEXT:    [[TMP32:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 6
// CHECK-NEXT:    store ptr null, ptr [[TMP32]], align 8
// CHECK-NEXT:    [[TMP33:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 7
// CHECK-NEXT:    store ptr [[VLA]], ptr [[TMP33]], align 8
// CHECK-NEXT:    [[TMP34:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 7
// CHECK-NEXT:    store ptr [[ARRAYIDX5]], ptr [[TMP34]], align 8
// CHECK-NEXT:    [[TMP35:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 7
// CHECK-NEXT:    store ptr null, ptr [[TMP35]], align 8
// CHECK-NEXT:    [[TMP36:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK-NEXT:    [[TMP37:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 0
// CHECK-NEXT:    [[TMP38:%.*]] = getelementptr inbounds [8 x i64], ptr [[DOTOFFLOAD_SIZES]], i32 0, i32 0
// CHECK-NEXT:    call void @__tgt_target_data_begin_mapper(ptr @[[GLOB1:[0-9]+]], i64 -1, i32 8, ptr [[TMP36]], ptr [[TMP37]], ptr [[TMP38]], ptr @.offload_maptypes, ptr null, ptr null)
// CHECK-NEXT:    [[TMP39:%.*]] = load ptr, ptr [[TMP11]], align 8
// CHECK-NEXT:    [[TMP40:%.*]] = load ptr, ptr [[TMP26]], align 8
// CHECK-NEXT:    store ptr [[TMP40]], ptr [[TMP]], align 8
// CHECK-NEXT:    [[TMP41:%.*]] = load ptr, ptr [[TMP29]], align 8
// CHECK-NEXT:    [[TMP42:%.*]] = load ptr, ptr [[TMP33]], align 8
// CHECK-NEXT:    [[TMP43:%.*]] = load float, ptr [[TMP39]], align 4
// CHECK-NEXT:    [[INC:%.*]] = fadd float [[TMP43]], 1.000000e+00
// CHECK-NEXT:    store float [[INC]], ptr [[TMP39]], align 4
// CHECK-NEXT:    [[TMP44:%.*]] = load ptr, ptr [[TMP14]], align 8
// CHECK-NEXT:    [[TMP45:%.*]] = load float, ptr [[TMP44]], align 4
// CHECK-NEXT:    [[INC6:%.*]] = fadd float [[TMP45]], 1.000000e+00
// CHECK-NEXT:    store float [[INC6]], ptr [[TMP44]], align 4
// CHECK-NEXT:    [[TMP46:%.*]] = load ptr, ptr [[TMP]], align 8, !nonnull [[META3]], !align [[META4]]
// CHECK-NEXT:    [[TMP47:%.*]] = load float, ptr [[TMP46]], align 4
// CHECK-NEXT:    [[INC7:%.*]] = fadd float [[TMP47]], 1.000000e+00
// CHECK-NEXT:    store float [[INC7]], ptr [[TMP46]], align 4
// CHECK-NEXT:    [[ARRAYIDX8:%.*]] = getelementptr inbounds [4 x float], ptr [[TMP41]], i64 0, i64 0
// CHECK-NEXT:    [[TMP48:%.*]] = load float, ptr [[ARRAYIDX8]], align 4
// CHECK-NEXT:    [[INC9:%.*]] = fadd float [[TMP48]], 1.000000e+00
// CHECK-NEXT:    store float [[INC9]], ptr [[ARRAYIDX8]], align 4
// CHECK-NEXT:    [[ARRAYIDX10:%.*]] = getelementptr inbounds float, ptr [[TMP42]], i64 0
// CHECK-NEXT:    [[TMP49:%.*]] = load float, ptr [[ARRAYIDX10]], align 4
// CHECK-NEXT:    [[INC11:%.*]] = fadd float [[TMP49]], 1.000000e+00
// CHECK-NEXT:    store float [[INC11]], ptr [[ARRAYIDX10]], align 4
// CHECK-NEXT:    [[TMP50:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK-NEXT:    [[TMP51:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 0
// CHECK-NEXT:    [[TMP52:%.*]] = getelementptr inbounds [8 x i64], ptr [[DOTOFFLOAD_SIZES]], i32 0, i32 0
// CHECK-NEXT:    call void @__tgt_target_data_end_mapper(ptr @[[GLOB1]], i64 -1, i32 8, ptr [[TMP50]], ptr [[TMP51]], ptr [[TMP52]], ptr @.offload_maptypes, ptr null, ptr null)

// CHECK-LABEL: define linkonce_odr void @_ZN1S3fooEv(
// CHECK-SAME: ptr noundef nonnull align 8 dereferenceable(40) [[THIS:%.*]]) {{.*}} {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[THIS_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[DOTOFFLOAD_BASEPTRS:%.*]] = alloca [8 x ptr], align 8
// CHECK-NEXT:    [[DOTOFFLOAD_PTRS:%.*]] = alloca [8 x ptr], align 8
// CHECK-NEXT:    [[DOTOFFLOAD_MAPPERS:%.*]] = alloca [8 x ptr], align 8
// CHECK-NEXT:    [[DOTOFFLOAD_SIZES:%.*]] = alloca [8 x i64], align 8
// CHECK-NEXT:    [[TMP:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[_TMP11:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[_TMP12:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[_TMP13:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[_TMP14:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// CHECK-NEXT:    [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// CHECK-NEXT:    [[A:%.*]] = getelementptr inbounds nuw [[STRUCT_S:%.*]], ptr [[THIS1]], i32 0, i32 0
// CHECK-NEXT:    [[REF:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 2
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[REF]], align 8, !nonnull [[META3]], !align [[META4]]
// CHECK-NEXT:    [[ARR:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 3
// CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw [4 x i32], ptr [[ARR]], i64 0, i64 0
// CHECK-NEXT:    [[A2:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 0
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[A2]], align 8
// CHECK-NEXT:    [[CONV:%.*]] = sext i32 [[TMP1]] to i64
// CHECK-NEXT:    [[TMP2:%.*]] = mul nuw i64 [[CONV]], 4
// CHECK-NEXT:    [[A3:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 0
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr [[A3]], align 8
// CHECK-NEXT:    [[TMP4:%.*]] = sext i32 [[TMP3]] to i64
// CHECK-NEXT:    [[LB_ADD_LEN:%.*]] = add nsw i64 -1, [[TMP4]]
// CHECK-NEXT:    [[ARR4:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 3
// CHECK-NEXT:    [[ARRAYIDX5:%.*]] = getelementptr inbounds nuw [4 x i32], ptr [[ARR4]], i64 0, i64 [[LB_ADD_LEN]]
// CHECK-NEXT:    [[TMP5:%.*]] = getelementptr i32, ptr [[ARRAYIDX5]], i32 1
// CHECK-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP5]] to i64
// CHECK-NEXT:    [[TMP7:%.*]] = ptrtoint ptr [[A]] to i64
// CHECK-NEXT:    [[TMP8:%.*]] = sub i64 [[TMP6]], [[TMP7]]
// CHECK-NEXT:    [[TMP9:%.*]] = sdiv exact i64 [[TMP8]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK-NEXT:    [[PTR:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 1
// CHECK-NEXT:    [[TMP10:%.*]] = load ptr, ptr [[PTR]], align 8
// CHECK-NEXT:    [[PTR6:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 1
// CHECK-NEXT:    [[TMP11:%.*]] = load ptr, ptr [[PTR6]], align 8
// CHECK-NEXT:    [[ARRAYIDX7:%.*]] = getelementptr inbounds nuw i32, ptr [[TMP11]], i64 3
// CHECK-NEXT:    [[PTR8:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 1
// CHECK-NEXT:    [[TMP12:%.*]] = load ptr, ptr [[PTR8]], align 8
// CHECK-NEXT:    [[PTR9:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 1
// CHECK-NEXT:    [[TMP13:%.*]] = load ptr, ptr [[PTR9]], align 8
// CHECK-NEXT:    [[ARRAYIDX10:%.*]] = getelementptr inbounds i32, ptr [[TMP13]], i64 0
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[DOTOFFLOAD_SIZES]], ptr align 8 @.offload_sizes.1, i64 64, i1 false)
// CHECK-NEXT:    [[TMP14:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK-NEXT:    store ptr [[THIS1]], ptr [[TMP14]], align 8
// CHECK-NEXT:    [[TMP15:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 0
// CHECK-NEXT:    store ptr [[A]], ptr [[TMP15]], align 8
// CHECK-NEXT:    [[TMP16:%.*]] = getelementptr inbounds [8 x i64], ptr [[DOTOFFLOAD_SIZES]], i32 0, i32 0
// CHECK-NEXT:    store i64 [[TMP9]], ptr [[TMP16]], align 8
// CHECK-NEXT:    [[TMP17:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 0
// CHECK-NEXT:    store ptr null, ptr [[TMP17]], align 8
// CHECK-NEXT:    [[TMP18:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 1
// CHECK-NEXT:    store ptr [[THIS1]], ptr [[TMP18]], align 8
// CHECK-NEXT:    [[TMP19:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 1
// CHECK-NEXT:    store ptr [[A]], ptr [[TMP19]], align 8
// CHECK-NEXT:    [[TMP20:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 1
// CHECK-NEXT:    store ptr null, ptr [[TMP20]], align 8
// CHECK-NEXT:    [[TMP21:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 2
// CHECK-NEXT:    store ptr [[THIS1]], ptr [[TMP21]], align 8
// CHECK-NEXT:    [[TMP22:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 2
// CHECK-NEXT:    store ptr [[TMP0]], ptr [[TMP22]], align 8
// CHECK-NEXT:    [[TMP23:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 2
// CHECK-NEXT:    store ptr null, ptr [[TMP23]], align 8
// CHECK-NEXT:    [[TMP24:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 3
// CHECK-NEXT:    store ptr [[THIS1]], ptr [[TMP24]], align 8
// CHECK-NEXT:    [[TMP25:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 3
// CHECK-NEXT:    store ptr [[ARRAYIDX]], ptr [[TMP25]], align 8
// CHECK-NEXT:    [[TMP26:%.*]] = getelementptr inbounds [8 x i64], ptr [[DOTOFFLOAD_SIZES]], i32 0, i32 3
// CHECK-NEXT:    store i64 [[TMP2]], ptr [[TMP26]], align 8
// CHECK-NEXT:    [[TMP27:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 3
// CHECK-NEXT:    store ptr null, ptr [[TMP27]], align 8
// CHECK-NEXT:    [[TMP28:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 4
// CHECK-NEXT:    store ptr [[TMP10]], ptr [[TMP28]], align 8
// CHECK-NEXT:    [[TMP29:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 4
// CHECK-NEXT:    store ptr [[ARRAYIDX7]], ptr [[TMP29]], align 8
// CHECK-NEXT:    [[TMP30:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 4
// CHECK-NEXT:    store ptr null, ptr [[TMP30]], align 8
// CHECK-NEXT:    [[TMP31:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 5
// CHECK-NEXT:    store ptr [[PTR]], ptr [[TMP31]], align 8
// CHECK-NEXT:    [[TMP32:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 5
// CHECK-NEXT:    store ptr [[ARRAYIDX7]], ptr [[TMP32]], align 8
// CHECK-NEXT:    [[TMP33:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 5
// CHECK-NEXT:    store ptr null, ptr [[TMP33]], align 8
// CHECK-NEXT:    [[TMP34:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 6
// CHECK-NEXT:    store ptr [[TMP12]], ptr [[TMP34]], align 8
// CHECK-NEXT:    [[TMP35:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 6
// CHECK-NEXT:    store ptr [[ARRAYIDX10]], ptr [[TMP35]], align 8
// CHECK-NEXT:    [[TMP36:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 6
// CHECK-NEXT:    store ptr null, ptr [[TMP36]], align 8
// CHECK-NEXT:    [[TMP37:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 7
// CHECK-NEXT:    store ptr [[PTR8]], ptr [[TMP37]], align 8
// CHECK-NEXT:    [[TMP38:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 7
// CHECK-NEXT:    store ptr [[ARRAYIDX10]], ptr [[TMP38]], align 8
// CHECK-NEXT:    [[TMP39:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 7
// CHECK-NEXT:    store ptr null, ptr [[TMP39]], align 8
// CHECK-NEXT:    [[TMP40:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK-NEXT:    [[TMP41:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 0
// CHECK-NEXT:    [[TMP42:%.*]] = getelementptr inbounds [8 x i64], ptr [[DOTOFFLOAD_SIZES]], i32 0, i32 0
// CHECK-NEXT:    call void @__tgt_target_data_begin_mapper(ptr @[[GLOB1]], i64 -1, i32 8, ptr [[TMP40]], ptr [[TMP41]], ptr [[TMP42]], ptr @.offload_maptypes.2, ptr null, ptr null)
// CHECK-NEXT:    [[TMP43:%.*]] = load ptr, ptr [[TMP18]], align 8
// CHECK-NEXT:    store ptr [[TMP43]], ptr [[TMP]], align 8
// CHECK-NEXT:    [[TMP44:%.*]] = load ptr, ptr [[TMP28]], align 8
// CHECK-NEXT:    store ptr [[TMP44]], ptr [[_TMP11]], align 8
// CHECK-NEXT:    [[TMP45:%.*]] = load ptr, ptr [[TMP21]], align 8
// CHECK-NEXT:    store ptr [[TMP45]], ptr [[_TMP12]], align 8
// CHECK-NEXT:    [[TMP46:%.*]] = load ptr, ptr [[TMP28]], align 8
// CHECK-NEXT:    store ptr [[TMP46]], ptr [[_TMP13]], align 8
// CHECK-NEXT:    [[TMP47:%.*]] = load ptr, ptr [[TMP24]], align 8
// CHECK-NEXT:    store ptr [[TMP47]], ptr [[_TMP14]], align 8
// CHECK-NEXT:    [[TMP48:%.*]] = load ptr, ptr [[TMP]], align 8, !nonnull [[META3]], !align [[META4]]
// CHECK-NEXT:    [[TMP49:%.*]] = load i32, ptr [[TMP48]], align 4
// CHECK-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP49]], 1
// CHECK-NEXT:    store i32 [[INC]], ptr [[TMP48]], align 4
// CHECK-NEXT:    [[TMP50:%.*]] = load ptr, ptr [[_TMP13]], align 8, !nonnull [[META3]], !align [[META5:![0-9]+]]
// CHECK-NEXT:    [[TMP51:%.*]] = load ptr, ptr [[TMP50]], align 8
// CHECK-NEXT:    [[TMP52:%.*]] = load i32, ptr [[TMP51]], align 4
// CHECK-NEXT:    [[INC15:%.*]] = add nsw i32 [[TMP52]], 1
// CHECK-NEXT:    store i32 [[INC15]], ptr [[TMP51]], align 4
// CHECK-NEXT:    [[TMP53:%.*]] = load ptr, ptr [[_TMP12]], align 8, !nonnull [[META3]], !align [[META4]]
// CHECK-NEXT:    [[TMP54:%.*]] = load i32, ptr [[TMP53]], align 4
// CHECK-NEXT:    [[INC16:%.*]] = add nsw i32 [[TMP54]], 1
// CHECK-NEXT:    store i32 [[INC16]], ptr [[TMP53]], align 4
// CHECK-NEXT:    [[TMP55:%.*]] = load ptr, ptr [[_TMP14]], align 8, !nonnull [[META3]], !align [[META4]]
// CHECK-NEXT:    [[ARRAYIDX17:%.*]] = getelementptr inbounds [4 x i32], ptr [[TMP55]], i64 0, i64 0
// CHECK-NEXT:    [[TMP56:%.*]] = load i32, ptr [[ARRAYIDX17]], align 4
// CHECK-NEXT:    [[INC18:%.*]] = add nsw i32 [[TMP56]], 1
// CHECK-NEXT:    store i32 [[INC18]], ptr [[ARRAYIDX17]], align 4
// CHECK-NEXT:    [[TMP57:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK-NEXT:    [[TMP58:%.*]] = getelementptr inbounds [8 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 0
// CHECK-NEXT:    [[TMP59:%.*]] = getelementptr inbounds [8 x i64], ptr [[DOTOFFLOAD_SIZES]], i32 0, i32 0
// CHECK-NEXT:    call void @__tgt_target_data_end_mapper(ptr @[[GLOB1]], i64 -1, i32 8, ptr [[TMP57]], ptr [[TMP58]], ptr [[TMP59]], ptr @.offload_maptypes.2, ptr null, ptr null)
// CHECK-NEXT:    ret void
#endif
