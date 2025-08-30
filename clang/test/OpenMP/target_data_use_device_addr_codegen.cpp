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

// CHECK: @.offload_sizes = private unnamed_addr constant [7 x i64] [i64 4, i64 16, i64 4, i64 8, i64 4, i64 0, i64 4]
// CHECK: @.offload_maptypes = private unnamed_addr constant [7 x i64] [i64 [[#0x43]], i64 [[#0x43]], i64 [[#0x3]], i64 [[#0x4000]], i64 [[#0x43]], i64 [[#0x43]], i64 [[#0x43]]]
// CHECK: @.offload_sizes.1 = private unnamed_addr constant [7 x i64] [i64 0, i64 4, i64 4, i64 0, i64 16, i64 4, i64 8]
// CHECK: @.offload_maptypes.2 = private unnamed_addr constant [7 x i64] [i64 [[#0x0]], i64 [[#0x1000000000043]], i64 [[#0x1000000000053]], i64 [[#0x1000000000043]], i64 [[#0x43]], i64 [[#0x3]], i64 [[#0x4000]]]
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
  //  &ptr[0],        &ptr[0],        sizeof(ptr[0]),        TO | FROM
  //  &ptr,           &ptr[0],        sizeof(void*),         ATTACH
  //  &ref_ptee(ref), &ref_ptee(ref), sizeof(ref_ptee(ref)), TO | FROM | RETURN_PARAM
  //  &arr,           &arr[0],        a * sizeof(arr[0]),    TO | FROM | RETURN_PARAM
  //  &vla,           &vla[0],        sizeof(vla[0]),        TO | FROM | RETURN_PARAM
  #pragma omp target data map(tofrom: a, ptr [3:4], ref, ptr[0], arr[:(int)a], vla[0]) use_device_addr(a, ptr [3:4], ref, ptr[0], arr[:(int)a], vla[0])
    ++a, ++*ptr, ++ref, ++arr[0], ++vla[0];
  return a;
}


// CHECK-LABEL: define {{.*}}main
// CHECK:  [[ENTRY:.*:]]
// CHECK:    [[RETVAL:%.*]] = alloca i32, align 4
// CHECK:    [[A:%.*]] = alloca float, align 4
// CHECK:    [[PTR:%.*]] = alloca ptr, align 8
// CHECK:    [[REF:%.*]] = alloca ptr, align 8
// CHECK:    [[ARR:%.*]] = alloca [4 x float], align 4
// CHECK:    [[SAVED_STACK:%.*]] = alloca ptr, align 8
// CHECK:    [[__VLA_EXPR0:%.*]] = alloca i64, align 8
// CHECK:    [[S:%.*]] = alloca [[STRUCT_S:%.*]], align 8
// CHECK:    [[DOTOFFLOAD_BASEPTRS:%.*]] = alloca [7 x ptr], align 8
// CHECK:    [[DOTOFFLOAD_PTRS:%.*]] = alloca [7 x ptr], align 8
// CHECK:    [[DOTOFFLOAD_MAPPERS:%.*]] = alloca [7 x ptr], align 8
// CHECK:    [[DOTOFFLOAD_SIZES:%.*]] = alloca [7 x i64], align 8
// CHECK:    [[TMP:%.*]] = alloca ptr, align 8
// CHECK:    store i32 0, ptr [[RETVAL]], align 4
// CHECK:    store float 0.000000e+00, ptr [[A]], align 4
// CHECK:    store ptr [[A]], ptr [[PTR]], align 8
// CHECK:    store ptr [[A]], ptr [[REF]], align 8
// CHECK:    [[TMP0:%.*]] = load float, ptr [[A]], align 4
// CHECK:    [[CONV:%.*]] = fptosi float [[TMP0]] to i32
// CHECK:    [[TMP1:%.*]] = zext i32 [[CONV]] to i64
// CHECK:    [[TMP2:%.*]] = call ptr @llvm.stacksave.p0()
// CHECK:    store ptr [[TMP2]], ptr [[SAVED_STACK]], align 8
// CHECK:    [[VLA:%.*]] = alloca float, i64 [[TMP1]], align 4
// CHECK:    store i64 [[TMP1]], ptr [[__VLA_EXPR0]], align 8
// CHECK:    call void @_ZN1SC1Ev(ptr noundef nonnull align 8 dereferenceable(40) [[S]])
// CHECK:    call void @_ZN1S3fooEv(ptr noundef nonnull align 8 dereferenceable(40) [[S]])
// CHECK:    [[TMP3:%.*]] = load ptr, ptr [[PTR]], align 8
// CHECK:    [[TMP4:%.*]] = load ptr, ptr [[PTR]], align 8
// CHECK:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw float, ptr [[TMP4]], i64 3
// CHECK:    [[TMP5:%.*]] = load ptr, ptr [[PTR]], align 8
// CHECK:    [[TMP6:%.*]] = load ptr, ptr [[PTR]], align 8
// CHECK:    [[ARRAYIDX1:%.*]] = getelementptr inbounds float, ptr [[TMP6]], i64 0
// CHECK:    [[TMP7:%.*]] = load ptr, ptr [[REF]], align 8, !nonnull [[META3:![0-9]+]], !align [[META4:![0-9]+]]
// CHECK:    [[TMP8:%.*]] = load ptr, ptr [[REF]], align 8, !nonnull [[META3]], !align [[META4]]
// CHECK:    [[ARRAYIDX2:%.*]] = getelementptr inbounds nuw [4 x float], ptr [[ARR]], i64 0, i64 0
// CHECK:    [[TMP9:%.*]] = load float, ptr [[A]], align 4
// CHECK:    [[CONV3:%.*]] = fptosi float [[TMP9]] to i32
// CHECK:    [[CONV4:%.*]] = sext i32 [[CONV3]] to i64
// CHECK:    [[TMP10:%.*]] = mul nuw i64 [[CONV4]], 4
// CHECK:    [[ARRAYIDX5:%.*]] = getelementptr inbounds float, ptr [[VLA]], i64 0
// CHECK:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[DOTOFFLOAD_SIZES]], ptr align 8 @.offload_sizes, i64 56, i1 false)
// CHECK:    [[TMP11:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK:    store ptr [[A]], ptr [[TMP11]], align 8
// CHECK:    [[TMP12:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 0
// CHECK:    store ptr [[A]], ptr [[TMP12]], align 8
// CHECK:    [[TMP13:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 0
// CHECK:    [[TMP14:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 1
// CHECK:    store ptr [[TMP3]], ptr [[TMP14]], align 8
// CHECK:    [[TMP15:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 1
// CHECK:    store ptr [[ARRAYIDX]], ptr [[TMP15]], align 8
// CHECK:    [[TMP16:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 1
// CHECK:    [[TMP17:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 2
// CHECK:    store ptr [[TMP5]], ptr [[TMP17]], align 8
// CHECK:    [[TMP18:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 2
// CHECK:    store ptr [[ARRAYIDX1]], ptr [[TMP18]], align 8
// CHECK:    [[TMP19:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 2
// CHECK:    [[TMP20:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 3
// CHECK:    store ptr [[PTR]], ptr [[TMP20]], align 8
// CHECK:    [[TMP21:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 3
// CHECK:    store ptr [[ARRAYIDX1]], ptr [[TMP21]], align 8
// CHECK:    [[TMP22:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 3
// CHECK:    [[TMP23:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 4
// CHECK:    store ptr [[TMP7]], ptr [[TMP23]], align 8
// CHECK:    [[TMP24:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 4
// CHECK:    store ptr [[TMP8]], ptr [[TMP24]], align 8
// CHECK:    [[TMP25:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 4
// CHECK:    [[TMP26:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 5
// CHECK:    store ptr [[ARR]], ptr [[TMP26]], align 8
// CHECK:    [[TMP27:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 5
// CHECK:    store ptr [[ARRAYIDX2]], ptr [[TMP27]], align 8
// CHECK:    [[TMP28:%.*]] = getelementptr inbounds [7 x i64], ptr [[DOTOFFLOAD_SIZES]], i32 0, i32 5
// CHECK:    store i64 [[TMP10]], ptr [[TMP28]], align 8
// CHECK:    [[TMP29:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 5
// CHECK:    [[TMP30:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 6
// CHECK:    store ptr [[VLA]], ptr [[TMP30]], align 8
// CHECK:    [[TMP31:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 6
// CHECK:    store ptr [[ARRAYIDX5]], ptr [[TMP31]], align 8
// CHECK:    [[TMP32:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 6
// CHECK:    [[TMP33:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK:    [[TMP34:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 0
// CHECK:    [[TMP35:%.*]] = getelementptr inbounds [7 x i64], ptr [[DOTOFFLOAD_SIZES]], i32 0, i32 0
// CHECK:    call void @__tgt_target_data_begin_mapper(ptr @[[GLOB1:[0-9]+]], i64 -1, i32 7, ptr [[TMP33]], ptr [[TMP34]], ptr [[TMP35]], ptr @.offload_maptypes, ptr null, ptr null)
// CHECK:    [[TMP36:%.*]] = load ptr, ptr [[TMP11]], align 8
// CHECK:    [[TMP37:%.*]] = load ptr, ptr [[TMP23]], align 8
// CHECK:    store ptr [[TMP37]], ptr [[TMP]], align 8
// CHECK:    [[TMP38:%.*]] = load ptr, ptr [[TMP26]], align 8
// CHECK:    [[TMP39:%.*]] = load ptr, ptr [[TMP30]], align 8
// CHECK:    [[TMP40:%.*]] = load float, ptr [[TMP36]], align 4
// CHECK:    [[INC:%.*]] = fadd float [[TMP40]], 1.000000e+00
// CHECK:    store float [[INC]], ptr [[TMP36]], align 4
// CHECK:    [[TMP41:%.*]] = load ptr, ptr [[TMP14]], align 8
// CHECK:    [[TMP42:%.*]] = load float, ptr [[TMP41]], align 4
// CHECK:    [[INC6:%.*]] = fadd float [[TMP42]], 1.000000e+00
// CHECK:    store float [[INC6]], ptr [[TMP41]], align 4
// CHECK:    [[TMP43:%.*]] = load ptr, ptr [[TMP]], align 8, !nonnull [[META3]], !align [[META4]]
// CHECK:    [[TMP44:%.*]] = load float, ptr [[TMP43]], align 4
// CHECK:    [[INC7:%.*]] = fadd float [[TMP44]], 1.000000e+00
// CHECK:    store float [[INC7]], ptr [[TMP43]], align 4
// CHECK:    [[ARRAYIDX8:%.*]] = getelementptr inbounds [4 x float], ptr [[TMP38]], i64 0, i64 0
// CHECK:    [[TMP45:%.*]] = load float, ptr [[ARRAYIDX8]], align 4
// CHECK:    [[INC9:%.*]] = fadd float [[TMP45]], 1.000000e+00
// CHECK:    store float [[INC9]], ptr [[ARRAYIDX8]], align 4
// CHECK:    [[ARRAYIDX10:%.*]] = getelementptr inbounds float, ptr [[TMP39]], i64 0
// CHECK:    [[TMP46:%.*]] = load float, ptr [[ARRAYIDX10]], align 4
// CHECK:    [[INC11:%.*]] = fadd float [[TMP46]], 1.000000e+00
// CHECK:    store float [[INC11]], ptr [[ARRAYIDX10]], align 4
// CHECK:    [[TMP47:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK:    [[TMP48:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 0
// CHECK:    [[TMP49:%.*]] = getelementptr inbounds [7 x i64], ptr [[DOTOFFLOAD_SIZES]], i32 0, i32 0
// CHECK:    call void @__tgt_target_data_end_mapper(ptr @[[GLOB1]], i64 -1, i32 7, ptr [[TMP47]], ptr [[TMP48]], ptr [[TMP49]], ptr @.offload_maptypes, ptr null, ptr null)


// CHECK-LABEL: define {{.*}}foo
// CHECK-SAME: ptr noundef nonnull align 8 dereferenceable(40) [[THIS:%.*]])
// CHECK:  [[ENTRY:.*:]]
// CHECK:    [[THIS_ADDR:%.*]] = alloca ptr, align 8
// CHECK:    [[DOTOFFLOAD_BASEPTRS:%.*]] = alloca [7 x ptr], align 8
// CHECK:    [[DOTOFFLOAD_PTRS:%.*]] = alloca [7 x ptr], align 8
// CHECK:    [[DOTOFFLOAD_MAPPERS:%.*]] = alloca [7 x ptr], align 8
// CHECK:    [[DOTOFFLOAD_SIZES:%.*]] = alloca [7 x i64], align 8
// CHECK:    [[TMP:%.*]] = alloca ptr, align 8
// CHECK:    [[_TMP11:%.*]] = alloca ptr, align 8
// CHECK:    [[_TMP12:%.*]] = alloca ptr, align 8
// CHECK:    [[_TMP13:%.*]] = alloca ptr, align 8
// CHECK:    [[_TMP14:%.*]] = alloca ptr, align 8
// CHECK:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// CHECK:    [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// CHECK:    [[A:%.*]] = getelementptr inbounds nuw [[STRUCT_S:%.*]], ptr [[THIS1]], i32 0, i32 0
// CHECK:    [[REF:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 2
// CHECK:    [[TMP0:%.*]] = load ptr, ptr [[REF]], align 8, !nonnull [[META3]], !align [[META4]]
// CHECK:    [[ARR:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 3
// CHECK:    [[ARRAYIDX:%.*]] = getelementptr inbounds nuw [4 x i32], ptr [[ARR]], i64 0, i64 0
// CHECK:    [[A2:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 0
// CHECK:    [[TMP1:%.*]] = load i32, ptr [[A2]], align 8
// CHECK:    [[CONV:%.*]] = sext i32 [[TMP1]] to i64
// CHECK:    [[TMP2:%.*]] = mul nuw i64 [[CONV]], 4
// CHECK:    [[A3:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 0
// CHECK:    [[TMP3:%.*]] = load i32, ptr [[A3]], align 8
// CHECK:    [[TMP4:%.*]] = sext i32 [[TMP3]] to i64
// CHECK:    [[LB_ADD_LEN:%.*]] = add nsw i64 -1, [[TMP4]]
// CHECK:    [[ARR4:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 3
// CHECK:    [[ARRAYIDX5:%.*]] = getelementptr inbounds nuw [4 x i32], ptr [[ARR4]], i64 0, i64 [[LB_ADD_LEN]]
// CHECK:    [[TMP5:%.*]] = getelementptr i32, ptr [[ARRAYIDX5]], i32 1
// CHECK:    [[TMP6:%.*]] = ptrtoint ptr [[TMP5]] to i64
// CHECK:    [[TMP7:%.*]] = ptrtoint ptr [[A]] to i64
// CHECK:    [[TMP8:%.*]] = sub i64 [[TMP6]], [[TMP7]]
// CHECK:    [[TMP9:%.*]] = sdiv exact i64 [[TMP8]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK:    [[PTR:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 1
// CHECK:    [[TMP10:%.*]] = load ptr, ptr [[PTR]], align 8
// CHECK:    [[PTR6:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 1
// CHECK:    [[TMP11:%.*]] = load ptr, ptr [[PTR6]], align 8
// CHECK:    [[ARRAYIDX7:%.*]] = getelementptr inbounds nuw i32, ptr [[TMP11]], i64 3
// CHECK:    [[PTR8:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 1
// CHECK:    [[TMP12:%.*]] = load ptr, ptr [[PTR8]], align 8
// CHECK:    [[PTR9:%.*]] = getelementptr inbounds nuw [[STRUCT_S]], ptr [[THIS1]], i32 0, i32 1
// CHECK:    [[TMP13:%.*]] = load ptr, ptr [[PTR9]], align 8
// CHECK:    [[ARRAYIDX10:%.*]] = getelementptr inbounds i32, ptr [[TMP13]], i64 0
// CHECK:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[DOTOFFLOAD_SIZES]], ptr align 8 @.offload_sizes.1, i64 56, i1 false)
// CHECK:    [[TMP14:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK:    store ptr [[THIS1]], ptr [[TMP14]], align 8
// CHECK:    [[TMP15:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 0
// CHECK:    store ptr [[A]], ptr [[TMP15]], align 8
// CHECK:    [[TMP16:%.*]] = getelementptr inbounds [7 x i64], ptr [[DOTOFFLOAD_SIZES]], i32 0, i32 0
// CHECK:    store i64 [[TMP9]], ptr [[TMP16]], align 8
// CHECK:    [[TMP17:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 0
// CHECK:    [[TMP18:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 1
// CHECK:    store ptr [[THIS1]], ptr [[TMP18]], align 8
// CHECK:    [[TMP19:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 1
// CHECK:    store ptr [[A]], ptr [[TMP19]], align 8
// CHECK:    [[TMP20:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 1
// CHECK:    [[TMP21:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 2
// CHECK:    store ptr [[THIS1]], ptr [[TMP21]], align 8
// CHECK:    [[TMP22:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 2
// CHECK:    store ptr [[TMP0]], ptr [[TMP22]], align 8
// CHECK:    [[TMP23:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 2
// CHECK:    [[TMP24:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 3
// CHECK:    store ptr [[THIS1]], ptr [[TMP24]], align 8
// CHECK:    [[TMP25:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 3
// CHECK:    store ptr [[ARRAYIDX]], ptr [[TMP25]], align 8
// CHECK:    [[TMP26:%.*]] = getelementptr inbounds [7 x i64], ptr [[DOTOFFLOAD_SIZES]], i32 0, i32 3
// CHECK:    store i64 [[TMP2]], ptr [[TMP26]], align 8
// CHECK:    [[TMP27:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 3
// CHECK:    [[TMP28:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 4
// CHECK:    store ptr [[TMP10]], ptr [[TMP28]], align 8
// CHECK:    [[TMP29:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 4
// CHECK:    store ptr [[ARRAYIDX7]], ptr [[TMP29]], align 8
// CHECK:    [[TMP30:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 4
// CHECK:    [[TMP31:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 5
// CHECK:    store ptr [[TMP12]], ptr [[TMP31]], align 8
// CHECK:    [[TMP32:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 5
// CHECK:    store ptr [[ARRAYIDX10]], ptr [[TMP32]], align 8
// CHECK:    [[TMP33:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 5
// CHECK:    [[TMP34:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 6
// CHECK:    store ptr [[PTR8]], ptr [[TMP34]], align 8
// CHECK:    [[TMP35:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 6
// CHECK:    store ptr [[ARRAYIDX10]], ptr [[TMP35]], align 8
// CHECK:    [[TMP36:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_MAPPERS]], i64 0, i64 6
// CHECK:    [[TMP37:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK:    [[TMP38:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 0
// CHECK:    [[TMP39:%.*]] = getelementptr inbounds [7 x i64], ptr [[DOTOFFLOAD_SIZES]], i32 0, i32 0
// CHECK:    call void @__tgt_target_data_begin_mapper(ptr @[[GLOB1]], i64 -1, i32 7, ptr [[TMP37]], ptr [[TMP38]], ptr [[TMP39]], ptr @.offload_maptypes.2, ptr null, ptr null)
// CHECK:    [[TMP40:%.*]] = load ptr, ptr [[TMP18]], align 8
// CHECK:    store ptr [[TMP40]], ptr [[TMP]], align 8
// CHECK:    [[TMP41:%.*]] = load ptr, ptr [[TMP28]], align 8
// CHECK:    store ptr [[TMP41]], ptr [[_TMP11]], align 8
// CHECK:    [[TMP42:%.*]] = load ptr, ptr [[TMP21]], align 8
// CHECK:    store ptr [[TMP42]], ptr [[_TMP12]], align 8
// CHECK:    [[TMP43:%.*]] = load ptr, ptr [[TMP28]], align 8
// CHECK:    store ptr [[TMP43]], ptr [[_TMP13]], align 8
// CHECK:    [[TMP44:%.*]] = load ptr, ptr [[TMP24]], align 8
// CHECK:    store ptr [[TMP44]], ptr [[_TMP14]], align 8
// CHECK:    [[TMP45:%.*]] = load ptr, ptr [[TMP]], align 8, !nonnull [[META3]], !align [[META4]]
// CHECK:    [[TMP46:%.*]] = load i32, ptr [[TMP45]], align 4
// CHECK:    [[INC:%.*]] = add nsw i32 [[TMP46]], 1
// CHECK:    store i32 [[INC]], ptr [[TMP45]], align 4
// CHECK:    [[TMP47:%.*]] = load ptr, ptr [[_TMP13]], align 8, !nonnull [[META3]], !align [[META5:![0-9]+]]
// CHECK:    [[TMP48:%.*]] = load ptr, ptr [[TMP47]], align 8
// CHECK:    [[TMP49:%.*]] = load i32, ptr [[TMP48]], align 4
// CHECK:    [[INC15:%.*]] = add nsw i32 [[TMP49]], 1
// CHECK:    store i32 [[INC15]], ptr [[TMP48]], align 4
// CHECK:    [[TMP50:%.*]] = load ptr, ptr [[_TMP12]], align 8, !nonnull [[META3]], !align [[META4]]
// CHECK:    [[TMP51:%.*]] = load i32, ptr [[TMP50]], align 4
// CHECK:    [[INC16:%.*]] = add nsw i32 [[TMP51]], 1
// CHECK:    store i32 [[INC16]], ptr [[TMP50]], align 4
// CHECK:    [[TMP52:%.*]] = load ptr, ptr [[_TMP14]], align 8, !nonnull [[META3]], !align [[META4]]
// CHECK:    [[ARRAYIDX17:%.*]] = getelementptr inbounds [4 x i32], ptr [[TMP52]], i64 0, i64 0
// CHECK:    [[TMP53:%.*]] = load i32, ptr [[ARRAYIDX17]], align 4
// CHECK:    [[INC18:%.*]] = add nsw i32 [[TMP53]], 1
// CHECK:    store i32 [[INC18]], ptr [[ARRAYIDX17]], align 4
// CHECK:    [[TMP54:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK:    [[TMP55:%.*]] = getelementptr inbounds [7 x ptr], ptr [[DOTOFFLOAD_PTRS]], i32 0, i32 0
// CHECK:    [[TMP56:%.*]] = getelementptr inbounds [7 x i64], ptr [[DOTOFFLOAD_SIZES]], i32 0, i32 0
// CHECK:    call void @__tgt_target_data_end_mapper(ptr @[[GLOB1]], i64 -1, i32 7, ptr [[TMP54]], ptr [[TMP55]], ptr [[TMP56]], ptr @.offload_maptypes.2, ptr null, ptr null)
#endif
