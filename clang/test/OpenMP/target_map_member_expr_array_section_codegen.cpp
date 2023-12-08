// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: [[SIZE_ENTER:@.+]] = private unnamed_addr constant [2 x i64] [i64 0, i64 24]
// 0 = OMP_MAP_NONE
// 281474976710656 = 0x1000000000000 = OMP_MAP_MEMBER_OF of 1-st element
// CHECK: [[MAP_ENTER:@.+]] = private unnamed_addr constant [2 x i64] [i64 0, i64 281474976710656]
// CHECK: [[SIZE_EXIT:@.+]] = private unnamed_addr constant [2 x i64] [i64 0, i64 24]
// 281474976710664 = 0x1000000000008 = OMP_MAP_MEMBER_OF of 1-st element | OMP_MAP_DELETE
// CHECK: [[MAP_EXIT:@.+]] = private unnamed_addr constant [2 x i64] [i64 0, i64 281474976710664]
template <typename T>
struct S {
  constexpr static int size = 6;
  T data[size];
};

template <typename T>
struct maptest {
  S<T> s;
  maptest() {
    // CHECK: [[BPTRS:%.+]] = alloca [2 x ptr],
    // CHECK: [[PTRS:%.+]] = alloca [2 x ptr],
    // CHECK: [[SIZES:%.+]] = alloca [2 x i64],
    // CHECK: getelementptr inbounds
    // CHECK: [[S_ADDR:%.+]] = getelementptr inbounds %struct.maptest, ptr [[THIS:%.+]], i32 0, i32 0
    // CHECK: [[S_DATA_ADDR:%.+]] = getelementptr inbounds %struct.S, ptr [[S_ADDR]], i32 0, i32 0
    // CHECK: [[S_DATA_0_ADDR:%.+]] = getelementptr inbounds [6 x float], ptr [[S_DATA_ADDR]], i64 0, i64 0

    // SZ = &this->s.data[6]-&this->s.data[0]
    // CHECK: [[S_ADDR:%.+]] = getelementptr inbounds %struct.maptest, ptr [[THIS]], i32 0, i32 0
    // CHECK: [[S_DATA_ADDR:%.+]] = getelementptr inbounds %struct.S, ptr [[S_ADDR]], i32 0, i32 0
    // CHECK: [[S_DATA_5_ADDR:%.+]] = getelementptr inbounds [6 x float], ptr [[S_DATA_ADDR]], i64 0, i64 5
    // CHECK: [[S_DATA_6_ADDR:%.+]] = getelementptr float, ptr [[S_DATA_5_ADDR]], i32 1
    // CHECK: [[END_BC:%.+]] = ptrtoint ptr [[S_DATA_6_ADDR]] to i64
    // CHECK: [[BEG_BC:%.+]] = ptrtoint ptr [[S_DATA_0_ADDR]] to i64
    // CHECK: [[DIFF:%.+]] = sub i64 [[END_BC]], [[BEG_BC]]
    // CHECK: [[SZ:%.+]] = sdiv exact i64 [[DIFF]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)

    // Fill mapping arrays
    // CHECK: [[BPTR0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BPTRS]], i32 0, i32 0
    // CHECK: store ptr [[THIS]], ptr [[BPTR0]],
    // CHECK: [[PTR0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[PTRS]], i32 0, i32 0
    // CHECK: store ptr [[S_DATA_0_ADDR]], ptr [[PTR0]],
    // CHECK: [[SIZE0:%.+]] = getelementptr inbounds [2 x i64], ptr [[SIZES]], i32 0, i32 0
    // CHECK: store i64 [[SZ]], ptr [[SIZE0]],
    // CHECK: [[BPTR1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BPTRS]], i32 0, i32 1
    // CHECK: store ptr [[THIS]], ptr [[BPTR1]],
    // CHECK: [[PTR1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[PTRS]], i32 0, i32 1
    // CHECK: store ptr [[S_DATA_0_ADDR]], ptr [[PTR1]],
    // CHECK: [[BPTR:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BPTRS]], i32 0, i32 0
    // CHECK: [[PTR:%.+]] = getelementptr inbounds [2 x ptr], ptr [[PTRS]], i32 0, i32 0
    // CHECK: [[SIZE:%.+]] = getelementptr inbounds [2 x i64], ptr [[SIZES]], i32 0, i32 0
    // CHECK: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 2, ptr [[BPTR]], ptr [[PTR]], ptr [[SIZE]], ptr [[MAP_ENTER]], ptr null, ptr null)
#pragma omp target enter data map(alloc : s.data[:6])
  }

  ~maptest() {
    // CHECK: [[BPTRS:%.+]] = alloca [2 x ptr],
    // CHECK: [[PTRS:%.+]] = alloca [2 x ptr],
    // CHECK: [[SIZE:%.+]] = alloca [2 x i64],
    // CHECK: [[S_ADDR:%.+]] = getelementptr inbounds %struct.maptest, ptr [[THIS:%.+]], i32 0, i32 0
    // CHECK: [[S_DATA_ADDR:%.+]] = getelementptr inbounds %struct.S, ptr [[S_ADDR]], i32 0, i32 0
    // CHECK: [[S_DATA_0_ADDR:%.+]] = getelementptr inbounds [6 x float], ptr [[S_DATA_ADDR]], i64 0, i64 0

    // SZ = &this->s.data[6]-&this->s.data[0]
    // CHECK: [[S_ADDR:%.+]] = getelementptr inbounds %struct.maptest, ptr [[THIS]], i32 0, i32 0
    // CHECK: [[S_DATA_ADDR:%.+]] = getelementptr inbounds %struct.S, ptr [[S_ADDR]], i32 0, i32 0
    // CHECK: [[S_DATA_5_ADDR:%.+]] = getelementptr inbounds [6 x float], ptr [[S_DATA_ADDR]], i64 0, i64 5
    // CHECK: [[S_DATA_6_ADDR:%.+]] = getelementptr float, ptr [[S_DATA_5_ADDR]], i32 1
    // CHECK: [[END_BC:%.+]] = ptrtoint ptr [[S_DATA_6_ADDR]] to i64
    // CHECK: [[BEG_BC:%.+]] = ptrtoint ptr [[S_DATA_0_ADDR]] to i64
    // CHECK: [[DIFF:%.+]] = sub i64 [[END_BC]], [[BEG_BC]]
    // CHECK: [[SZ:%.+]] = sdiv exact i64 [[DIFF]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)

    // Fill mapping arrays
    // CHECK: [[BPTR0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BPTRS]], i32 0, i32 0
    // CHECK: store ptr [[THIS]], ptr [[BPTR0]],
    // CHECK: [[PTR0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[PTRS]], i32 0, i32 0
    // CHECK: store ptr [[S_DATA_0_ADDR]], ptr [[PTR0]],
    // CHECK: [[SIZE0:%.+]] = getelementptr inbounds [2 x i64], ptr [[SIZES]], i32 0, i32 0
    // CHECK: store i64 [[SZ]], ptr [[SIZE0]],
    // CHECK: [[BPTR1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BPTRS]], i32 0, i32 1
    // CHECK: store ptr [[THIS]], ptr [[BPTR1]],
    // CHECK: [[PTR1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[PTRS]], i32 0, i32 1
    // CHECK: store ptr [[S_DATA_0_ADDR]], ptr [[PTR1]],
    // CHECK: [[BPTR:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BPTRS]], i32 0, i32 0
    // CHECK: [[PTR:%.+]] = getelementptr inbounds [2 x ptr], ptr [[PTRS]], i32 0, i32 0
    // CHECK: [[SIZE:%.+]] = getelementptr inbounds [2 x i64], ptr [[SIZES]], i32 0, i32 0
    // CHECK: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 2, ptr [[BPTR]], ptr [[PTR]], ptr [[SIZE]], ptr [[MAP_EXIT]], ptr null, ptr null)
#pragma omp target exit data map(delete : s.data[:6])
  }
};

maptest<float> a;

#endif
