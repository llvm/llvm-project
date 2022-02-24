// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// 0 = OMP_MAP_NONE
// 281474976710656 = 0x1000000000000 = OMP_MAP_MEMBER_OF of 1-st element
// CHECK: [[MAP_ENTER:@.+]] = private unnamed_addr constant [2 x i64] [i64 0, i64 281474976710656]
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
    // CHECK: [[BPTRS:%.+]] = alloca [2 x i8*],
    // CHECK: [[PTRS:%.+]] = alloca [2 x i8*],
    // CHECK: [[SIZES:%.+]] = alloca [2 x i64],
    // CHECK: getelementptr inbounds
    // CHECK: [[S_ADDR:%.+]] = getelementptr inbounds %struct.maptest, %struct.maptest* [[THIS:%.+]], i32 0, i32 0
    // CHECK: [[S_DATA_ADDR:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[S_ADDR]], i32 0, i32 0
    // CHECK: [[S_DATA_0_ADDR:%.+]] = getelementptr inbounds [6 x float], [6 x float]* [[S_DATA_ADDR]], i64 0, i64 0

    // SZ = &this->s.data[6]-&this->s.data[0]
    // CHECK: [[S_ADDR:%.+]] = getelementptr inbounds %struct.maptest, %struct.maptest* [[THIS]], i32 0, i32 0
    // CHECK: [[S_DATA_ADDR:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[S_ADDR]], i32 0, i32 0
    // CHECK: [[S_DATA_5_ADDR:%.+]] = getelementptr inbounds [6 x float], [6 x float]* [[S_DATA_ADDR]], i64 0, i64 5
    // CHECK: [[S_DATA_6_ADDR:%.+]] = getelementptr float, float* [[S_DATA_5_ADDR]], i32 1
    // CHECK: [[BEG:%.+]] = bitcast float* [[S_DATA_0_ADDR]] to i8*
    // CHECK: [[END:%.+]] = bitcast float* [[S_DATA_6_ADDR]] to i8*
    // CHECK: [[END_BC:%.+]] = ptrtoint i8* [[END]] to i64
    // CHECK: [[BEG_BC:%.+]] = ptrtoint i8* [[BEG]] to i64
    // CHECK: [[DIFF:%.+]] = sub i64 [[END_BC]], [[BEG_BC]]
    // CHECK: [[SZ:%.+]] = sdiv exact i64 [[DIFF]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)

    // Fill mapping arrays
    // CHECK: [[BPTR0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BPTRS]], i32 0, i32 0
    // CHECK: [[BPTR0_THIS:%.+]] = bitcast i8** [[BPTR0]] to %struct.maptest**
    // CHECK: store %struct.maptest* [[THIS]], %struct.maptest** [[BPTR0_THIS]],
    // CHECK: [[PTR0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[PTRS]], i32 0, i32 0
    // CHECK: [[PTR0_DATA:%.+]] = bitcast i8** [[PTR0]] to float**
    // CHECK: store float* [[S_DATA_0_ADDR]], float** [[PTR0_DATA]],
    // CHECK: [[SIZE0:%.+]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZES]], i32 0, i32 0
    // CHECK: store i64 [[SZ]], i64* [[SIZE0]],
    // CHECK: [[BPTR1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BPTRS]], i32 0, i32 1
    // CHECK: [[BPTR1_THIS:%.+]] = bitcast i8** [[BPTR1]] to %struct.maptest**
    // CHECK: store %struct.maptest* [[THIS]], %struct.maptest** [[BPTR1_THIS]],
    // CHECK: [[PTR1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[PTRS]], i32 0, i32 1
    // CHECK: [[PTR1_DATA:%.+]] = bitcast i8** [[PTR1]] to float**
    // CHECK: store float* [[S_DATA_0_ADDR]], float** [[PTR1_DATA]],
    // CHECK: [[SIZE1:%.+]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZES]], i32 0, i32 1
    // CHECK: store i64 24, i64* [[SIZE1]],
    // CHECK: [[BPTR:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BPTRS]], i32 0, i32 0
    // CHECK: [[PTR:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[PTRS]], i32 0, i32 0
    // CHECK: [[SIZE:%.+]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZES]], i32 0, i32 0
    // CHECK: call void @__tgt_target_data_begin_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 2, i8** [[BPTR]], i8** [[PTR]], i64* [[SIZE]], i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[MAP_ENTER]], i32 0, i32 0), i8** null, i8** null)
#pragma omp target enter data map(alloc : s.data[:6])
  }

  ~maptest() {
    // CHECK: [[BPTRS:%.+]] = alloca [2 x i8*],
    // CHECK: [[PTRS:%.+]] = alloca [2 x i8*],
    // CHECK: [[SIZE:%.+]] = alloca [2 x i64],
    // CHECK: [[S_ADDR:%.+]] = getelementptr inbounds %struct.maptest, %struct.maptest* [[THIS:%.+]], i32 0, i32 0
    // CHECK: [[S_DATA_ADDR:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[S_ADDR]], i32 0, i32 0
    // CHECK: [[S_DATA_0_ADDR:%.+]] = getelementptr inbounds [6 x float], [6 x float]* [[S_DATA_ADDR]], i64 0, i64 0

    // SZ = &this->s.data[6]-&this->s.data[0]
    // CHECK: [[S_ADDR:%.+]] = getelementptr inbounds %struct.maptest, %struct.maptest* [[THIS]], i32 0, i32 0
    // CHECK: [[S_DATA_ADDR:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[S_ADDR]], i32 0, i32 0
    // CHECK: [[S_DATA_5_ADDR:%.+]] = getelementptr inbounds [6 x float], [6 x float]* [[S_DATA_ADDR]], i64 0, i64 5
    // CHECK: [[S_DATA_6_ADDR:%.+]] = getelementptr float, float* [[S_DATA_5_ADDR]], i32 1
    // CHECK: [[BEG:%.+]] = bitcast float* [[S_DATA_0_ADDR]] to i8*
    // CHECK: [[END:%.+]] = bitcast float* [[S_DATA_6_ADDR]] to i8*
    // CHECK: [[END_BC:%.+]] = ptrtoint i8* [[END]] to i64
    // CHECK: [[BEG_BC:%.+]] = ptrtoint i8* [[BEG]] to i64
    // CHECK: [[DIFF:%.+]] = sub i64 [[END_BC]], [[BEG_BC]]
    // CHECK: [[SZ:%.+]] = sdiv exact i64 [[DIFF]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)

    // Fill mapping arrays
    // CHECK: [[BPTR0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BPTRS]], i32 0, i32 0
    // CHECK: [[BPTR0_THIS:%.+]] = bitcast i8** [[BPTR0]] to %struct.maptest**
    // CHECK: store %struct.maptest* [[THIS]], %struct.maptest** [[BPTR0_THIS]],
    // CHECK: [[PTR0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[PTRS]], i32 0, i32 0
    // CHECK: [[PTR0_DATA:%.+]] = bitcast i8** [[PTR0]] to float**
    // CHECK: store float* [[S_DATA_0_ADDR]], float** [[PTR0_DATA]],
    // CHECK: [[SIZE0:%.+]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZES]], i32 0, i32 0
    // CHECK: store i64 [[SZ]], i64* [[SIZE0]],
    // CHECK: [[BPTR1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BPTRS]], i32 0, i32 1
    // CHECK: [[BPTR1_THIS:%.+]] = bitcast i8** [[BPTR1]] to %struct.maptest**
    // CHECK: store %struct.maptest* [[THIS]], %struct.maptest** [[BPTR1_THIS]],
    // CHECK: [[PTR1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[PTRS]], i32 0, i32 1
    // CHECK: [[PTR1_DATA:%.+]] = bitcast i8** [[PTR1]] to float**
    // CHECK: store float* [[S_DATA_0_ADDR]], float** [[PTR1_DATA]],
    // CHECK: [[SIZE1:%.+]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZES]], i32 0, i32 1
    // CHECK: store i64 24, i64* [[SIZE1]],
    // CHECK: [[BPTR:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BPTRS]], i32 0, i32 0
    // CHECK: [[PTR:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[PTRS]], i32 0, i32 0
    // CHECK: [[SIZE:%.+]] = getelementptr inbounds [2 x i64], [2 x i64]* [[SIZES]], i32 0, i32 0
    // CHECK: call void @__tgt_target_data_end_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 2, i8** [[BPTR]], i8** [[PTR]], i64* [[SIZE]], i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[MAP_EXIT]], i32 0, i32 0), i8** null, i8** null)
#pragma omp target exit data map(delete : s.data[:6])
  }
};

maptest<float> a;

#endif
