// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK-DAG: [[SIZES1:@.+]] = private unnamed_addr constant [5 x i64] zeroinitializer
// 64 = 0x40 = OMP_MAP_RETURN_PARAM
// CHECK-DAG: [[MAPTYPES1:@.+]] = private unnamed_addr constant [5 x i64] [i64 64, i64 64, i64 64, i64 64, i64 64]
// CHECK-DAG: [[SIZES2:@.+]] = private unnamed_addr constant [5 x i64] zeroinitializer
// 0 = OMP_MAP_NONE
// 281474976710720 = 0x1000000000040 = OMP_MAP_MEMBER_OF | OMP_MAP_RETURN_PARAM
// CHECK-DAG: [[MAPTYPES2:@.+]] = private unnamed_addr constant [5 x i64] [i64 0, i64 281474976710720, i64 281474976710720, i64 281474976710720, i64 281474976710720]
struct S {
  int a = 0;
  int *ptr = &a;
  int &ref = a;
  int arr[4];
  S() {}
  void foo() {
#pragma omp target data use_device_addr(a, ptr [3:4], ref, ptr[0], arr[:a])
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
#pragma omp target data use_device_addr(a, ptr [3:4], ref, ptr[0], arr[:(int)a], vla[0])
  ++a, ++*ptr, ++ref, ++arr[0], ++vla[0];
  return a;
}

// CHECK-LABEL: @main()
// CHECK: [[A_ADDR:%.+]] = alloca float,
// CHECK: [[PTR_ADDR:%.+]] = alloca float*,
// CHECK: [[REF_ADDR:%.+]] = alloca float*,
// CHECK: [[ARR_ADDR:%.+]] = alloca [4 x float],
// CHECK: [[BPTRS:%.+]] = alloca [5 x i8*],
// CHECK: [[PTRS:%.+]] = alloca [5 x i8*],
// CHECK: [[VLA_ADDR:%.+]] = alloca float, i64 %{{.+}},
// CHECK: [[PTR:%.+]] = load float*, float** [[PTR_ADDR]],
// CHECK: [[REF:%.+]] = load float*, float** [[REF_ADDR]],
// CHECK: [[ARR:%.+]] = getelementptr inbounds [4 x float], [4 x float]* [[ARR_ADDR]], i64 0, i64 0
// CHECK: [[BPTR0:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BPTRS]], i32 0, i32 0
// CHECK: [[BPTR0_A_ADDR:%.+]] = bitcast i8** [[BPTR0]] to float**
// CHECK: store float* [[A_ADDR]], float** [[BPTR0_A_ADDR]],
// CHECK: [[PTR0:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i32 0, i32 0
// CHECK: [[PTR0_A_ADDR:%.+]] = bitcast i8** [[PTR0]] to float**
// CHECK: store float* [[A_ADDR]], float** [[PTR0_A_ADDR]],
// CHECK: [[BPTR1:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BPTRS]], i32 0, i32 1
// CHECK: [[BPTR1_PTR_ADDR:%.+]] = bitcast i8** [[BPTR1]] to float**
// CHECK: store float* [[PTR]], float** [[BPTR1_PTR_ADDR]],
// CHECK: [[PTR1:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i32 0, i32 1
// CHECK: [[PTR1_PTR_ADDR:%.+]] = bitcast i8** [[PTR1]] to float**
// CHECK: store float* [[PTR]], float** [[PTR1_PTR_ADDR]],
// CHECK: [[BPTR2:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BPTRS]], i32 0, i32 2
// CHECK: [[BPTR2_REF_ADDR:%.+]] = bitcast i8** [[BPTR2]] to float**
// CHECK: store float* [[REF]], float** [[BPTR2_REF_ADDR]],
// CHECK: [[PTR2:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i32 0, i32 2
// CHECK: [[PTR2_REF_ADDR:%.+]] = bitcast i8** [[PTR2]] to float**
// CHECK: store float* [[REF]], float** [[PTR2_REF_ADDR]],
// CHECK: [[BPTR3:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BPTRS]], i32 0, i32 3
// CHECK: [[BPTR3_ARR_ADDR:%.+]] = bitcast i8** [[BPTR3]] to float**
// CHECK: store float* [[ARR]], float** [[BPTR3_ARR_ADDR]],
// CHECK: [[PTR3:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i32 0, i32 3
// CHECK: [[PTR3_ARR_ADDR:%.+]] = bitcast i8** [[PTR3]] to float**
// CHECK: store float* [[ARR]], float** [[PTR3_ARR_ADDR]],
// CHECK: [[BPTR4:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BPTRS]], i32 0, i32 4
// CHECK: [[BPTR4_VLA_ADDR:%.+]] = bitcast i8** [[BPTR4]] to float**
// CHECK: store float* [[VLA_ADDR]], float** [[BPTR4_VLA_ADDR]],
// CHECK: [[PTR4:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i32 0, i32 4
// CHECK: [[PTR4_VLA_ADDR:%.+]] = bitcast i8** [[PTR4]] to float**
// CHECK: store float* [[VLA_ADDR]], float** [[PTR4_VLA_ADDR]],
// CHECK: [[BPTR:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BPTRS]], i32 0, i32 0
// CHECK: [[PTR:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_begin_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 5, i8** [[BPTR]], i8** [[PTR]], i64* getelementptr inbounds ([5 x i64], [5 x i64]* [[SIZES1]], i32 0, i32 0), i64* getelementptr inbounds ([5 x i64], [5 x i64]* [[MAPTYPES1]], i32 0, i32 0), i8** null, i8** null)
// CHECK: [[A_REF:%.+]] = load float*, float** [[BPTR0_A_ADDR]],
// CHECK: [[REF_REF:%.+]] = load float*, float** [[BPTR2_REF_ADDR]],
// CHECK: store float* [[REF_REF]], float** [[TMP_REF_ADDR:%.+]],
// CHECK: [[BPTR3_ARR_ADDR_CAST:%.+]] = bitcast float** [[BPTR3_ARR_ADDR]] to [4 x float]**
// CHECK: [[ARR_REF:%.+]] = load [4 x float]*, [4 x float]** [[BPTR3_ARR_ADDR_CAST]],
// CHECK: [[VLA_REF:%.+]] = load float*, float** [[BPTR4_VLA_ADDR]],
// CHECK: [[A:%.+]] = load float, float* [[A_REF]],
// CHECK: [[INC:%.+]] = fadd float [[A]], 1.000000e+00
// CHECK: store float [[INC]], float* [[A_REF]],
// CHECK: [[PTR_ADDR:%.+]] = load float*, float** [[BPTR1_PTR_ADDR]],
// CHECK: [[VAL:%.+]] = load float, float* [[PTR_ADDR]],
// CHECK: [[INC:%.+]] = fadd float [[VAL]], 1.000000e+00
// CHECK: store float [[INC]], float* [[PTR_ADDR]],
// CHECK: [[REF_ADDR:%.+]] = load float*, float** [[TMP_REF_ADDR]],
// CHECK: [[REF:%.+]] = load float, float* [[REF_ADDR]],
// CHECK: [[INC:%.+]] = fadd float [[REF]], 1.000000e+00
// CHECK: store float [[INC]], float* [[REF_ADDR]],
// CHECK: [[ARR0_ADDR:%.+]] = getelementptr inbounds [4 x float], [4 x float]* [[ARR_REF]], i64 0, i64 0
// CHECK: [[ARR0:%.+]] = load float, float* [[ARR0_ADDR]],
// CHECK: [[INC:%.+]] = fadd float [[ARR0]], 1.000000e+00
// CHECK: store float [[INC]], float* [[ARR0_ADDR]],
// CHECK: [[VLA0_ADDR:%.+]] = getelementptr inbounds float, float* [[VLA_REF]], i64 0
// CHECK: [[VLA0:%.+]] = load float, float* [[VLA0_ADDR]],
// CHECK: [[INC:%.+]] = fadd float [[VLA0]], 1.000000e+00
// CHECK: store float [[INC]], float* [[VLA0_ADDR]],
// CHECK: [[BPTR:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BPTRS]], i32 0, i32 0
// CHECK: [[PTR:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_end_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 5, i8** [[BPTR]], i8** [[PTR]], i64* getelementptr inbounds ([5 x i64], [5 x i64]* [[SIZES1]], i32 0, i32 0), i64* getelementptr inbounds ([5 x i64], [5 x i64]* [[MAPTYPES1]], i32 0, i32 0), i8** null, i8** null)

// CHECK: foo
// %this.addr = alloca %struct.S*, align 8
// CHECK: [[BPTRS:%.+]] = alloca [5 x i8*],
// CHECK: [[PTRS:%.+]] = alloca [5 x i8*],
// CHECK: [[SIZES:%.+]] = alloca [5 x i64],
// %tmp = alloca i32*, align 8
// %tmp6 = alloca i32**, align 8
// %tmp7 = alloca i32*, align 8
// %tmp8 = alloca i32**, align 8
// %tmp9 = alloca [4 x i32]*, align 8
// store %struct.S* %this, %struct.S** %this.addr, align 8
// %this1 = load %struct.S*, %struct.S** %this.addr, align 8
// CHECK: [[A_ADDR:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[THIS:%.+]], i32 0, i32 0
// %ptr = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 1
// %ref = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 2
// %0 = load i32*, i32** %ref, align 8
// CHECK: [[ARR_ADDR:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[THIS]], i32 0, i32 3
// CHECK: [[A_ADDR2:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[THIS]], i32 0, i32 0
// CHECK: [[PTR_ADDR:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[THIS]], i32 0, i32 1
// CHECK: [[REF_REF:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[THIS]], i32 0, i32 2
// CHECK: [[REF_PTR:%.+]] = load i32*, i32** [[REF_REF]],
// CHECK: [[ARR_ADDR2:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[THIS]], i32 0, i32 3
// CHECK: [[ARR_END:%.+]] = getelementptr [4 x i32], [4 x i32]* [[ARR_ADDR]], i32 1
// CHECK: [[BEGIN:%.+]] = bitcast i32* [[A_ADDR]] to i8*
// CHECK: [[END:%.+]] = bitcast [4 x i32]* [[ARR_END]] to i8*
// CHECK: [[E:%.+]] = ptrtoint i8* [[END]] to i64
// CHECK: [[B:%.+]] = ptrtoint i8* [[BEGIN]] to i64
// CHECK: [[DIFF:%.+]] = sub i64 [[E]], [[B]]
// CHECK: [[SZ:%.+]] = sdiv exact i64 [[DIFF]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
// CHECK: [[BPTR0:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BPTRS]], i32 0, i32 0
// CHECK: [[BPTR0_S:%.+]] = bitcast i8** [[BPTR0]] to %struct.S**
// CHECK: store %struct.S* [[THIS]], %struct.S** [[BPTR0_S]],
// CHECK: [[PTR0:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i32 0, i32 0
// CHECK: [[PTR0_BEGIN:%.+]] = bitcast i8** [[PTR0]] to i32**
// CHECK: store i32* [[A_ADDR]], i32** [[PTR0_BEGIN]],
// CHECK: [[SIZE0:%.+]] = getelementptr inbounds [5 x i64], [5 x i64]* [[SIZES]], i32 0, i32 0
// CHECK: store i64 [[SZ]], i64* [[SIZE0]],
// CHECK: [[BPTR1:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BPTRS]], i32 0, i32 1
// CHECK: [[BPTR1_A_ADDR:%.+]] = bitcast i8** [[BPTR1]] to i32**
// CHECK: store i32* [[A_ADDR2]], i32** [[BPTR1_A_ADDR]],
// CHECK: [[PTR1:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i32 0, i32 1
// CHECK: [[PTR1_A_ADDR:%.+]] = bitcast i8** [[PTR1]] to i32**
// CHECK: store i32* [[A_ADDR2]], i32** [[PTR1_A_ADDR]],
// CHECK: [[BPTR2:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BPTRS]], i32 0, i32 2
// CHECK: [[BPTR2_PTR_ADDR:%.+]] = bitcast i8** [[BPTR2]] to i32***
// CHECK: store i32** [[PTR_ADDR]], i32*** [[BPTR2_PTR_ADDR]],
// CHECK: [[PTR2:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i32 0, i32 2
// CHECK: [[PTR2_PTR_ADDR:%.+]] = bitcast i8** [[PTR2]] to i32***
// CHECK: store i32** [[PTR_ADDR]], i32*** [[PTR2_PTR_ADDR]],
// CHECK: [[BPTR3:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BPTRS]], i32 0, i32 3
// CHECK: [[BPTR3_REF_PTR:%.+]] = bitcast i8** [[BPTR3]] to i32**
// CHECK: store i32* [[REF_PTR]], i32** [[BPTR3_REF_PTR]],
// CHECK: [[PTR3:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i32 0, i32 3
// CHECK: [[PTR3_REF_PTR:%.+]] = bitcast i8** [[PTR3]] to i32**
// CHECK: store i32* [[REF_PTR]], i32** [[PTR3_REF_PTR]],
// CHECK: [[BPTR4:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BPTRS]], i32 0, i32 4
// CHECK: [[BPTR4_ARR_ADDR:%.+]] = bitcast i8** [[BPTR4]] to [4 x i32]**
// CHECK: store [4 x i32]* [[ARR_ADDR2]], [4 x i32]** [[BPTR4_ARR_ADDR]],
// CHECK: [[PTR4:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i32 0, i32 4
// CHECK: [[PTR4_ARR_ADDR:%.+]] = bitcast i8** [[PTR4]] to [4 x i32]**
// CHECK: store [4 x i32]* [[ARR_ADDR2]], [4 x i32]** [[PTR4_ARR_ADDR]],
// CHECK: [[BPTR:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BPTRS]], i32 0, i32 0
// CHECK: [[PTR:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i32 0, i32 0
// CHECK: [[SIZE:%.+]] = getelementptr inbounds [5 x i64], [5 x i64]* [[SIZES]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_begin_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 5, i8** [[BPTR]], i8** [[PTR]], i64* [[SIZE]], i64* getelementptr inbounds ([5 x i64], [5 x i64]* [[MAPTYPES2]], i32 0, i32 0), i8** null, i8** null)
// CHECK: [[A_ADDR:%.+]] = load i32*, i32** [[BPTR1_A_ADDR]],
// CHECK: store i32* [[A_ADDR]], i32** [[A_REF:%.+]],
// CHECK: [[PTR_ADDR:%.+]] = load i32**, i32*** [[BPTR2_PTR_ADDR]],
// CHECK: store i32** [[PTR_ADDR]], i32*** [[PTR_REF:%.+]],
// CHECK: [[REF_PTR:%.+]] = load i32*, i32** [[BPTR3_REF_PTR]],
// CHECK: store i32* [[REF_PTR]], i32** [[REF_REF:%.+]],
// CHECK: [[PTR_ADDR:%.+]] = load i32**, i32*** [[BPTR2_PTR_ADDR]],
// CHECK: store i32** [[PTR_ADDR]], i32*** [[PTR_REF2:%.+]],
// CHECK: [[ARR_ADDR:%.+]] = load [4 x i32]*, [4 x i32]** [[BPTR4_ARR_ADDR]],
// CHECK: store [4 x i32]* [[ARR_ADDR]], [4 x i32]** [[ARR_REF:%.+]],
// CHECK: [[A_ADDR:%.+]] = load i32*, i32** [[A_REF]],
// CHECK: [[A:%.+]] = load i32, i32* [[A_ADDR]],
// CHECK: [[INC:%.+]] = add nsw i32 [[A]], 1
// CHECK: store i32 [[INC]], i32* [[A_ADDR]],
// CHECK: [[PTR_PTR:%.+]] = load i32**, i32*** [[PTR_REF2]],
// CHECK: [[PTR:%.+]] = load i32*, i32** [[PTR_PTR]],
// CHECK: [[VAL:%.+]] = load i32, i32* [[PTR]],
// CHECK: [[INC:%.+]] = add nsw i32 [[VAL]], 1
// CHECK: store i32 [[INC]], i32* [[PTR]],
// CHECK: [[REF_PTR:%.+]] = load i32*, i32** [[REF_REF]],
// CHECK: [[VAL:%.+]] = load i32, i32* [[REF_PTR]],
// CHECK: [[INC:%.+]] = add nsw i32 [[VAL]], 1
// CHECK: store i32 [[INC]], i32* [[REF_PTR]],
// CHECK: [[ARR_ADDR:%.+]] = load [4 x i32]*, [4 x i32]** [[ARR_REF]],
// CHECK: [[ARR0_ADDR:%.+]] = getelementptr inbounds [4 x i32], [4 x i32]* [[ARR_ADDR]], i64 0, i64 0
// CHECK: [[VAL:%.+]] = load i32, i32* [[ARR0_ADDR]],
// CHECK: [[INC:%.+]] = add nsw i32 [[VAL]], 1
// CHECK: store i32 [[INC]], i32* [[ARR0_ADDR]],
// CHECK: [[BPTR:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BPTRS]], i32 0, i32 0
// CHECK: [[PTR:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i32 0, i32 0
// CHECK: [[SIZE:%.+]] = getelementptr inbounds [5 x i64], [5 x i64]* [[SIZES]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_end_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 5, i8** [[BPTR]], i8** [[PTR]], i64* [[SIZE]], i64* getelementptr inbounds ([5 x i64], [5 x i64]* [[MAPTYPES2]], i32 0, i32 0), i8** null, i8** null)

#endif
