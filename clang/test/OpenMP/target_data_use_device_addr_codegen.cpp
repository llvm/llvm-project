// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
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
// CHECK: [[PTR_ADDR:%.+]] = alloca ptr,
// CHECK: [[REF_ADDR:%.+]] = alloca ptr,
// CHECK: [[ARR_ADDR:%.+]] = alloca [4 x float],
// CHECK: [[BPTRS:%.+]] = alloca [5 x ptr],
// CHECK: [[PTRS:%.+]] = alloca [5 x ptr],
// CHECK: [[VLA_ADDR:%.+]] = alloca float, i64 %{{.+}},
// CHECK: [[PTR:%.+]] = load ptr, ptr [[PTR_ADDR]],
// CHECK: [[REF:%.+]] = load ptr, ptr [[REF_ADDR]],
// CHECK: [[ARR:%.+]] = getelementptr inbounds [4 x float], ptr [[ARR_ADDR]], i64 0, i64 0
// CHECK: [[BPTR0:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BPTRS]], i32 0, i32 0
// CHECK: store ptr [[A_ADDR]], ptr [[BPTR0]],
// CHECK: [[PTR0:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 0
// CHECK: store ptr [[A_ADDR]], ptr [[PTR0]],
// CHECK: [[BPTR1:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BPTRS]], i32 0, i32 1
// CHECK: store ptr [[PTR]], ptr [[BPTR1]],
// CHECK: [[PTR1:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 1
// CHECK: store ptr [[PTR]], ptr [[PTR1]],
// CHECK: [[BPTR2:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BPTRS]], i32 0, i32 2
// CHECK: store ptr [[REF]], ptr [[BPTR2]],
// CHECK: [[PTR2:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 2
// CHECK: store ptr [[REF]], ptr [[PTR2]],
// CHECK: [[BPTR3:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BPTRS]], i32 0, i32 3
// CHECK: store ptr [[ARR]], ptr [[BPTR3]],
// CHECK: [[PTR3:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 3
// CHECK: store ptr [[ARR]], ptr [[PTR3]],
// CHECK: [[BPTR4:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BPTRS]], i32 0, i32 4
// CHECK: store ptr [[VLA_ADDR]], ptr [[BPTR4]],
// CHECK: [[PTR4:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 4
// CHECK: store ptr [[VLA_ADDR]], ptr [[PTR4]],
// CHECK: [[BPTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BPTRS]], i32 0, i32 0
// CHECK: [[PTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 5, ptr [[BPTR]], ptr [[PTR]], ptr [[SIZES1]], ptr [[MAPTYPES1]], ptr null, ptr null)
// CHECK: [[A_REF:%.+]] = load ptr, ptr [[BPTR0]],
// CHECK: [[REF_REF:%.+]] = load ptr, ptr [[BPTR2]],
// CHECK: store ptr [[REF_REF]], ptr [[TMP_REF_ADDR:%.+]],
// CHECK: [[ARR_REF:%.+]] = load ptr, ptr [[BPTR3]],
// CHECK: [[VLA_REF:%.+]] = load ptr, ptr [[BPTR4]],
// CHECK: [[A:%.+]] = load float, ptr [[A_REF]],
// CHECK: [[INC:%.+]] = fadd float [[A]], 1.000000e+00
// CHECK: store float [[INC]], ptr [[A_REF]],
// CHECK: [[PTR_ADDR:%.+]] = load ptr, ptr [[BPTR1]],
// CHECK: [[VAL:%.+]] = load float, ptr [[PTR_ADDR]],
// CHECK: [[INC:%.+]] = fadd float [[VAL]], 1.000000e+00
// CHECK: store float [[INC]], ptr [[PTR_ADDR]],
// CHECK: [[REF_ADDR:%.+]] = load ptr, ptr [[TMP_REF_ADDR]],
// CHECK: [[REF:%.+]] = load float, ptr [[REF_ADDR]],
// CHECK: [[INC:%.+]] = fadd float [[REF]], 1.000000e+00
// CHECK: store float [[INC]], ptr [[REF_ADDR]],
// CHECK: [[ARR0_ADDR:%.+]] = getelementptr inbounds [4 x float], ptr [[ARR_REF]], i64 0, i64 0
// CHECK: [[ARR0:%.+]] = load float, ptr [[ARR0_ADDR]],
// CHECK: [[INC:%.+]] = fadd float [[ARR0]], 1.000000e+00
// CHECK: store float [[INC]], ptr [[ARR0_ADDR]],
// CHECK: [[VLA0_ADDR:%.+]] = getelementptr inbounds float, ptr [[VLA_REF]], i64 0
// CHECK: [[VLA0:%.+]] = load float, ptr [[VLA0_ADDR]],
// CHECK: [[INC:%.+]] = fadd float [[VLA0]], 1.000000e+00
// CHECK: store float [[INC]], ptr [[VLA0_ADDR]],
// CHECK: [[BPTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BPTRS]], i32 0, i32 0
// CHECK: [[PTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 5, ptr [[BPTR]], ptr [[PTR]], ptr [[SIZES1]], ptr [[MAPTYPES1]], ptr null, ptr null)

// CHECK: foo
// %this.addr = alloca ptr, align 8
// CHECK: [[BPTRS:%.+]] = alloca [5 x ptr],
// CHECK: [[PTRS:%.+]] = alloca [5 x ptr],
// CHECK: [[SIZES:%.+]] = alloca [5 x i64],
// %tmp = alloca ptr, align 8
// %tmp6 = alloca ptr, align 8
// %tmp7 = alloca ptr, align 8
// %tmp8 = alloca ptr, align 8
// %tmp9 = alloca ptr, align 8
// store ptr %this, ptr %this.addr, align 8
// %this1 = load ptr, ptr %this.addr, align 8
// CHECK: [[A_ADDR:%.+]] = getelementptr inbounds %struct.S, ptr [[THIS:%.+]], i32 0, i32 0
// %ptr = getelementptr inbounds %struct.S, ptr %this1, i32 0, i32 1
// %ref = getelementptr inbounds %struct.S, ptr %this1, i32 0, i32 2
// %0 = load ptr, ptr %ref, align 8
// CHECK: [[ARR_ADDR:%.+]] = getelementptr inbounds %struct.S, ptr [[THIS]], i32 0, i32 3
// CHECK: [[A_ADDR2:%.+]] = getelementptr inbounds %struct.S, ptr [[THIS]], i32 0, i32 0
// CHECK: [[PTR_ADDR:%.+]] = getelementptr inbounds %struct.S, ptr [[THIS]], i32 0, i32 1
// CHECK: [[REF_REF:%.+]] = getelementptr inbounds %struct.S, ptr [[THIS]], i32 0, i32 2
// CHECK: [[REF_PTR:%.+]] = load ptr, ptr [[REF_REF]],
// CHECK: [[ARR_ADDR2:%.+]] = getelementptr inbounds %struct.S, ptr [[THIS]], i32 0, i32 3
// CHECK: [[ARR_END:%.+]] = getelementptr [4 x i32], ptr [[ARR_ADDR]], i32 1
// CHECK: [[E:%.+]] = ptrtoint ptr [[ARR_END]] to i64
// CHECK: [[B:%.+]] = ptrtoint ptr [[A_ADDR]] to i64
// CHECK: [[DIFF:%.+]] = sub i64 [[E]], [[B]]
// CHECK: [[SZ:%.+]] = sdiv exact i64 [[DIFF]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: [[BPTR0:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BPTRS]], i32 0, i32 0
// CHECK: store ptr [[THIS]], ptr [[BPTR0]],
// CHECK: [[PTR0:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 0
// CHECK: store ptr [[A_ADDR]], ptr [[PTR0]],
// CHECK: [[SIZE0:%.+]] = getelementptr inbounds [5 x i64], ptr [[SIZES]], i32 0, i32 0
// CHECK: store i64 [[SZ]], ptr [[SIZE0]],
// CHECK: [[BPTR1:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BPTRS]], i32 0, i32 1
// CHECK: store ptr [[A_ADDR2]], ptr [[BPTR1]],
// CHECK: [[PTR1:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 1
// CHECK: store ptr [[A_ADDR2]], ptr [[PTR1]],
// CHECK: [[BPTR2:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BPTRS]], i32 0, i32 2
// CHECK: store ptr [[PTR_ADDR]], ptr [[BPTR2]],
// CHECK: [[PTR2:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 2
// CHECK: store ptr [[PTR_ADDR]], ptr [[PTR2]],
// CHECK: [[BPTR3:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BPTRS]], i32 0, i32 3
// CHECK: store ptr [[REF_PTR]], ptr [[BPTR3]],
// CHECK: [[PTR3:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 3
// CHECK: store ptr [[REF_PTR]], ptr [[PTR3]],
// CHECK: [[BPTR4:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BPTRS]], i32 0, i32 4
// CHECK: store ptr [[ARR_ADDR2]], ptr [[BPTR4]],
// CHECK: [[PTR4:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 4
// CHECK: store ptr [[ARR_ADDR2]], ptr [[PTR4]],
// CHECK: [[BPTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BPTRS]], i32 0, i32 0
// CHECK: [[PTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 0
// CHECK: [[SIZE:%.+]] = getelementptr inbounds [5 x i64], ptr [[SIZES]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 5, ptr [[BPTR]], ptr [[PTR]], ptr [[SIZE]], ptr [[MAPTYPES2]], ptr null, ptr null)
// CHECK: [[A_ADDR:%.+]] = load ptr, ptr [[BPTR1]],
// CHECK: store ptr [[A_ADDR]], ptr [[A_REF:%.+]],
// CHECK: [[PTR_ADDR:%.+]] = load ptr, ptr [[BPTR2]],
// CHECK: store ptr [[PTR_ADDR]], ptr [[PTR_REF:%.+]],
// CHECK: [[REF_PTR:%.+]] = load ptr, ptr [[BPTR3]],
// CHECK: store ptr [[REF_PTR]], ptr [[REF_REF:%.+]],
// CHECK: [[PTR_ADDR:%.+]] = load ptr, ptr [[BPTR2]],
// CHECK: store ptr [[PTR_ADDR]], ptr [[PTR_REF2:%.+]],
// CHECK: [[ARR_ADDR:%.+]] = load ptr, ptr [[BPTR4]],
// CHECK: store ptr [[ARR_ADDR]], ptr [[ARR_REF:%.+]],
// CHECK: [[A_ADDR:%.+]] = load ptr, ptr [[A_REF]],
// CHECK: [[A:%.+]] = load i32, ptr [[A_ADDR]],
// CHECK: [[INC:%.+]] = add nsw i32 [[A]], 1
// CHECK: store i32 [[INC]], ptr [[A_ADDR]],
// CHECK: [[PTR_PTR:%.+]] = load ptr, ptr [[PTR_REF2]],
// CHECK: [[PTR:%.+]] = load ptr, ptr [[PTR_PTR]],
// CHECK: [[VAL:%.+]] = load i32, ptr [[PTR]],
// CHECK: [[INC:%.+]] = add nsw i32 [[VAL]], 1
// CHECK: store i32 [[INC]], ptr [[PTR]],
// CHECK: [[REF_PTR:%.+]] = load ptr, ptr [[REF_REF]],
// CHECK: [[VAL:%.+]] = load i32, ptr [[REF_PTR]],
// CHECK: [[INC:%.+]] = add nsw i32 [[VAL]], 1
// CHECK: store i32 [[INC]], ptr [[REF_PTR]],
// CHECK: [[ARR_ADDR:%.+]] = load ptr, ptr [[ARR_REF]],
// CHECK: [[ARR0_ADDR:%.+]] = getelementptr inbounds [4 x i32], ptr [[ARR_ADDR]], i64 0, i64 0
// CHECK: [[VAL:%.+]] = load i32, ptr [[ARR0_ADDR]],
// CHECK: [[INC:%.+]] = add nsw i32 [[VAL]], 1
// CHECK: store i32 [[INC]], ptr [[ARR0_ADDR]],
// CHECK: [[BPTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BPTRS]], i32 0, i32 0
// CHECK: [[PTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 0
// CHECK: [[SIZE:%.+]] = getelementptr inbounds [5 x i64], ptr [[SIZES]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 5, ptr [[BPTR]], ptr [[PTR]], ptr [[SIZE]], ptr [[MAPTYPES2]], ptr null, ptr null)

#endif
