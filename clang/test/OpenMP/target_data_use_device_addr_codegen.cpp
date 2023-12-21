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

// CHECK-DAG: [[SIZES1:@.+]] = private unnamed_addr constant [6 x i64] [i64 4, i64 16, i64 4, i64 4, i64 0, i64 4]
// 64 = 0x40 = OMP_MAP_RETURN_PARAM
// CHECK-DAG: [[MAPTYPES1:@.+]] = private unnamed_addr constant [6 x i64] [i64 67, i64 67, i64 3, i64 67, i64 67, i64 67]
// CHECK-DAG: [[SIZES2:@.+]] = private unnamed_addr constant [6 x i64] [i64 0, i64 4, i64 16, i64 4, i64 4, i64 0]
// 0 = OMP_MAP_NONE
// 281474976710720 = 0x1000000000040 = OMP_MAP_MEMBER_OF | OMP_MAP_RETURN_PARAM
// CHECK-DAG: [[MAPTYPES2:@.+]] = private unnamed_addr constant [6 x i64] [i64 0, i64 281474976710723, i64 281474976710739, i64 281474976710739, i64 281474976710675, i64 281474976710723]
struct S {
  int a = 0;
  int *ptr = &a;
  int &ref = a;
  int arr[4];
  S() {}
  void foo() {
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
#pragma omp target data map(tofrom: a, ptr [3:4], ref, ptr[0], arr[:(int)a], vla[0]) use_device_addr(a, ptr [3:4], ref, ptr[0], arr[:(int)a], vla[0])
  ++a, ++*ptr, ++ref, ++arr[0], ++vla[0];
  return a;
}

// CHECK-LABEL: @main()
// CHECK: [[A_ADDR:%.+]] = alloca float,
// CHECK: [[PTR_ADDR:%.+]] = alloca ptr,
// CHECK: [[REF_ADDR:%.+]] = alloca ptr,
// CHECK: [[ARR_ADDR:%.+]] = alloca [4 x float],
// CHECK: [[BPTRS:%.+]] = alloca [6 x ptr],
// CHECK: [[PTRS:%.+]] = alloca [6 x ptr],
// CHECK: [[MAP_PTRS:%.+]] = alloca [6 x ptr],
// CHECK: [[SIZES:%.+]] = alloca [6 x i64],
// CHECK: [[VLA_ADDR:%.+]] = alloca float, i64 %{{.+}},
// CHECK: [[PTR:%.+]] = load ptr, ptr [[PTR_ADDR]],
// CHECK-NEXT: [[P4:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8
// CHECK-NEXT: [[ARR_IDX:%.+]] = getelementptr inbounds float, ptr [[P4]], i64 3
// CHECK: [[P5:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8
// CHECK-NEXT: [[P6:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8
// CHECK-NEXT: [[ARR_IDX1:%.+]] = getelementptr inbounds float, ptr [[P6]], i64 0
// CHECK: [[P7:%.+]] = load ptr, ptr [[REF_ADDR]],
// CHECK-NEXT: [[REF:%.+]] = load ptr, ptr [[REF_ADDR]],
// CHECK-NEXT: [[ARR_IDX2:%.+]] = getelementptr inbounds [4 x float], ptr [[ARR_ADDR]], i64 0, i64 0
// CHECK: [[P10:%.+]] = mul nuw i64 {{.+}}, 4
// CHECK-NEXT: [[ARR_IDX5:%.+]] = getelementptr inbounds float, ptr [[VLA_ADDR]], i64 0
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[SIZES]], ptr align 8 [[SIZES1]], i64 48, i1 false)
// CHECK: [[BPTR0:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 0
// CHECK: store ptr [[A_ADDR]], ptr [[BPTR0]],
// CHECK: [[PTR0:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 0
// CHECK: store ptr [[A_ADDR]], ptr [[PTR0]],
// CHECK: [[BPTR1:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 1
// CHECK: store ptr [[PTR]], ptr [[BPTR1]],
// CHECK: [[PTR1:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 1
// CHECK: store ptr [[ARR_IDX]], ptr [[PTR1]],
// CHECK: [[BPTR2:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 2
// CHECK: store ptr [[P5]], ptr [[BPTR2]],
// CHECK: [[PTR2:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 2
// CHECK: store ptr [[ARR_IDX1]], ptr [[PTR2]],
// CHECK: [[BPTR3:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 3
// CHECK: store ptr [[P7]], ptr [[BPTR3]],
// CHECK: [[PTR3:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 3
// CHECK: store ptr [[REF]], ptr [[PTR3]],
// CHECK: [[BPTR4:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 4
// CHECK: store ptr [[ARR_ADDR]], ptr [[BPTR4]], align 
// CHECK: [[PTR4:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 4
// CHECK: store ptr [[ARR_IDX2]], ptr [[PTR4]], align 8
// CHECK: [[SIZE_PTR:%.+]] = getelementptr inbounds [6 x i64], ptr [[SIZES]], i32 0, i32 4
// CHECK: store i64 [[P10:%.+]], ptr [[SIZE_PTR]], align 8
// CHECK: [[MAP_PTR:%.+]] = getelementptr inbounds [6 x ptr], ptr [[MAP_PTRS]], i64 0, i64 4
// CHECK: store ptr null, ptr [[MAP_PTR]], align 8
// CHECK: [[BPTR5:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 5
// CHECK: store ptr [[VLA_ADDR]], ptr [[BPTR5]],
// CHECK: [[PTR5:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 5
// CHECK: store ptr [[ARR_IDX5]], ptr [[PTR5]],

// CHECK: [[BPTR:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 0
// CHECK: [[PTR:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 0
// CHECK: [[SIZE:%.+]] = getelementptr inbounds [6 x i64], ptr [[SIZES]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 6, ptr [[BPTR]], ptr [[PTR]], ptr [[SIZE]], ptr [[MAPTYPES1]], ptr null, ptr null)
// CHECK: [[A_REF:%.+]] = load ptr, ptr [[BPTR0]],
// CHECK: [[REF_REF:%.+]] = load ptr, ptr [[BPTR3]],
// CHECK: store ptr [[REF_REF]], ptr [[TMP_REF_ADDR:%.+]],
// CHECK: [[ARR_REF:%.+]] = load ptr, ptr [[BPTR4]],
// CHECK: [[VLA_REF:%.+]] = load ptr, ptr [[BPTR5]],
// CHECK: [[A:%.+]] = load float, ptr [[A_REF]],
// CHECK: [[INC:%.+]] = fadd float [[A]], 1.000000e+00
// CHECK: store float [[INC]], ptr [[A_REF]],
// CHECK: [[PTR:%.+]] = load ptr, ptr [[BPTR1]],
// CHECK: [[VAL:%.+]] = load float, ptr [[PTR]],
// CHECK: [[INC:%.+]] = fadd float [[VAL]], 1.000000e+00
// CHECK: store float [[INC]], ptr [[PTR]],
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
// CHECK: [[BPTR:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 0
// CHECK: [[PTR:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 0
// CHECK: [[SIZE:%.+]] = getelementptr inbounds [6 x i64], ptr [[SIZES]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 6, ptr [[BPTR]], ptr [[PTR]], ptr [[SIZE]], ptr [[MAPTYPES1]], ptr null, ptr null)

// CHECK: foo
// CHECK: [[BPTRS:%.+]] = alloca [6 x ptr],
// CHECK: [[PTRS:%.+]] = alloca [6 x ptr],
// CHECK: [[MAP_PTRS:%.+]] = alloca [6 x ptr],
// CHECK: [[SIZES:%.+]] = alloca [6 x i64],
// CHECK: [[A_ADDR:%.+]] = getelementptr inbounds %struct.S, ptr [[THIS:%.+]], i32 0, i32 0
// CHECK: [[PTR_ADDR:%.+]] = getelementptr inbounds %struct.S, ptr [[THIS]], i32 0, i32 1
// CHECK: [[ARR_IDX:%.+]] = getelementptr inbounds i32, ptr %{{.+}}, i64 3
// CHECK: [[REF_REF:%.+]] = getelementptr inbounds %struct.S, ptr [[THIS]], i32 0, i32 2
// CHECK: [[REF_PTR:%.+]] = load ptr, ptr [[REF_REF]],
// CHECK-NEXT: [[P3:%.+]] = getelementptr inbounds %struct.S, ptr [[THIS]], i32 0, i32 1
// CHECK: [[ARR_IDX5:%.+]] = getelementptr inbounds i32, ptr {{.+}}, i64 0
// CHECK: [[ARR_ADDR:%.+]] = getelementptr inbounds %struct.S, ptr [[THIS]], i32 0, i32 3

// CHECK: [[ARR_IDX6:%.+]] = getelementptr inbounds [4 x i32], ptr [[ARR_ADDR]], i64 0, i64 0
// CHECK: [[A_ADDR2:%.+]] = getelementptr inbounds %struct.S, ptr [[THIS]], i32 0, i32 0
// CHECK: [[P4:%.+]] = mul nuw i64 [[CONV:%.+]], 4
// CHECK: [[A_ADDR3:%.+]] = getelementptr inbounds %struct.S, ptr [[THIS]], i32 0, i32 0
// CHECK: [[L5:%.+]] = load i32, ptr [[A_ADDR3]]
// CHECK: [[L6:%.+]] = sext i32 [[L5]] to i64
// CHECK: [[LB_ADD_LEN:%lb_add_len]] = add nsw i64 -1, [[L6]]
// CHECK: [[ARR_ADDR9:%.+]] = getelementptr inbounds %struct.S, ptr [[THIS]], i32 0, i32 3
// CHECK: [[ARR_IDX10:%arrayidx.+]] = getelementptr inbounds [4 x i32], ptr [[ARR_ADDR9]], i64 0, i64 %lb_add_len
// CHECK: [[ARR_END:%.+]] = getelementptr i32, ptr [[ARR_IDX10]], i32 1
// CHECK: [[E:%.+]] = ptrtoint ptr [[ARR_END]] to i64
// CHECK: [[B:%.+]] = ptrtoint ptr [[A_ADDR]] to i64
// CHECK: [[DIFF:%.+]] = sub i64 [[E]], [[B]]
// CHECK: [[SZ:%.+]] = sdiv exact i64 [[DIFF]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CHECK: [[BPTR0:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 0
// CHECK: store ptr [[THIS]], ptr [[BPTR0]],
// CHECK: [[PTR0:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 0
// CHECK: store ptr [[A_ADDR]], ptr [[PTR0]],
// CHECK: [[SIZE0:%.+]] = getelementptr inbounds [6 x i64], ptr [[SIZES]], i32 0, i32 0
// CHECK: store i64 [[SZ]], ptr [[SIZE0]],
// CHECK: [[BPTR1:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 1
// CHECK: store ptr [[THIS]], ptr [[BPTR1]]
// CHECK: [[PTR1:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 1
// CHECK: store ptr [[A_ADDR]], ptr [[PTR1]],
// CHECK: [[BPTR2:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 2
// CHECK: store ptr [[PTR_ADDR]], ptr [[BPTR2]],
// CHECK: [[PTR2:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 2
// CHECK: store ptr [[ARR_IDX]], ptr [[PTR2]],
// CHECK: [[BPTR3:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 3
// CHECK: store ptr [[THIS]], ptr [[BPTR3]]
// CHECK: [[PTR3:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 3
// CHECK: store ptr [[REF_PTR]], ptr [[PTR3]],
// CHECK: [[BPTR4:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 4
// CHECK: store ptr [[P3]], ptr [[BPTR4]],
// CHECK: [[PTR4:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 4
// CHECK: store ptr [[ARR_IDX5]], ptr [[PTR4]]

// CHECK: [[BPTR5:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 5
// CHECK: store ptr [[THIS]], ptr [[BPTR5]], align 8
// CHECK: [[PTR5:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 5
// CHECK: store ptr [[ARR_IDX6]], ptr [[PTR5]], align 8
// CHECK: [[SIZE1:%.+]] = getelementptr inbounds [6 x i64], ptr [[SIZES]], i32 0, i32 5
// CHECK: store i64 [[P4]], ptr [[SIZE1]], align 8
// CHECK: [[BPTR:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 0
// CHECK: [[PTR:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 0
// CHECK: [[SIZE:%.+]] = getelementptr inbounds [6 x i64], ptr [[SIZES]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 6, ptr [[BPTR]], ptr [[PTR]], ptr [[SIZE]], ptr [[MAPTYPES2]], ptr null, ptr null)
// CHECK: [[A_ADDR:%.+]] = load ptr, ptr [[BPTR1]],
// CHECK: store ptr [[A_ADDR]], ptr [[A_REF:%.+]],
// CHECK: [[PTR_ADDR:%.+]] = load ptr, ptr [[BPTR2]],
// CHECK: store ptr [[PTR_ADDR]], ptr [[PTR_REF:%.+]],
// CHECK: [[REF_PTR:%.+]] = load ptr, ptr [[BPTR3]],
// CHECK: store ptr [[REF_PTR]], ptr [[REF_REF:%.+]],
// CHECK: [[PTR_ADDR:%.+]] = load ptr, ptr [[BPTR2]],
// CHECK: store ptr [[PTR_ADDR]], ptr [[PTR_REF2:%.+]],
// CHECK: [[ARR_ADDR:%.+]] = load ptr, ptr [[BPTR5]],
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
// CHECK: [[BPTR:%.+]] = getelementptr inbounds [6 x ptr], ptr [[BPTRS]], i32 0, i32 0
// CHECK: [[PTR:%.+]] = getelementptr inbounds [6 x ptr], ptr [[PTRS]], i32 0, i32 0
// CHECK: [[SIZE:%.+]] = getelementptr inbounds [6 x i64], ptr [[SIZES]], i32 0, i32 0
// CHECK: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 6, ptr [[BPTR]], ptr [[PTR]], ptr [[SIZE]], ptr [[MAPTYPES2]], ptr null, ptr null)

#endif
