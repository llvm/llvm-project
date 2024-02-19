// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
//
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-LABEL: @main
int main() {
  float *p;
  int a = 10;
  // kmp_task_affinity_info_t affs[1];
  // CHECK: [[AFFS_ADDR:%.+]] = alloca [1 x %struct.kmp_task_affinity_info_t],
  // CHECK: [[TD:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 [[GTID:%.+]], i32 1, i64 40, i64 1, ptr @{{.+}})
  // CHECK: [[AFFINE_LST_ADDR:%.+]] = getelementptr inbounds [1 x %struct.kmp_task_affinity_info_t], ptr [[AFFS_ADDR]], i64 0, i64 0
  // CHECK: [[P:%.+]] = load ptr, ptr [[P_ADDR:%.+]],
  // CHECK: [[A_VAL:%.+]] = load i32, ptr [[A_ADDR:%.+]],
  // CHECK: [[A_SZ:%.+]] = sext i32 [[A_VAL]] to i64
  // CHECK: [[BYTES:%.+]] = mul nuw i64 4, [[A_SZ]]
  // CHECK: [[SZ:%.+]] = mul nuw i64 [[BYTES]], 10
  // CHECK: [[A_VAL:%.+]] = load i32, ptr [[A_ADDR]],
  // CHECK: [[A_SZ1:%.+]] = sext i32 [[A_VAL]] to i64
  // CHECK: [[SIZE:%.+]] = mul nuw i64 [[SZ]], [[A_SZ]]
  // CHECK: [[AFFS_0_ADDR:%.+]] = getelementptr %struct.kmp_task_affinity_info_t, ptr [[AFFINE_LST_ADDR]], i64 0

  // affs[0].base = p;
  // CHECK: [[AFFS_0_BASE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_task_affinity_info_t, ptr [[AFFS_0_ADDR]], i32 0, i32 0
  // CHECK: [[P_INTPTR:%.+]] = ptrtoint ptr [[P]] to i64
  // CHECK: store i64 [[P_INTPTR]], ptr [[AFFS_0_BASE_ADDR]],

  // affs[0].size = sizeof(*p) * a * 10 * a;
  // CHECK: [[AFFS_0_SIZE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_task_affinity_info_t, ptr [[AFFS_0_ADDR]], i32 0, i32 1
  // CHECK: store i64 [[SIZE]], ptr [[AFFS_0_SIZE_ADDR]],
  // CHECK: call i32 @__kmpc_omp_reg_task_with_affinity(ptr @{{.+}}, i32 [[GTID]], ptr [[TD]], i32 1, ptr [[AFFINE_LST_ADDR]])
#pragma omp task affinity(([a][10][a])p)
  ;
  // CHECK: [[TD:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 [[GTID]], i32 1, i64 40, i64 1, ptr @{{.+}})
  // CHECK: [[A_VAL:%.+]] = load i32, ptr [[A_ADDR]],
  // CHECK: [[SUB:%.+]] = sub nsw i32 [[A_VAL]], 0
  // CHECK: [[CONV:%.+]] = zext i32 [[SUB]] to i64

  // <num_elem> = <num_iters> + 1 constant affinity for affinity(a)
  // CHECK: [[NUM_ELEMS:%.+]] = add nuw i64 1, [[CONV]]
  // CHECK: [[SV:%.+]] = call ptr @llvm.stacksave.p0()
  // CHECK: store ptr [[SV]], ptr [[SV_ADDR:%.+]],

  // kmp_task_affinity_info_t affs[<num_elem>];
  // CHECK: [[AFFS_ADDR:%.+]] = alloca %struct.kmp_task_affinity_info_t, i64 [[NUM_ELEMS]],
  // store i64 %21, ptr %__vla_expr0, align 8
  // CHECK: [[NAFFS:%.+]] = trunc i64 [[NUM_ELEMS]] to i32
  // CHECK: [[AFFS_0_ADDR:%.+]] = getelementptr %struct.kmp_task_affinity_info_t, ptr [[AFFS_ADDR]], i64 0

  // affs[0].base = &a;
  // CHECK: [[AFFS_0_BASE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_task_affinity_info_t, ptr [[AFFS_0_ADDR]], i32 0, i32 0
  // CHECK: [[A_INTPTR:%.+]] = ptrtoint ptr [[A_ADDR]] to i64
  // CHECK: store i64 [[A_INTPTR]], ptr [[AFFS_0_BASE_ADDR]],

  // affs[0].size = sizeof(a);
  // CHECK: [[AFFS_0_SIZE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_task_affinity_info_t, ptr [[AFFS_0_ADDR]], i32 0, i32 1
  // CHECK: store i64 4, ptr [[AFFS_0_SIZE_ADDR]],

  // affs_cnt = 1;
  // CHECK: store i64 1, ptr [[AFFS_CNT_ADDR:%.+]],
  // CHECK: [[A_VAL:%.+]] = load i32, ptr [[A_ADDR]],
  // CHECK: [[NITERS:%.+]] = sub nsw i32 [[A_VAL]], 0
  // CHECK: store i32 0, ptr [[CNT_ADDR:%.+]],
  // CHECK: br label %[[CONT:[^,]+]]

  //for (int cnt = 0; cnt < (a-0); ++cnt) {
  // int i = cnt + 0;
  // affs[affs_cnt].base = &p[i];
  // affs[affs_cnt].size = sizeof(p[i]);
  // ++affs_cnt;
  // }

  // CHECK: [[CONT]]:
  // CHECK: [[CNT:%.+]] = load i32, ptr [[CNT_ADDR]],
  // CHECK: [[CMP:%.+]] = icmp slt i32 [[CNT]], [[NITERS]]
  // CHECK: br i1 [[CMP]], label %[[BODY:[^,]+]], label %[[DONE:[^,]+]]

  // CHECK: [[BODY]]:
  // i = cnt + 0;
  // CHECK: [[CNT:%.+]] = load i32, ptr [[CNT_ADDR]],
  // CHECK: [[VAL:%.+]] = add nsw i32 0, [[CNT]]
  // CHECK: store i32 [[VAL]], ptr [[I_ADDR:%.+]],

  // &p[i]
  // CHECK: [[P:%.+]] = load ptr, ptr [[P_ADDR]],
  // CHECK: [[I:%.+]] = load i32, ptr [[I_ADDR]],
  // CHECK: [[IDX:%.+]] = sext i32 [[I]] to i64
  // CHECK: [[P_I_ADDR:%.+]] = getelementptr inbounds float, ptr [[P]], i64 [[IDX]]

  // affs[affs_cnt]
  // CHECK: [[AFFS_CNT:%.+]] = load i64, ptr [[AFFS_CNT_ADDR]],
  // CHECK: [[AFFS_ELEM_ADDR:%.+]] = getelementptr %struct.kmp_task_affinity_info_t, ptr [[AFFS_ADDR]], i64 [[AFFS_CNT]]

  // affs[affs_cnt].base = &p[i];
  // CHECK: [[AFFS_ELEM_BASE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_task_affinity_info_t, ptr [[AFFS_ELEM_ADDR]], i32 0, i32 0
  // CHECK: [[CAST:%.+]] = ptrtoint ptr [[P_I_ADDR]] to i64
  // CHECK: store i64 [[CAST]], ptr [[AFFS_ELEM_BASE_ADDR]],

  // affs[affs_cnt].size = sizeof(p[i]);
  // CHECK: [[AFFS_ELEM_SIZE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_task_affinity_info_t, ptr [[AFFS_ELEM_ADDR]], i32 0, i32 1
  // CHECK: store i64 4, ptr [[AFFS_ELEM_SIZE_ADDR]],

  // ++affs_cnt;
  // CHECK: [[AFFS_CNT_NEXT:%.+]] = add nuw i64 [[AFFS_CNT]], 1
  // CHECK: store i64 [[AFFS_CNT_NEXT]], ptr [[AFFS_CNT_ADDR]],

  // ++cnt;
  // CHECK: [[CNT:%.+]] = load i32, ptr [[CNT_ADDR]],
  // CHECK: [[CNT_NEXT:%.+]] = add nsw i32 [[CNT]], 1
  // CHECK: store i32 [[CNT_NEXT]], ptr [[CNT_ADDR]],
  // CHECK: br label %[[CONT]]

  // CHECK: [[DONE]]:
  // CHECK: call i32 @__kmpc_omp_reg_task_with_affinity(ptr @{{.+}} i32 [[GTID]], ptr [[TD]], i32 [[NAFFS]], ptr [[AFFS_ADDR]])
  // CHECK: [[SV:%.+]] = load ptr, ptr [[SV_ADDR]],
  // CHECK: call void @llvm.stackrestore.p0(ptr [[SV]])
#pragma omp task affinity(iterator(i=0:a): p[i]) affinity(a)
  ;
  return 0;
}

#endif
