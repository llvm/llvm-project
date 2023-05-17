// RUN: %clang_cc1 -triple x86_64-linux-gnu -target-cpu core2 %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple i686-linux-gnu -target-cpu core2 %s -S -emit-llvm -o - | FileCheck -check-prefix=CHECK32 %s

long double testinc(_Atomic long double *addr) {
  // CHECK-LABEL: @testinc
  // CHECK: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 8
  // CHECK: [[INT_VALUE:%.+]] = load atomic i128, ptr [[ADDR]] seq_cst, align 16
  // CHECK: store i128 [[INT_VALUE]], ptr [[LD_ADDR:%.+]], align 16
  // CHECK: [[LD_VALUE:%.+]] = load x86_fp80, ptr [[LD_ADDR]], align 16
  // CHECK: br label %[[ATOMIC_OP:.+]]
  // CHECK: [[ATOMIC_OP]]
  // CHECK: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK: [[INC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[OLD_VALUE_ADDR:%.+]], i8 0, i64 16, i1 false)
  // CHECK: store x86_fp80 [[OLD_VALUE]], ptr [[OLD_VALUE_ADDR]], align 16
  // CHECK: [[OLD_INT:%.+]] = load i128, ptr [[OLD_VALUE_ADDR]], align 16
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[NEW_VALUE_ADDR:%.+]], i8 0, i64 16, i1 false)
  // CHECK: store x86_fp80 [[INC_VALUE]], ptr [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[NEW_INT:%.+]] = load i128, ptr [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[RES:%.+]] = cmpxchg ptr [[ADDR]], i128 [[OLD_INT]], i128 [[NEW_INT]] seq_cst seq_cst, align 16
  // CHECK: [[OLD_VALUE:%.+]] = extractvalue { i128, i1 } [[RES]], 0
  // CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
  // CHECK: store i128 [[OLD_VALUE]], ptr [[OLD_VALUE_RES_PTR:%.+]], align 16
  // CHECK: [[LD_VALUE]] = load x86_fp80, ptr [[OLD_VALUE_RES_PTR]], align 16
  // CHECK: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK: [[ATOMIC_CONT]]
  // CHECK: ret x86_fp80 [[INC_VALUE]]
  // CHECK32-LABEL: @testinc
  // CHECK32: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 4
  // CHECK32: call void @__atomic_load(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[TEMP_LD_ADDR:%.+]], i32 noundef 5)
  // CHECK32: [[LD_VALUE:%.+]] = load x86_fp80, ptr [[TEMP_LD_ADDR]], align 4
  // CHECK32: br label %[[ATOMIC_OP:.+]]
  // CHECK32: [[ATOMIC_OP]]
  // CHECK32: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK32: [[INC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK32: call void @llvm.memset.p0.i64(ptr align 4 [[OLD_VALUE_ADDR:%.+]], i8 0, i64 12, i1 false)
  // CHECK32: store x86_fp80 [[OLD_VALUE]], ptr [[OLD_VALUE_ADDR]], align 4
  // CHECK32: call void @llvm.memset.p0.i64(ptr align 4 [[DESIRED_VALUE_ADDR:%.+]], i8 0, i64 12, i1 false)
  // CHECK32: store x86_fp80 [[INC_VALUE]], ptr [[DESIRED_VALUE_ADDR]], align 4
  // CHECK32: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[OLD_VALUE_ADDR]], ptr noundef [[DESIRED_VALUE_ADDR]], i32 noundef 5, i32 noundef 5)
  // CHECK32: [[LD_VALUE:%.+]] = load x86_fp80, ptr [[OLD_VALUE_ADDR]], align 4
  // CHECK32: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK32: [[ATOMIC_CONT]]
  // CHECK32: ret x86_fp80 [[INC_VALUE]]

  return ++*addr;
}

long double testdec(_Atomic long double *addr) {
  // CHECK-LABEL: @testdec
  // CHECK: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 8
  // CHECK: [[INT_VALUE:%.+]] = load atomic i128, ptr [[ADDR]] seq_cst, align 16
  // CHECK: store i128 [[INT_VALUE]], ptr [[LD_ADDR:%.+]], align 16
  // CHECK: [[ORIG_LD_VALUE:%.+]] = load x86_fp80, ptr [[LD_ADDR]], align 16
  // CHECK: br label %[[ATOMIC_OP:.+]]
  // CHECK: [[ATOMIC_OP]]
  // CHECK: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[ORIG_LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK: [[DEC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[OLD_VALUE_ADDR:%.+]], i8 0, i64 16, i1 false)
  // CHECK: store x86_fp80 [[OLD_VALUE]], ptr [[OLD_VALUE_ADDR]], align 16
  // CHECK: [[OLD_INT:%.+]] = load i128, ptr [[OLD_VALUE_ADDR]], align 16
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[NEW_VALUE_ADDR:%.+]], i8 0, i64 16, i1 false)
  // CHECK: store x86_fp80 [[DEC_VALUE]], ptr [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[NEW_INT:%.+]] = load i128, ptr [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[RES:%.+]] = cmpxchg ptr [[ADDR]], i128 [[OLD_INT]], i128 [[NEW_INT]] seq_cst seq_cst, align 16
  // CHECK: [[OLD_VALUE:%.+]] = extractvalue { i128, i1 } [[RES]], 0
  // CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
  // CHECK: store i128 [[OLD_VALUE]], ptr [[OLD_VALUE_RES_PTR:%.+]], align 16
  // CHECK: [[LD_VALUE]] = load x86_fp80, ptr [[OLD_VALUE_RES_PTR]], align 16
  // CHECK: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK: [[ATOMIC_CONT]]
  // CHECK: ret x86_fp80 [[ORIG_LD_VALUE]]
  // CHECK32-LABEL: @testdec
  // CHECK32: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 4
  // CHECK32: call void @__atomic_load(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[TEMP_LD_ADDR:%.+]], i32 noundef 5)
  // CHECK32: [[ORIG_LD_VALUE:%.+]] = load x86_fp80, ptr [[TEMP_LD_ADDR]], align 4
  // CHECK32: br label %[[ATOMIC_OP:.+]]
  // CHECK32: [[ATOMIC_OP]]
  // CHECK32: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[ORIG_LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK32: [[DEC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK32: call void @llvm.memset.p0.i64(ptr align 4 [[OLD_VALUE_ADDR:%.+]], i8 0, i64 12, i1 false)
  // CHECK32: store x86_fp80 [[OLD_VALUE]], ptr [[OLD_VALUE_ADDR]], align 4
  // CHECK32: call void @llvm.memset.p0.i64(ptr align 4 [[DESIRED_VALUE_ADDR:%.+]], i8 0, i64 12, i1 false)
  // CHECK32: store x86_fp80 [[DEC_VALUE]], ptr [[DESIRED_VALUE_ADDR]], align 4
  // CHECK32: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[OLD_VALUE_ADDR]], ptr noundef [[DESIRED_VALUE_ADDR]], i32 noundef 5, i32 noundef 5)
  // CHECK32: [[LD_VALUE]] = load x86_fp80, ptr [[OLD_VALUE_ADDR]], align 4
  // CHECK32: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK32: [[ATOMIC_CONT]]
  // CHECK32: ret x86_fp80 [[ORIG_LD_VALUE]]

  return (*addr)--;
}

long double testcompassign(_Atomic long double *addr) {
  *addr -= 25;
  // CHECK-LABEL: @testcompassign
  // CHECK: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 8
  // CHECK: [[INT_VALUE:%.+]] = load atomic i128, ptr [[ADDR]] seq_cst, align 16
  // CHECK: store i128 [[INT_VALUE]], ptr [[LD_ADDR:%.+]], align 16
  // CHECK: [[LD_VALUE:%.+]] = load x86_fp80, ptr [[LD_ADDR]], align 16
  // CHECK: br label %[[ATOMIC_OP:.+]]
  // CHECK: [[ATOMIC_OP]]
  // CHECK: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK: [[SUB_VALUE:%.+]] = fsub x86_fp80 [[OLD_VALUE]],
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[OLD_VALUE_ADDR:%.+]], i8 0, i64 16, i1 false)
  // CHECK: store x86_fp80 [[OLD_VALUE]], ptr [[OLD_VALUE_ADDR]], align 16
  // CHECK: [[OLD_INT:%.+]] = load i128, ptr [[OLD_VALUE_ADDR]], align 16
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[NEW_VALUE_ADDR:%.+]], i8 0, i64 16, i1 false)
  // CHECK: store x86_fp80 [[SUB_VALUE]], ptr [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[NEW_INT:%.+]] = load i128, ptr [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[RES:%.+]] = cmpxchg ptr [[ADDR]], i128 [[OLD_INT]], i128 [[NEW_INT]] seq_cst seq_cst, align 16
  // CHECK: [[OLD_VALUE:%.+]] = extractvalue { i128, i1 } [[RES]], 0
  // CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
  // CHECK: store i128 [[OLD_VALUE]], ptr [[OLD_VALUE_RES_PTR:%.+]], align 16
  // CHECK: [[LD_VALUE]] = load x86_fp80, ptr [[OLD_VALUE_RES_PTR]], align 16
  // CHECK: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK: [[ATOMIC_CONT]]
  // CHECK: [[ADDR:%.+]] = load ptr, ptr %{{.+}}, align 8
  // CHECK: [[INT_VAL:%.+]] = load atomic i128, ptr [[ADDR]] seq_cst, align 16
  // CHECK: store i128 [[INT_VAL]], ptr [[INT_LD_TEMP:%.+]], align 16
  // CHECK: [[RET_VAL:%.+]] = load x86_fp80, ptr [[LD_TEMP:%.+]], align 16
  // CHECK: ret x86_fp80 [[RET_VAL]]
  // CHECK32-LABEL: @testcompassign
  // CHECK32: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 4
  // CHECK32: call void @__atomic_load(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[TEMP_LD_ADDR:%.+]], i32 noundef 5)
  // CHECK32: [[LD_VALUE:%.+]] = load x86_fp80, ptr [[TEMP_LD_ADDR]], align 4
  // CHECK32: br label %[[ATOMIC_OP:.+]]
  // CHECK32: [[ATOMIC_OP]]
  // CHECK32: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK32: [[INC_VALUE:%.+]] = fsub x86_fp80 [[OLD_VALUE]],
  // CHECK32: call void @llvm.memset.p0.i64(ptr align 4 [[OLD_VALUE_ADDR:%.+]], i8 0, i64 12, i1 false)
  // CHECK32: store x86_fp80 [[OLD_VALUE]], ptr [[OLD_VALUE_ADDR]], align 4
  // CHECK32: call void @llvm.memset.p0.i64(ptr align 4 [[DESIRED_VALUE_ADDR:%.+]], i8 0, i64 12, i1 false)
  // CHECK32: store x86_fp80 [[INC_VALUE]], ptr [[DESIRED_VALUE_ADDR]], align 4
  // CHECK32: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[OLD_VALUE_ADDR]], ptr noundef [[DESIRED_VALUE_ADDR]], i32 noundef 5, i32 noundef 5)
  // CHECK32: [[LD_VALUE]] = load x86_fp80, ptr [[OLD_VALUE_ADDR]], align 4
  // CHECK32: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK32: [[ATOMIC_CONT]]
  // CHECK32: [[ADDR:%.+]] = load ptr, ptr %{{.+}}, align 4
  // CHECK32: call void @__atomic_load(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[GET_ADDR:%.+]], i32 noundef 5)
  // CHECK32: [[RET_VAL:%.+]] = load x86_fp80, ptr [[GET_ADDR]], align 4
  // CHECK32: ret x86_fp80 [[RET_VAL]]
  return *addr;
}

long double testassign(_Atomic long double *addr) {
  // CHECK-LABEL: @testassign
  // CHECK: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 8
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[STORE_TEMP_PTR:%.+]], i8 0, i64 16, i1 false)
  // CHECK: store x86_fp80 {{.+}}, ptr [[STORE_TEMP_PTR]], align 16
  // CHECK: [[STORE_TEMP_INT:%.+]] = load i128, ptr [[STORE_TEMP_PTR]], align 16
  // CHECK: store atomic i128 [[STORE_TEMP_INT]], ptr [[ADDR]] seq_cst, align 16
  // CHECK32-LABEL: @testassign
  // CHECK32: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 4
  // CHECK32: call void @llvm.memset.p0.i64(ptr align 4 [[STORE_TEMP_PTR:%.+]], i8 0, i64 12, i1 false)
  // CHECK32: store x86_fp80 {{.+}}, ptr [[STORE_TEMP_PTR]], align 4
  // CHECK32: call void @__atomic_store(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[STORE_TEMP_PTR]], i32 noundef 5)
  *addr = 115;
  // CHECK: [[ADDR:%.+]] = load ptr, ptr %{{.+}}, align 8
  // CHECK: [[INT_VAL:%.+]] = load atomic i128, ptr [[ADDR]] seq_cst, align 16
  // CHECK: store i128 [[INT_VAL]], ptr [[INT_LD_TEMP:%.+]], align 16
  // CHECK: [[RET_VAL:%.+]] = load x86_fp80, ptr [[LD_TEMP:%.+]], align 16
  // CHECK: ret x86_fp80 [[RET_VAL]]
  // CHECK32: [[ADDR:%.+]] = load ptr, ptr %{{.+}}, align 4
  // CHECK32: call void @__atomic_load(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[LD_TEMP:%.+]], i32 noundef 5)
  // CHECK32: [[RET_VAL:%.+]] = load x86_fp80, ptr [[LD_TEMP]], align 4
  // CHECK32: ret x86_fp80 [[RET_VAL]]

  return *addr;
}

long double test_volatile_inc(volatile _Atomic long double *addr) {
  // CHECK-LABEL: @test_volatile_inc
  // CHECK: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 8
  // CHECK: [[INT_VALUE:%.+]] = load atomic volatile i128, ptr [[ADDR]] seq_cst, align 16
  // CHECK: store i128 [[INT_VALUE]], ptr [[LD_ADDR:%.+]], align 16
  // CHECK: [[LD_VALUE:%.+]] = load x86_fp80, ptr [[LD_ADDR]], align 16
  // CHECK: br label %[[ATOMIC_OP:.+]]
  // CHECK: [[ATOMIC_OP]]
  // CHECK: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK: [[INC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[OLD_VALUE_ADDR:%.+]], i8 0, i64 16, i1 false)
  // CHECK: store x86_fp80 [[OLD_VALUE]], ptr [[OLD_VALUE_ADDR]], align 16
  // CHECK: [[OLD_INT:%.+]] = load i128, ptr [[OLD_VALUE_ADDR]], align 16
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[NEW_VALUE_ADDR:%.+]], i8 0, i64 16, i1 false)
  // CHECK: store x86_fp80 [[INC_VALUE]], ptr [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[NEW_INT:%.+]] = load i128, ptr [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[RES:%.+]] = cmpxchg volatile ptr [[ADDR]], i128 [[OLD_INT]], i128 [[NEW_INT]] seq_cst seq_cst, align 16
  // CHECK: [[OLD_VALUE:%.+]] = extractvalue { i128, i1 } [[RES]], 0
  // CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
  // CHECK: store i128 [[OLD_VALUE]], ptr [[OLD_VALUE_RES_PTR:%.+]], align 16
  // CHECK: [[LD_VALUE]] = load x86_fp80, ptr [[OLD_VALUE_RES_PTR]], align 16
  // CHECK: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK: [[ATOMIC_CONT]]
  // CHECK: ret x86_fp80 [[INC_VALUE]]
  // CHECK32-LABEL: @test_volatile_inc
  // CHECK32: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 4
  // CHECK32: call void @__atomic_load(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[TEMP_LD_ADDR:%.+]], i32 noundef 5)
  // CHECK32: [[LD_VALUE:%.+]] = load x86_fp80, ptr [[TEMP_LD_ADDR]], align 4
  // CHECK32: br label %[[ATOMIC_OP:.+]]
  // CHECK32: [[ATOMIC_OP]]
  // CHECK32: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK32: [[INC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK32: call void @llvm.memset.p0.i64(ptr align 4 [[OLD_VALUE_ADDR:%.+]], i8 0, i64 12, i1 false)
  // CHECK32: store x86_fp80 [[OLD_VALUE]], ptr [[OLD_VALUE_ADDR]], align 4
  // CHECK32: call void @llvm.memset.p0.i64(ptr align 4 [[DESIRED_VALUE_ADDR:%.+]], i8 0, i64 12, i1 false)
  // CHECK32: store x86_fp80 [[INC_VALUE]], ptr [[DESIRED_VALUE_ADDR]], align 4
  // CHECK32: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[OLD_VALUE_ADDR]], ptr noundef [[DESIRED_VALUE_ADDR]], i32 noundef 5, i32 noundef 5)
  // CHECK32: [[LD_VALUE]] = load x86_fp80, ptr [[OLD_VALUE_ADDR]], align 4
  // CHECK32: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK32: [[ATOMIC_CONT]]
  // CHECK32: ret x86_fp80 [[INC_VALUE]]
  return ++*addr;
}

long double test_volatile_dec(volatile _Atomic long double *addr) {
  // CHECK-LABEL: @test_volatile_dec
  // CHECK: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 8
  // CHECK: [[INT_VALUE:%.+]] = load atomic volatile i128, ptr [[ADDR]] seq_cst, align 16
  // CHECK: store i128 [[INT_VALUE]], ptr [[LD_ADDR:%.+]], align 16
  // CHECK: [[ORIG_LD_VALUE:%.+]] = load x86_fp80, ptr [[LD_ADDR]], align 16
  // CHECK: br label %[[ATOMIC_OP:.+]]
  // CHECK: [[ATOMIC_OP]]
  // CHECK: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[ORIG_LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK: [[DEC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[OLD_VALUE_ADDR:%.+]], i8 0, i64 16, i1 false)
  // CHECK: store x86_fp80 [[OLD_VALUE]], ptr [[OLD_VALUE_ADDR]], align 16
  // CHECK: [[OLD_INT:%.+]] = load i128, ptr [[OLD_VALUE_ADDR]], align 16
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[NEW_VALUE_ADDR:%.+]], i8 0, i64 16, i1 false)
  // CHECK: store x86_fp80 [[DEC_VALUE]], ptr [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[NEW_INT:%.+]] = load i128, ptr [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[RES:%.+]] = cmpxchg volatile ptr [[ADDR]], i128 [[OLD_INT]], i128 [[NEW_INT]] seq_cst seq_cst, align 16
  // CHECK: [[OLD_VALUE:%.+]] = extractvalue { i128, i1 } [[RES]], 0
  // CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
  // CHECK: store i128 [[OLD_VALUE]], ptr [[OLD_VALUE_RES_PTR:%.+]], align 16
  // CHECK: [[LD_VALUE]] = load x86_fp80, ptr [[OLD_VALUE_RES_PTR]], align 16
  // CHECK: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK: [[ATOMIC_CONT]]
  // CHECK: ret x86_fp80 [[ORIG_LD_VALUE]]
  // CHECK32-LABEL: @test_volatile_dec
  // CHECK32: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 4
  // CHECK32: call void @__atomic_load(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[TEMP_LD_ADDR:%.+]], i32 noundef 5)
  // CHECK32: [[ORIG_LD_VALUE:%.+]] = load x86_fp80, ptr [[TEMP_LD_ADDR]], align 4
  // CHECK32: br label %[[ATOMIC_OP:.+]]
  // CHECK32: [[ATOMIC_OP]]
  // CHECK32: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[ORIG_LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK32: [[DEC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK32: call void @llvm.memset.p0.i64(ptr align 4 [[OLD_VALUE_ADDR:%.+]], i8 0, i64 12, i1 false)
  // CHECK32: store x86_fp80 [[OLD_VALUE]], ptr [[OLD_VALUE_ADDR]], align 4
  // CHECK32: call void @llvm.memset.p0.i64(ptr align 4 [[DESIRED_VALUE_ADDR:%.+]], i8 0, i64 12, i1 false)
  // CHECK32: store x86_fp80 [[DEC_VALUE]], ptr [[DESIRED_VALUE_ADDR]], align 4
  // CHECK32: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[OLD_VALUE_ADDR]], ptr noundef [[DESIRED_VALUE_ADDR]], i32 noundef 5, i32 noundef 5)
  // CHECK32: [[LD_VALUE]] = load x86_fp80, ptr [[OLD_VALUE_ADDR]], align 4
  // CHECK32: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK32: [[ATOMIC_CONT]]
  // CHECK32: ret x86_fp80 [[ORIG_LD_VALUE]]
  return (*addr)--;
}

long double test_volatile_compassign(volatile _Atomic long double *addr) {
  *addr -= 25;
  // CHECK-LABEL: @test_volatile_compassign
  // CHECK: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 8
  // CHECK: [[INT_VALUE:%.+]] = load atomic volatile i128, ptr [[ADDR]] seq_cst, align 16
  // CHECK: store i128 [[INT_VALUE]], ptr [[LD_ADDR:%.+]], align 16
  // CHECK: [[LD_VALUE:%.+]] = load x86_fp80, ptr [[LD_ADDR]], align 16
  // CHECK: br label %[[ATOMIC_OP:.+]]
  // CHECK: [[ATOMIC_OP]]
  // CHECK: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK: [[SUB_VALUE:%.+]] = fsub x86_fp80 [[OLD_VALUE]],
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[OLD_VALUE_ADDR:%.+]], i8 0, i64 16, i1 false)
  // CHECK: store x86_fp80 [[OLD_VALUE]], ptr [[OLD_VALUE_ADDR]], align 16
  // CHECK: [[OLD_INT:%.+]] = load i128, ptr [[OLD_VALUE_ADDR]], align 16
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[NEW_VALUE_ADDR:%.+]], i8 0, i64 16, i1 false)
  // CHECK: store x86_fp80 [[SUB_VALUE]], ptr [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[NEW_INT:%.+]] = load i128, ptr [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[RES:%.+]] = cmpxchg volatile ptr [[ADDR]], i128 [[OLD_INT]], i128 [[NEW_INT]] seq_cst seq_cst, align 16
  // CHECK: [[OLD_VALUE:%.+]] = extractvalue { i128, i1 } [[RES]], 0
  // CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
  // CHECK: store i128 [[OLD_VALUE]], ptr [[OLD_VALUE_RES_PTR:%.+]], align 16
  // CHECK: [[LD_VALUE]] = load x86_fp80, ptr [[OLD_VALUE_RES_PTR]], align 16
  // CHECK: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK: [[ATOMIC_CONT]]
  // CHECK: [[ADDR:%.+]] = load ptr, ptr %{{.+}}, align 8
  // CHECK: [[INT_VAL:%.+]] = load atomic volatile i128, ptr [[ADDR]] seq_cst, align 16
  // CHECK: store i128 [[INT_VAL]], ptr [[INT_LD_TEMP:%.+]], align 16
  // CHECK: [[RET_VAL:%.+]] = load x86_fp80, ptr [[LD_TEMP:%.+]], align 16
  // CHECK32-LABEL: @test_volatile_compassign
  // CHECK32: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 4
  // CHECK32: call void @__atomic_load(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[TEMP_LD_ADDR:%.+]], i32 noundef 5)
  // CHECK32: [[LD_VALUE:%.+]] = load x86_fp80, ptr [[TEMP_LD_ADDR]], align 4
  // CHECK32: br label %[[ATOMIC_OP:.+]]
  // CHECK32: [[ATOMIC_OP]]
  // CHECK32: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK32: [[INC_VALUE:%.+]] = fsub x86_fp80 [[OLD_VALUE]],
  // CHECK32: call void @llvm.memset.p0.i64(ptr align 4 [[OLD_VALUE_ADDR:%.+]], i8 0, i64 12, i1 false)
  // CHECK32: store x86_fp80 [[OLD_VALUE]], ptr [[OLD_VALUE_ADDR]], align 4
  // CHECK32: call void @llvm.memset.p0.i64(ptr align 4 [[DESIRED_VALUE_ADDR:%.+]], i8 0, i64 12, i1 false)
  // CHECK32: store x86_fp80 [[INC_VALUE]], ptr [[DESIRED_VALUE_ADDR]], align 4
  // CHECK32: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[OLD_VALUE_ADDR]], ptr noundef [[DESIRED_VALUE_ADDR]], i32 noundef 5, i32 noundef 5)
  // CHECK32: [[LD_VALUE]] = load x86_fp80, ptr [[OLD_VALUE_ADDR]], align 4
  // CHECK32: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK32: [[ATOMIC_CONT]]
  // CHECK32: [[ADDR:%.+]] = load ptr, ptr %{{.+}}, align 4
  // CHECK32: call void @__atomic_load(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[GET_ADDR:%.+]], i32 noundef 5)
  // CHECK32: [[RET_VAL:%.+]] = load x86_fp80, ptr [[GET_ADDR]], align 4
  // CHECK32: ret x86_fp80 [[RET_VAL]]
  return *addr;
}

long double test_volatile_assign(volatile _Atomic long double *addr) {
  // CHECK-LABEL: @test_volatile_assign
  // CHECK: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 8
  // CHECK: call void @llvm.memset.p0.i64(ptr align 16 [[STORE_TEMP_PTR:%.+]], i8 0, i64 16, i1 false)
  // CHECK: store x86_fp80 {{.+}}, ptr [[STORE_TEMP_PTR]], align 16
  // CHECK: [[STORE_TEMP_INT:%.+]] = load i128, ptr [[STORE_TEMP_PTR]], align 16
  // CHECK: store atomic volatile i128 [[STORE_TEMP_INT]], ptr [[ADDR]] seq_cst, align 16
  // CHECK32-LABEL: @test_volatile_assign
  // CHECK32: store ptr %{{.+}}, ptr [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load ptr, ptr [[ADDR_ADDR]], align 4
  // CHECK32: call void @llvm.memset.p0.i64(ptr align 4 [[STORE_TEMP_PTR:%.+]], i8 0, i64 12, i1 false)
  // CHECK32: store x86_fp80 {{.+}}, ptr [[STORE_TEMP_PTR]], align 4
  // CHECK32: call void @__atomic_store(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[STORE_TEMP_PTR]], i32 noundef 5)
  *addr = 115;
  // CHECK: [[ADDR:%.+]] = load ptr, ptr %{{.+}}, align 8
  // CHECK: [[INT_VAL:%.+]] = load atomic volatile i128, ptr [[ADDR]] seq_cst, align 16
  // CHECK: store i128 [[INT_VAL]], ptr [[INT_LD_TEMP:%.+]], align 16
  // CHECK: [[RET_VAL:%.+]] = load x86_fp80, ptr [[LD_TEMP:%.+]], align 16
  // CHECK: ret x86_fp80 [[RET_VAL]]
  // CHECK32: [[ADDR:%.+]] = load ptr, ptr %{{.+}}, align 4
  // CHECK32: call void @__atomic_load(i32 noundef 12, ptr noundef [[ADDR]], ptr noundef [[LD_TEMP:%.+]], i32 noundef 5)
  // CHECK32: [[RET_VAL:%.+]] = load x86_fp80, ptr [[LD_TEMP]], align 4
  // CHECK32: ret x86_fp80 [[RET_VAL]]

  return *addr;
}
