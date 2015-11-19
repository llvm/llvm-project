// RUN: %clang_cc1 -triple x86_64-linux-gnu -target-cpu core2 %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple i686-linux-gnu -target-cpu core2 %s -S -emit-llvm -o - | FileCheck -check-prefix=CHECK32 %s

long double testinc(_Atomic long double *addr) {
  // CHECK-LABEL: @testinc
  // CHECK: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 8
  // CHECK: [[INT_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[INT_VALUE:%.+]] = load atomic i128, i128* [[INT_ADDR]] seq_cst, align 16
  // CHECK: [[INT_LOAD_ADDR:%.+]] = bitcast x86_fp80* [[LD_ADDR:%.+]] to i128*
  // CHECK: store i128 [[INT_VALUE]], i128* [[INT_LOAD_ADDR]], align 16
  // CHECK: [[LD_VALUE:%.+]] = load x86_fp80, x86_fp80* [[LD_ADDR]], align 16
  // CHECK: br label %[[ATOMIC_OP:.+]]
  // CHECK: [[ATOMIC_OP]]
  // CHECK: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK: [[INC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK: [[OLD_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR:%.+]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[OLD_VALUE_VOID_ADDR]], i8 0, i64 16, i32 16, i1 false)
  // CHECK: store x86_fp80 [[OLD_VALUE]], x86_fp80* [[OLD_VALUE_ADDR]], align 16
  // CHECK: [[OLD_INT_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR]] to i128*
  // CHECK: [[OLD_INT:%.+]] = load i128, i128* [[OLD_INT_ADDR]], align 16
  // CHECK: [[NEW_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[NEW_VALUE_ADDR:%.+]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[NEW_VALUE_VOID_ADDR]], i8 0, i64 16, i32 16, i1 false)
  // CHECK: store x86_fp80 [[INC_VALUE]], x86_fp80* [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[NEW_INT_ADDR:%.+]] = bitcast x86_fp80* [[NEW_VALUE_ADDR]] to i128*
  // CHECK: [[NEW_INT:%.+]] = load i128, i128* [[NEW_INT_ADDR]], align 16
  // CHECK: [[OBJ_INT_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[RES:%.+]] = cmpxchg i128* [[OBJ_INT_ADDR]], i128 [[OLD_INT]], i128 [[NEW_INT]] seq_cst seq_cst
  // CHECK: [[OLD_VALUE:%.+]] = extractvalue { i128, i1 } [[RES]], 0
  // CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
  // CHECK: [[OLD_VALUE_RES_INT_PTR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_RES_PTR:%.+]] to i128*
  // CHECK: store i128 [[OLD_VALUE]], i128* [[OLD_VALUE_RES_INT_PTR]], align 16
  // CHECK: [[LD_VALUE]] = load x86_fp80, x86_fp80* [[OLD_VALUE_RES_PTR]], align 16
  // CHECK: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK: [[ATOMIC_CONT]]
  // CHECK: ret x86_fp80 [[INC_VALUE]]
  // CHECK32-LABEL: @testinc
  // CHECK32: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 4
  // CHECK32: [[VOID_PTR:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[TEMP_LD_PTR:%.+]] = bitcast x86_fp80* [[TEMP_LD_ADDR:%.+]] to i8*
  // CHECK32: call void @__atomic_load(i32 12, i8* [[VOID_PTR]], i8* [[TEMP_LD_PTR]], i32 5)
  // CHECK32: [[LD_VALUE:%.+]] = load x86_fp80, x86_fp80* [[TEMP_LD_ADDR]], align 4
  // CHECK32: br label %[[ATOMIC_OP:.+]]
  // CHECK32: [[ATOMIC_OP]]
  // CHECK32: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK32: [[INC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK32: [[OLD_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR:%.+]] to i8*
  // CHECK32: call void @llvm.memset.p0i8.i64(i8* [[OLD_VALUE_VOID_ADDR]], i8 0, i64 12, i32 4, i1 false)
  // CHECK32: store x86_fp80 [[OLD_VALUE]], x86_fp80* [[OLD_VALUE_ADDR]], align 4
  // CHECK32: [[DESIRED_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[DESIRED_VALUE_ADDR:%.+]] to i8*
  // CHECK32: call void @llvm.memset.p0i8.i64(i8* [[DESIRED_VALUE_VOID_ADDR]], i8 0, i64 12, i32 4, i1 false)
  // CHECK32: store x86_fp80 [[INC_VALUE]], x86_fp80* [[DESIRED_VALUE_ADDR]], align 4
  // CHECK32: [[OBJ:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[EXPECTED:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR]] to i8*
  // CHECK32: [[DESIRED:%.+]] = bitcast x86_fp80* [[DESIRED_VALUE_ADDR]] to i8*
  // CHECK32: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i32 12, i8* [[OBJ]], i8* [[EXPECTED]], i8* [[DESIRED]], i32 5, i32 5)
  // CHECK32: [[LD_VALUE:%.+]] = load x86_fp80, x86_fp80* [[OLD_VALUE_ADDR]], align 4
  // CHECK32: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK32: [[ATOMIC_CONT]]
  // CHECK32: ret x86_fp80 [[INC_VALUE]]

  return ++*addr;
}

long double testdec(_Atomic long double *addr) {
  // CHECK-LABEL: @testdec
  // CHECK: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 8
  // CHECK: [[INT_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[INT_VALUE:%.+]] = load atomic i128, i128* [[INT_ADDR]] seq_cst, align 16
  // CHECK: [[INT_LOAD_ADDR:%.+]] = bitcast x86_fp80* [[LD_ADDR:%.+]] to i128*
  // CHECK: store i128 [[INT_VALUE]], i128* [[INT_LOAD_ADDR]], align 16
  // CHECK: [[ORIG_LD_VALUE:%.+]] = load x86_fp80, x86_fp80* [[LD_ADDR]], align 16
  // CHECK: br label %[[ATOMIC_OP:.+]]
  // CHECK: [[ATOMIC_OP]]
  // CHECK: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[ORIG_LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK: [[DEC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK: [[OLD_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR:%.+]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[OLD_VALUE_VOID_ADDR]], i8 0, i64 16, i32 16, i1 false)
  // CHECK: store x86_fp80 [[OLD_VALUE]], x86_fp80* [[OLD_VALUE_ADDR]], align 16
  // CHECK: [[OLD_INT_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR]] to i128*
  // CHECK: [[OLD_INT:%.+]] = load i128, i128* [[OLD_INT_ADDR]], align 16
  // CHECK: [[NEW_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[NEW_VALUE_ADDR:%.+]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[NEW_VALUE_VOID_ADDR]], i8 0, i64 16, i32 16, i1 false)
  // CHECK: store x86_fp80 [[DEC_VALUE]], x86_fp80* [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[NEW_INT_ADDR:%.+]] = bitcast x86_fp80* [[NEW_VALUE_ADDR]] to i128*
  // CHECK: [[NEW_INT:%.+]] = load i128, i128* [[NEW_INT_ADDR]], align 16
  // CHECK: [[OBJ_INT_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[RES:%.+]] = cmpxchg i128* [[OBJ_INT_ADDR]], i128 [[OLD_INT]], i128 [[NEW_INT]] seq_cst seq_cst
  // CHECK: [[OLD_VALUE:%.+]] = extractvalue { i128, i1 } [[RES]], 0
  // CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
  // CHECK: [[OLD_VALUE_RES_INT_PTR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_RES_PTR:%.+]] to i128*
  // CHECK: store i128 [[OLD_VALUE]], i128* [[OLD_VALUE_RES_INT_PTR]], align 16
  // CHECK: [[LD_VALUE]] = load x86_fp80, x86_fp80* [[OLD_VALUE_RES_PTR]], align 16
  // CHECK: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK: [[ATOMIC_CONT]]
  // CHECK: ret x86_fp80 [[ORIG_LD_VALUE]]
  // CHECK32-LABEL: @testdec
  // CHECK32: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 4
  // CHECK32: [[VOID_PTR:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[TEMP_LD_PTR:%.+]] = bitcast x86_fp80* [[TEMP_LD_ADDR:%.+]] to i8*
  // CHECK32: call void @__atomic_load(i32 12, i8* [[VOID_PTR]], i8* [[TEMP_LD_PTR]], i32 5)
  // CHECK32: [[ORIG_LD_VALUE:%.+]] = load x86_fp80, x86_fp80* [[TEMP_LD_ADDR]], align 4
  // CHECK32: br label %[[ATOMIC_OP:.+]]
  // CHECK32: [[ATOMIC_OP]]
  // CHECK32: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[ORIG_LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK32: [[DEC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK32: [[OLD_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR:%.+]] to i8*
  // CHECK32: call void @llvm.memset.p0i8.i64(i8* [[OLD_VALUE_VOID_ADDR]], i8 0, i64 12, i32 4, i1 false)
  // CHECK32: store x86_fp80 [[OLD_VALUE]], x86_fp80* [[OLD_VALUE_ADDR]], align 4
  // CHECK32: [[DESIRED_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[DESIRED_VALUE_ADDR:%.+]] to i8*
  // CHECK32: call void @llvm.memset.p0i8.i64(i8* [[DESIRED_VALUE_VOID_ADDR]], i8 0, i64 12, i32 4, i1 false)
  // CHECK32: store x86_fp80 [[DEC_VALUE]], x86_fp80* [[DESIRED_VALUE_ADDR]], align 4
  // CHECK32: [[OBJ:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[EXPECTED:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR]] to i8*
  // CHECK32: [[DESIRED:%.+]] = bitcast x86_fp80* [[DESIRED_VALUE_ADDR]] to i8*
  // CHECK32: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i32 12, i8* [[OBJ]], i8* [[EXPECTED]], i8* [[DESIRED]], i32 5, i32 5)
  // CHECK32: [[LD_VALUE]] = load x86_fp80, x86_fp80* [[OLD_VALUE_ADDR]], align 4
  // CHECK32: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK32: [[ATOMIC_CONT]]
  // CHECK32: ret x86_fp80 [[ORIG_LD_VALUE]]

  return (*addr)--;
}

long double testcompassign(_Atomic long double *addr) {
  *addr -= 25;
  // CHECK-LABEL: @testcompassign
  // CHECK: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 8
  // CHECK: [[INT_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[INT_VALUE:%.+]] = load atomic i128, i128* [[INT_ADDR]] seq_cst, align 16
  // CHECK: [[INT_LOAD_ADDR:%.+]] = bitcast x86_fp80* [[LD_ADDR:%.+]] to i128*
  // CHECK: store i128 [[INT_VALUE]], i128* [[INT_LOAD_ADDR]], align 16
  // CHECK: [[LD_VALUE:%.+]] = load x86_fp80, x86_fp80* [[LD_ADDR]], align 16
  // CHECK: br label %[[ATOMIC_OP:.+]]
  // CHECK: [[ATOMIC_OP]]
  // CHECK: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK: [[SUB_VALUE:%.+]] = fsub x86_fp80 [[OLD_VALUE]],
  // CHECK: [[OLD_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR:%.+]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[OLD_VALUE_VOID_ADDR]], i8 0, i64 16, i32 16, i1 false)
  // CHECK: store x86_fp80 [[OLD_VALUE]], x86_fp80* [[OLD_VALUE_ADDR]], align 16
  // CHECK: [[OLD_INT_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR]] to i128*
  // CHECK: [[OLD_INT:%.+]] = load i128, i128* [[OLD_INT_ADDR]], align 16
  // CHECK: [[NEW_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[NEW_VALUE_ADDR:%.+]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[NEW_VALUE_VOID_ADDR]], i8 0, i64 16, i32 16, i1 false)
  // CHECK: store x86_fp80 [[SUB_VALUE]], x86_fp80* [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[NEW_INT_ADDR:%.+]] = bitcast x86_fp80* [[NEW_VALUE_ADDR]] to i128*
  // CHECK: [[NEW_INT:%.+]] = load i128, i128* [[NEW_INT_ADDR]], align 16
  // CHECK: [[OBJ_INT_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[RES:%.+]] = cmpxchg i128* [[OBJ_INT_ADDR]], i128 [[OLD_INT]], i128 [[NEW_INT]] seq_cst seq_cst
  // CHECK: [[OLD_VALUE:%.+]] = extractvalue { i128, i1 } [[RES]], 0
  // CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
  // CHECK: [[OLD_VALUE_RES_INT_PTR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_RES_PTR:%.+]] to i128*
  // CHECK: store i128 [[OLD_VALUE]], i128* [[OLD_VALUE_RES_INT_PTR]], align 16
  // CHECK: [[LD_VALUE]] = load x86_fp80, x86_fp80* [[OLD_VALUE_RES_PTR]], align 16
  // CHECK: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK: [[ATOMIC_CONT]]
  // CHECK: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** %{{.+}}, align 8
  // CHECK: [[ADDR_INT:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[INT_VAL:%.+]] = load atomic i128, i128* [[ADDR_INT]] seq_cst, align 16
  // CHECK: [[INT_LD_TEMP:%.+]] = bitcast x86_fp80* [[LD_TEMP:%.+]] to i128*
  // CHECK: store i128 [[INT_VAL]], i128* [[INT_LD_TEMP:%.+]], align 16
  // CHECK: [[RET_VAL:%.+]] = load x86_fp80, x86_fp80* [[LD_TEMP]], align 16
  // CHECK: ret x86_fp80 [[RET_VAL]]
  // CHECK32-LABEL: @testcompassign
  // CHECK32: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 4
  // CHECK32: [[VOID_PTR:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[TEMP_LD_PTR:%.+]] = bitcast x86_fp80* [[TEMP_LD_ADDR:%.+]] to i8*
  // CHECK32: call void @__atomic_load(i32 12, i8* [[VOID_PTR]], i8* [[TEMP_LD_PTR]], i32 5)
  // CHECK32: [[LD_VALUE:%.+]] = load x86_fp80, x86_fp80* [[TEMP_LD_ADDR]], align 4
  // CHECK32: br label %[[ATOMIC_OP:.+]]
  // CHECK32: [[ATOMIC_OP]]
  // CHECK32: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK32: [[INC_VALUE:%.+]] = fsub x86_fp80 [[OLD_VALUE]],
  // CHECK32: [[OLD_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR:%.+]] to i8*
  // CHECK32: call void @llvm.memset.p0i8.i64(i8* [[OLD_VALUE_VOID_ADDR]], i8 0, i64 12, i32 4, i1 false)
  // CHECK32: store x86_fp80 [[OLD_VALUE]], x86_fp80* [[OLD_VALUE_ADDR]], align 4
  // CHECK32: [[DESIRED_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[DESIRED_VALUE_ADDR:%.+]] to i8*
  // CHECK32: call void @llvm.memset.p0i8.i64(i8* [[DESIRED_VALUE_VOID_ADDR]], i8 0, i64 12, i32 4, i1 false)
  // CHECK32: store x86_fp80 [[INC_VALUE]], x86_fp80* [[DESIRED_VALUE_ADDR]], align 4
  // CHECK32: [[OBJ:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[EXPECTED:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR]] to i8*
  // CHECK32: [[DESIRED:%.+]] = bitcast x86_fp80* [[DESIRED_VALUE_ADDR]] to i8*
  // CHECK32: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i32 12, i8* [[OBJ]], i8* [[EXPECTED]], i8* [[DESIRED]], i32 5, i32 5)
  // CHECK32: [[LD_VALUE]] = load x86_fp80, x86_fp80* [[OLD_VALUE_ADDR]], align 4
  // CHECK32: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK32: [[ATOMIC_CONT]]
  // CHECK32: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** %{{.+}}, align 4
  // CHECK32: [[VOID_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[VOID_GET_ADDR:%.+]] = bitcast x86_fp80* [[GET_ADDR:%.+]] to i8*
  // CHECK32: call void @__atomic_load(i32 12, i8* [[VOID_ADDR]], i8* [[VOID_GET_ADDR]], i32 5)
  // CHECK32: [[RET_VAL:%.+]] = load x86_fp80, x86_fp80* [[GET_ADDR]], align 4
  // CHECK32: ret x86_fp80 [[RET_VAL]]
  return *addr;
}

long double testassign(_Atomic long double *addr) {
  // CHECK-LABEL: @testassign
  // CHECK: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 8
  // CHECK: [[STORE_TEMP_VOID_PTR:%.+]] = bitcast x86_fp80* [[STORE_TEMP_PTR:%.+]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[STORE_TEMP_VOID_PTR]], i8 0, i64 16, i32 16, i1 false)
  // CHECK: store x86_fp80 {{.+}}, x86_fp80* [[STORE_TEMP_PTR]], align 16
  // CHECK: [[STORE_TEMP_INT_PTR:%.+]] = bitcast x86_fp80* [[STORE_TEMP_PTR]] to i128*
  // CHECK: [[STORE_TEMP_INT:%.+]] = load i128, i128* [[STORE_TEMP_INT_PTR]], align 16
  // CHECK: [[ADDR_INT:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: store atomic i128 [[STORE_TEMP_INT]], i128* [[ADDR_INT]] seq_cst, align 16
  // CHECK32-LABEL: @testassign
  // CHECK32: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 4
  // CHECK32: [[STORE_TEMP_VOID_PTR:%.+]] = bitcast x86_fp80* [[STORE_TEMP_PTR:%.+]] to i8*
  // CHECK32: call void @llvm.memset.p0i8.i64(i8* [[STORE_TEMP_VOID_PTR]], i8 0, i64 12, i32 4, i1 false)
  // CHECK32: store x86_fp80 {{.+}}, x86_fp80* [[STORE_TEMP_PTR]], align 4
  // CHECK32: [[ADDR_VOID:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[STORE_TEMP_VOID_PTR:%.+]] = bitcast x86_fp80* [[STORE_TEMP_PTR]] to i8*
  // CHECK32: call void @__atomic_store(i32 12, i8* [[ADDR_VOID]], i8* [[STORE_TEMP_VOID_PTR]], i32 5)
  *addr = 115;
  // CHECK: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** %{{.+}}, align 8
  // CHECK: [[ADDR_INT:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[INT_VAL:%.+]] = load atomic i128, i128* [[ADDR_INT]] seq_cst, align 16
  // CHECK: [[INT_LD_TEMP:%.+]] = bitcast x86_fp80* [[LD_TEMP:%.+]] to i128*
  // CHECK: store i128 [[INT_VAL]], i128* [[INT_LD_TEMP:%.+]], align 16
  // CHECK: [[RET_VAL:%.+]] = load x86_fp80, x86_fp80* [[LD_TEMP]], align 16
  // CHECK: ret x86_fp80 [[RET_VAL]]
  // CHECK32: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** %{{.+}}, align 4
  // CHECK32: [[VOID_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[VOID_LD_TEMP:%.+]] = bitcast x86_fp80* [[LD_TEMP:%.+]] to i8*
  // CHECK32: call void @__atomic_load(i32 12, i8* [[VOID_ADDR]], i8* [[VOID_LD_TEMP]], i32 5)
  // CHECK32: [[RET_VAL:%.+]] = load x86_fp80, x86_fp80* [[LD_TEMP]], align 4
  // CHECK32: ret x86_fp80 [[RET_VAL]]

  return *addr;
}

long double test_volatile_inc(volatile _Atomic long double *addr) {
  // CHECK-LABEL: @test_volatile_inc
  // CHECK: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 8
  // CHECK: [[INT_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[INT_VALUE:%.+]] = load atomic volatile i128, i128* [[INT_ADDR]] seq_cst, align 16
  // CHECK: [[INT_LOAD_ADDR:%.+]] = bitcast x86_fp80* [[LD_ADDR:%.+]] to i128*
  // CHECK: store i128 [[INT_VALUE]], i128* [[INT_LOAD_ADDR]], align 16
  // CHECK: [[LD_VALUE:%.+]] = load x86_fp80, x86_fp80* [[LD_ADDR]], align 16
  // CHECK: br label %[[ATOMIC_OP:.+]]
  // CHECK: [[ATOMIC_OP]]
  // CHECK: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK: [[INC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK: [[OLD_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR:%.+]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[OLD_VALUE_VOID_ADDR]], i8 0, i64 16, i32 16, i1 false)
  // CHECK: store x86_fp80 [[OLD_VALUE]], x86_fp80* [[OLD_VALUE_ADDR]], align 16
  // CHECK: [[OLD_INT_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR]] to i128*
  // CHECK: [[OLD_INT:%.+]] = load i128, i128* [[OLD_INT_ADDR]], align 16
  // CHECK: [[NEW_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[NEW_VALUE_ADDR:%.+]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[NEW_VALUE_VOID_ADDR]], i8 0, i64 16, i32 16, i1 false)
  // CHECK: store x86_fp80 [[INC_VALUE]], x86_fp80* [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[NEW_INT_ADDR:%.+]] = bitcast x86_fp80* [[NEW_VALUE_ADDR]] to i128*
  // CHECK: [[NEW_INT:%.+]] = load i128, i128* [[NEW_INT_ADDR]], align 16
  // CHECK: [[OBJ_INT_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[RES:%.+]] = cmpxchg volatile i128* [[OBJ_INT_ADDR]], i128 [[OLD_INT]], i128 [[NEW_INT]] seq_cst seq_cst
  // CHECK: [[OLD_VALUE:%.+]] = extractvalue { i128, i1 } [[RES]], 0
  // CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
  // CHECK: [[OLD_VALUE_RES_INT_PTR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_RES_PTR:%.+]] to i128*
  // CHECK: store i128 [[OLD_VALUE]], i128* [[OLD_VALUE_RES_INT_PTR]], align 16
  // CHECK: [[LD_VALUE]] = load x86_fp80, x86_fp80* [[OLD_VALUE_RES_PTR]], align 16
  // CHECK: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK: [[ATOMIC_CONT]]
  // CHECK: ret x86_fp80 [[INC_VALUE]]
  // CHECK32-LABEL: @test_volatile_inc
  // CHECK32: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 4
  // CHECK32: [[VOID_PTR:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[TEMP_LD_PTR:%.+]] = bitcast x86_fp80* [[TEMP_LD_ADDR:%.+]] to i8*
  // CHECK32: call void @__atomic_load(i32 12, i8* [[VOID_PTR]], i8* [[TEMP_LD_PTR]], i32 5)
  // CHECK32: [[LD_VALUE:%.+]] = load x86_fp80, x86_fp80* [[TEMP_LD_ADDR]], align 4
  // CHECK32: br label %[[ATOMIC_OP:.+]]
  // CHECK32: [[ATOMIC_OP]]
  // CHECK32: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK32: [[INC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK32: [[OLD_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR:%.+]] to i8*
  // CHECK32: call void @llvm.memset.p0i8.i64(i8* [[OLD_VALUE_VOID_ADDR]], i8 0, i64 12, i32 4, i1 false)
  // CHECK32: store x86_fp80 [[OLD_VALUE]], x86_fp80* [[OLD_VALUE_ADDR]], align 4
  // CHECK32: [[DESIRED_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[DESIRED_VALUE_ADDR:%.+]] to i8*
  // CHECK32: call void @llvm.memset.p0i8.i64(i8* [[DESIRED_VALUE_VOID_ADDR]], i8 0, i64 12, i32 4, i1 false)
  // CHECK32: store x86_fp80 [[INC_VALUE]], x86_fp80* [[DESIRED_VALUE_ADDR]], align 4
  // CHECK32: [[OBJ:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[EXPECTED:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR]] to i8*
  // CHECK32: [[DESIRED:%.+]] = bitcast x86_fp80* [[DESIRED_VALUE_ADDR]] to i8*
  // CHECK32: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i32 12, i8* [[OBJ]], i8* [[EXPECTED]], i8* [[DESIRED]], i32 5, i32 5)
  // CHECK32: [[LD_VALUE]] = load x86_fp80, x86_fp80* [[OLD_VALUE_ADDR]], align 4
  // CHECK32: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK32: [[ATOMIC_CONT]]
  // CHECK32: ret x86_fp80 [[INC_VALUE]]
  return ++*addr;
}

long double test_volatile_dec(volatile _Atomic long double *addr) {
  // CHECK-LABEL: @test_volatile_dec
  // CHECK: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 8
  // CHECK: [[INT_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[INT_VALUE:%.+]] = load atomic volatile i128, i128* [[INT_ADDR]] seq_cst, align 16
  // CHECK: [[INT_LOAD_ADDR:%.+]] = bitcast x86_fp80* [[LD_ADDR:%.+]] to i128*
  // CHECK: store i128 [[INT_VALUE]], i128* [[INT_LOAD_ADDR]], align 16
  // CHECK: [[ORIG_LD_VALUE:%.+]] = load x86_fp80, x86_fp80* [[LD_ADDR]], align 16
  // CHECK: br label %[[ATOMIC_OP:.+]]
  // CHECK: [[ATOMIC_OP]]
  // CHECK: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[ORIG_LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK: [[DEC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK: [[OLD_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR:%.+]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[OLD_VALUE_VOID_ADDR]], i8 0, i64 16, i32 16, i1 false)
  // CHECK: store x86_fp80 [[OLD_VALUE]], x86_fp80* [[OLD_VALUE_ADDR]], align 16
  // CHECK: [[OLD_INT_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR]] to i128*
  // CHECK: [[OLD_INT:%.+]] = load i128, i128* [[OLD_INT_ADDR]], align 16
  // CHECK: [[NEW_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[NEW_VALUE_ADDR:%.+]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[NEW_VALUE_VOID_ADDR]], i8 0, i64 16, i32 16, i1 false)
  // CHECK: store x86_fp80 [[DEC_VALUE]], x86_fp80* [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[NEW_INT_ADDR:%.+]] = bitcast x86_fp80* [[NEW_VALUE_ADDR]] to i128*
  // CHECK: [[NEW_INT:%.+]] = load i128, i128* [[NEW_INT_ADDR]], align 16
  // CHECK: [[OBJ_INT_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[RES:%.+]] = cmpxchg volatile i128* [[OBJ_INT_ADDR]], i128 [[OLD_INT]], i128 [[NEW_INT]] seq_cst seq_cst
  // CHECK: [[OLD_VALUE:%.+]] = extractvalue { i128, i1 } [[RES]], 0
  // CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
  // CHECK: [[OLD_VALUE_RES_INT_PTR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_RES_PTR:%.+]] to i128*
  // CHECK: store i128 [[OLD_VALUE]], i128* [[OLD_VALUE_RES_INT_PTR]], align 16
  // CHECK: [[LD_VALUE]] = load x86_fp80, x86_fp80* [[OLD_VALUE_RES_PTR]], align 16
  // CHECK: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK: [[ATOMIC_CONT]]
  // CHECK: ret x86_fp80 [[ORIG_LD_VALUE]]
  // CHECK32-LABEL: @test_volatile_dec
  // CHECK32: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 4
  // CHECK32: [[VOID_PTR:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[TEMP_LD_PTR:%.+]] = bitcast x86_fp80* [[TEMP_LD_ADDR:%.+]] to i8*
  // CHECK32: call void @__atomic_load(i32 12, i8* [[VOID_PTR]], i8* [[TEMP_LD_PTR]], i32 5)
  // CHECK32: [[ORIG_LD_VALUE:%.+]] = load x86_fp80, x86_fp80* [[TEMP_LD_ADDR]], align 4
  // CHECK32: br label %[[ATOMIC_OP:.+]]
  // CHECK32: [[ATOMIC_OP]]
  // CHECK32: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[ORIG_LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK32: [[DEC_VALUE:%.+]] = fadd x86_fp80 [[OLD_VALUE]],
  // CHECK32: [[OLD_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR:%.+]] to i8*
  // CHECK32: call void @llvm.memset.p0i8.i64(i8* [[OLD_VALUE_VOID_ADDR]], i8 0, i64 12, i32 4, i1 false)
  // CHECK32: store x86_fp80 [[OLD_VALUE]], x86_fp80* [[OLD_VALUE_ADDR]], align 4
  // CHECK32: [[DESIRED_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[DESIRED_VALUE_ADDR:%.+]] to i8*
  // CHECK32: call void @llvm.memset.p0i8.i64(i8* [[DESIRED_VALUE_VOID_ADDR]], i8 0, i64 12, i32 4, i1 false)
  // CHECK32: store x86_fp80 [[DEC_VALUE]], x86_fp80* [[DESIRED_VALUE_ADDR]], align 4
  // CHECK32: [[OBJ:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[EXPECTED:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR]] to i8*
  // CHECK32: [[DESIRED:%.+]] = bitcast x86_fp80* [[DESIRED_VALUE_ADDR]] to i8*
  // CHECK32: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i32 12, i8* [[OBJ]], i8* [[EXPECTED]], i8* [[DESIRED]], i32 5, i32 5)
  // CHECK32: [[LD_VALUE]] = load x86_fp80, x86_fp80* [[OLD_VALUE_ADDR]], align 4
  // CHECK32: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK32: [[ATOMIC_CONT]]
  // CHECK32: ret x86_fp80 [[ORIG_LD_VALUE]]
  return (*addr)--;
}

long double test_volatile_compassign(volatile _Atomic long double *addr) {
  *addr -= 25;
  // CHECK-LABEL: @test_volatile_compassign
  // CHECK: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 8
  // CHECK: [[INT_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[INT_VALUE:%.+]] = load atomic volatile i128, i128* [[INT_ADDR]] seq_cst, align 16
  // CHECK: [[INT_LOAD_ADDR:%.+]] = bitcast x86_fp80* [[LD_ADDR:%.+]] to i128*
  // CHECK: store i128 [[INT_VALUE]], i128* [[INT_LOAD_ADDR]], align 16
  // CHECK: [[LD_VALUE:%.+]] = load x86_fp80, x86_fp80* [[LD_ADDR]], align 16
  // CHECK: br label %[[ATOMIC_OP:.+]]
  // CHECK: [[ATOMIC_OP]]
  // CHECK: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK: [[SUB_VALUE:%.+]] = fsub x86_fp80 [[OLD_VALUE]],
  // CHECK: [[OLD_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR:%.+]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[OLD_VALUE_VOID_ADDR]], i8 0, i64 16, i32 16, i1 false)
  // CHECK: store x86_fp80 [[OLD_VALUE]], x86_fp80* [[OLD_VALUE_ADDR]], align 16
  // CHECK: [[OLD_INT_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR]] to i128*
  // CHECK: [[OLD_INT:%.+]] = load i128, i128* [[OLD_INT_ADDR]], align 16
  // CHECK: [[NEW_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[NEW_VALUE_ADDR:%.+]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[NEW_VALUE_VOID_ADDR]], i8 0, i64 16, i32 16, i1 false)
  // CHECK: store x86_fp80 [[SUB_VALUE]], x86_fp80* [[NEW_VALUE_ADDR]], align 16
  // CHECK: [[NEW_INT_ADDR:%.+]] = bitcast x86_fp80* [[NEW_VALUE_ADDR]] to i128*
  // CHECK: [[NEW_INT:%.+]] = load i128, i128* [[NEW_INT_ADDR]], align 16
  // CHECK: [[OBJ_INT_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[RES:%.+]] = cmpxchg volatile i128* [[OBJ_INT_ADDR]], i128 [[OLD_INT]], i128 [[NEW_INT]] seq_cst seq_cst
  // CHECK: [[OLD_VALUE:%.+]] = extractvalue { i128, i1 } [[RES]], 0
  // CHECK: [[FAIL_SUCCESS:%.+]] = extractvalue { i128, i1 } [[RES]], 1
  // CHECK: [[OLD_VALUE_RES_INT_PTR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_RES_PTR:%.+]] to i128*
  // CHECK: store i128 [[OLD_VALUE]], i128* [[OLD_VALUE_RES_INT_PTR]], align 16
  // CHECK: [[LD_VALUE]] = load x86_fp80, x86_fp80* [[OLD_VALUE_RES_PTR]], align 16
  // CHECK: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK: [[ATOMIC_CONT]]
  // CHECK: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** %{{.+}}, align 8
  // CHECK: [[ADDR_INT:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[INT_VAL:%.+]] = load atomic volatile i128, i128* [[ADDR_INT]] seq_cst, align 16
  // CHECK: [[INT_LD_TEMP:%.+]] = bitcast x86_fp80* [[LD_TEMP:%.+]] to i128*
  // CHECK: store i128 [[INT_VAL]], i128* [[INT_LD_TEMP:%.+]], align 16
  // CHECK: [[RET_VAL:%.+]] = load x86_fp80, x86_fp80* [[LD_TEMP]], align 16
  // CHECK32-LABEL: @test_volatile_compassign
  // CHECK32: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 4
  // CHECK32: [[VOID_PTR:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[TEMP_LD_PTR:%.+]] = bitcast x86_fp80* [[TEMP_LD_ADDR:%.+]] to i8*
  // CHECK32: call void @__atomic_load(i32 12, i8* [[VOID_PTR]], i8* [[TEMP_LD_PTR]], i32 5)
  // CHECK32: [[LD_VALUE:%.+]] = load x86_fp80, x86_fp80* [[TEMP_LD_ADDR]], align 4
  // CHECK32: br label %[[ATOMIC_OP:.+]]
  // CHECK32: [[ATOMIC_OP]]
  // CHECK32: [[OLD_VALUE:%.+]] = phi x86_fp80 [ [[LD_VALUE]], %{{.+}} ], [ [[LD_VALUE:%.+]], %[[ATOMIC_OP]] ]
  // CHECK32: [[INC_VALUE:%.+]] = fsub x86_fp80 [[OLD_VALUE]],
  // CHECK32: [[OLD_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR:%.+]] to i8*
  // CHECK32: call void @llvm.memset.p0i8.i64(i8* [[OLD_VALUE_VOID_ADDR]], i8 0, i64 12, i32 4, i1 false)
  // CHECK32: store x86_fp80 [[OLD_VALUE]], x86_fp80* [[OLD_VALUE_ADDR]], align 4
  // CHECK32: [[DESIRED_VALUE_VOID_ADDR:%.+]] = bitcast x86_fp80* [[DESIRED_VALUE_ADDR:%.+]] to i8*
  // CHECK32: call void @llvm.memset.p0i8.i64(i8* [[DESIRED_VALUE_VOID_ADDR]], i8 0, i64 12, i32 4, i1 false)
  // CHECK32: store x86_fp80 [[INC_VALUE]], x86_fp80* [[DESIRED_VALUE_ADDR]], align 4
  // CHECK32: [[OBJ:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[EXPECTED:%.+]] = bitcast x86_fp80* [[OLD_VALUE_ADDR]] to i8*
  // CHECK32: [[DESIRED:%.+]] = bitcast x86_fp80* [[DESIRED_VALUE_ADDR]] to i8*
  // CHECK32: [[FAIL_SUCCESS:%.+]] = call zeroext i1 @__atomic_compare_exchange(i32 12, i8* [[OBJ]], i8* [[EXPECTED]], i8* [[DESIRED]], i32 5, i32 5)
  // CHECK32: [[LD_VALUE]] = load x86_fp80, x86_fp80* [[OLD_VALUE_ADDR]], align 4
  // CHECK32: br i1 [[FAIL_SUCCESS]], label %[[ATOMIC_CONT:.+]], label %[[ATOMIC_OP]]
  // CHECK32: [[ATOMIC_CONT]]
  // CHECK32: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** %{{.+}}, align 4
  // CHECK32: [[VOID_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[VOID_GET_ADDR:%.+]] = bitcast x86_fp80* [[GET_ADDR:%.+]] to i8*
  // CHECK32: call void @__atomic_load(i32 12, i8* [[VOID_ADDR]], i8* [[VOID_GET_ADDR]], i32 5)
  // CHECK32: [[RET_VAL:%.+]] = load x86_fp80, x86_fp80* [[GET_ADDR]], align 4
  // CHECK32: ret x86_fp80 [[RET_VAL]]
  return *addr;
}

long double test_volatile_assign(volatile _Atomic long double *addr) {
  // CHECK-LABEL: @test_volatile_assign
  // CHECK: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 8
  // CHECK: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 8
  // CHECK: [[STORE_TEMP_VOID_PTR:%.+]] = bitcast x86_fp80* [[STORE_TEMP_PTR:%.+]] to i8*
  // CHECK: call void @llvm.memset.p0i8.i64(i8* [[STORE_TEMP_VOID_PTR]], i8 0, i64 16, i32 16, i1 false)
  // CHECK: store x86_fp80 {{.+}}, x86_fp80* [[STORE_TEMP_PTR]], align 16
  // CHECK: [[STORE_TEMP_INT_PTR:%.+]] = bitcast x86_fp80* [[STORE_TEMP_PTR]] to i128*
  // CHECK: [[STORE_TEMP_INT:%.+]] = load i128, i128* [[STORE_TEMP_INT_PTR]], align 16
  // CHECK: [[ADDR_INT:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: store atomic volatile i128 [[STORE_TEMP_INT]], i128* [[ADDR_INT]] seq_cst, align 16
  // CHECK32-LABEL: @test_volatile_assign
  // CHECK32: store x86_fp80* %{{.+}}, x86_fp80** [[ADDR_ADDR:%.+]], align 4
  // CHECK32: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** [[ADDR_ADDR]], align 4
  // CHECK32: [[STORE_TEMP_VOID_PTR:%.+]] = bitcast x86_fp80* [[STORE_TEMP_PTR:%.+]] to i8*
  // CHECK32: call void @llvm.memset.p0i8.i64(i8* [[STORE_TEMP_VOID_PTR]], i8 0, i64 12, i32 4, i1 false)
  // CHECK32: store x86_fp80 {{.+}}, x86_fp80* [[STORE_TEMP_PTR]], align 4
  // CHECK32: [[ADDR_VOID:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[STORE_TEMP_VOID_PTR:%.+]] = bitcast x86_fp80* [[STORE_TEMP_PTR]] to i8*
  // CHECK32: call void @__atomic_store(i32 12, i8* [[ADDR_VOID]], i8* [[STORE_TEMP_VOID_PTR]], i32 5)
  *addr = 115;
  // CHECK: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** %{{.+}}, align 8
  // CHECK: [[ADDR_INT:%.+]] = bitcast x86_fp80* [[ADDR]] to i128*
  // CHECK: [[INT_VAL:%.+]] = load atomic volatile i128, i128* [[ADDR_INT]] seq_cst, align 16
  // CHECK: [[INT_LD_TEMP:%.+]] = bitcast x86_fp80* [[LD_TEMP:%.+]] to i128*
  // CHECK: store i128 [[INT_VAL]], i128* [[INT_LD_TEMP:%.+]], align 16
  // CHECK: [[RET_VAL:%.+]] = load x86_fp80, x86_fp80* [[LD_TEMP]], align 16
  // CHECK: ret x86_fp80 [[RET_VAL]]
  // CHECK32: [[ADDR:%.+]] = load x86_fp80*, x86_fp80** %{{.+}}, align 4
  // CHECK32: [[VOID_ADDR:%.+]] = bitcast x86_fp80* [[ADDR]] to i8*
  // CHECK32: [[VOID_LD_TEMP:%.+]] = bitcast x86_fp80* [[LD_TEMP:%.+]] to i8*
  // CHECK32: call void @__atomic_load(i32 12, i8* [[VOID_ADDR]], i8* [[VOID_LD_TEMP]], i32 5)
  // CHECK32: [[RET_VAL:%.+]] = load x86_fp80, x86_fp80* [[LD_TEMP]], align 4
  // CHECK32: ret x86_fp80 [[RET_VAL]]

  return *addr;
}
