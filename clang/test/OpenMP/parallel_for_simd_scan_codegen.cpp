// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

void foo();
void bar();

// CHECK: define{{.*}} void @{{.*}}baz{{.*}}(i32 noundef %n)
void baz(int n) {
  static float a[10];
  static double b;

  // CHECK: call ptr @llvm.stacksave()
  // CHECK: [[A_BUF_SIZE:%.+]] = mul nuw i64 10, [[NUM_ELEMS:%[^,]+]]

  // float a_buffer[10][n];
  // CHECK: [[A_BUF:%.+]] = alloca float, i64 [[A_BUF_SIZE]],
  // CHECK: [[B_BUF:%.+]] = alloca double, i64 10,

  // CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(

  // CHECK: [[A_BUF_SIZE:%.+]] = mul nuw i64 10, [[NUM_ELEMS:%[^,]+]]

  // float a_buffer[10][n];
  // CHECK: [[A_BUF:%.+]] = alloca float, i64 [[A_BUF_SIZE]],

  // double b_buffer[10];
  // CHECK: [[B_BUF:%.+]] = alloca double, i64 10,
  // CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(
  // CHECK: call void @llvm.stackrestore(ptr

#pragma omp parallel for simd reduction(inscan, +:a[:n], b)
  for (int i = 0; i < 10; ++i) {
    // CHECK: call void @__kmpc_for_static_init_4(
    // CHECK: call ptr @llvm.stacksave()
    // CHECK: store float 0.000000e+00, ptr %
    // CHECK: store double 0.000000e+00, ptr [[B_PRIV_ADDR:%.+]],
    // CHECK: br label %[[DISPATCH:[^,]+]]
    // CHECK: [[INPUT_PHASE:.+]]:
    // CHECK: call void @{{.+}}foo{{.+}}()

    // a_buffer[i][0..n] = a_priv[[0..n];
    // CHECK: [[BASE_IDX_I:%.+]] = load i32, ptr [[IV_ADDR:%.+]],
    // CHECK: [[BASE_IDX:%.+]] = zext i32 [[BASE_IDX_I]] to i64
    // CHECK: [[IDX:%.+]] = mul nsw i64 [[BASE_IDX]], [[NUM_ELEMS:%.+]]
    // CHECK: [[A_BUF_IDX:%.+]] = getelementptr inbounds float, ptr [[A_BUF:%.+]], i64 [[IDX]]
    // CHECK: [[A_PRIV:%.+]] = getelementptr inbounds [10 x float], ptr [[A_PRIV_ADDR:%.+]], i64 0, i64 0
    // CHECK: [[BYTES:%.+]] = mul nuw i64 [[NUM_ELEMS:%.+]], 4
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}[[A_BUF_IDX]], ptr {{.*}}[[A_PRIV]], i64 [[BYTES]], i1 false)

    // b_buffer[i] = b_priv;
    // CHECK: [[B_BUF_IDX:%.+]] = getelementptr inbounds double, ptr [[B_BUF:%.+]], i64 [[BASE_IDX]]
    // CHECK: [[B_PRIV:%.+]] = load double, ptr [[B_PRIV_ADDR]],
    // CHECK: store double [[B_PRIV]], ptr [[B_BUF_IDX]],
    // CHECK: br label %[[LOOP_CONTINUE:.+]]

    // CHECK: [[DISPATCH]]:
    // CHECK: br label %[[INPUT_PHASE]]
    // CHECK: [[LOOP_CONTINUE]]:
    // CHECK: call void @llvm.stackrestore(ptr %
    // CHECK: call void @__kmpc_for_static_fini(
    // CHECK: call void @__kmpc_barrier(
    foo();
#pragma omp scan inclusive(a[:n], b)
    // CHECK: [[LOG2_10:%.+]] = call double @llvm.log2.f64(double 1.000000e+01)
    // CHECK: [[CEIL_LOG2_10:%.+]] = call double @llvm.ceil.f64(double [[LOG2_10]])
    // CHECK: [[CEIL_LOG2_10_INT:%.+]] = fptoui double [[CEIL_LOG2_10]] to i32
    // CHECK: br label %[[OUTER_BODY:[^,]+]]
    // CHECK: [[OUTER_BODY]]:
    // CHECK: [[K:%.+]] = phi i32 [ 0, %{{.+}} ], [ [[K_NEXT:%.+]], %{{.+}} ]
    // CHECK: [[K2POW:%.+]] = phi i64 [ 1, %{{.+}} ], [ [[K2POW_NEXT:%.+]], %{{.+}} ]
    // CHECK: [[CMP:%.+]] = icmp uge i64 9, [[K2POW]]
    // CHECK: br i1 [[CMP]], label %[[INNER_BODY:[^,]+]], label %[[INNER_EXIT:[^,]+]]
    // CHECK: [[INNER_BODY]]:
    // CHECK: [[I:%.+]] = phi i64 [ 9, %[[OUTER_BODY]] ], [ [[I_PREV:%.+]], %{{.+}} ]

    // a_buffer[i] += a_buffer[i-pow(2, k)];
    // CHECK: [[IDX:%.+]] = mul nsw i64 [[I]], [[NUM_ELEMS]]
    // CHECK: [[A_BUF_IDX:%.+]] = getelementptr inbounds float, ptr [[A_BUF]], i64 [[IDX]]
    // CHECK: [[IDX_SUB_K2POW:%.+]] = sub nuw i64 [[I]], [[K2POW]]
    // CHECK: [[IDX:%.+]] = mul nsw i64 [[IDX_SUB_K2POW]], [[NUM_ELEMS]]
    // CHECK: [[A_BUF_IDX_SUB_K2POW:%.+]] = getelementptr inbounds float, ptr [[A_BUF]], i64 [[IDX]]
    // CHECK: [[B_BUF_IDX:%.+]] = getelementptr inbounds double, ptr [[B_BUF]], i64 [[I]]
    // CHECK: [[IDX_SUB_K2POW:%.+]] = sub nuw i64 [[I]], [[K2POW]]
    // CHECK: [[B_BUF_IDX_SUB_K2POW:%.+]] = getelementptr inbounds double, ptr [[B_BUF]], i64 [[IDX_SUB_K2POW]]
    // CHECK: [[A_BUF_END:%.+]] = getelementptr float, ptr [[A_BUF_IDX]], i64 [[NUM_ELEMS]]
    // CHECK: [[ISEMPTY:%.+]] = icmp eq ptr [[A_BUF_IDX]], [[A_BUF_END]]
    // CHECK: br i1 [[ISEMPTY]], label %[[RED_DONE:[^,]+]], label %[[RED_BODY:[^,]+]]
    // CHECK: [[RED_BODY]]:
    // CHECK: [[A_BUF_IDX_SUB_K2POW_ELEM:%.+]] = phi ptr [ [[A_BUF_IDX_SUB_K2POW]], %[[INNER_BODY]] ], [ [[A_BUF_IDX_SUB_K2POW_NEXT:%.+]], %[[RED_BODY]] ]
    // CHECK: [[A_BUF_IDX_ELEM:%.+]] = phi ptr [ [[A_BUF_IDX]], %[[INNER_BODY]] ], [ [[A_BUF_IDX_NEXT:%.+]], %[[RED_BODY]] ]
    // CHECK: [[A_BUF_IDX_VAL:%.+]] = load float, ptr [[A_BUF_IDX_ELEM]],
    // CHECK: [[A_BUF_IDX_SUB_K2POW_VAL:%.+]] = load float, ptr [[A_BUF_IDX_SUB_K2POW_ELEM]],
    // CHECK: [[RED:%.+]] = fadd float [[A_BUF_IDX_VAL]], [[A_BUF_IDX_SUB_K2POW_VAL]]
    // CHECK: store float [[RED]], ptr [[A_BUF_IDX_ELEM]],
    // CHECK: [[A_BUF_IDX_NEXT]] = getelementptr float, ptr [[A_BUF_IDX_ELEM]], i32 1
    // CHECK: [[A_BUF_IDX_SUB_K2POW_NEXT]] = getelementptr float, ptr [[A_BUF_IDX_SUB_K2POW_ELEM]], i32 1
    // CHECK: [[DONE:%.+]] = icmp eq ptr [[A_BUF_IDX_NEXT]], [[A_BUF_END]]
    // CHECK: br i1 [[DONE]], label %[[RED_DONE]], label %[[RED_BODY]]
    // CHECK: [[RED_DONE]]:

    // b_buffer[i] += b_buffer[i-pow(2, k)];
    // CHECK: [[B_BUF_IDX_VAL:%.+]] = load double, ptr [[B_BUF_IDX]],
    // CHECK: [[B_BUF_IDX_SUB_K2POW_VAL:%.+]] = load double, ptr [[B_BUF_IDX_SUB_K2POW]],
    // CHECK: [[RED:%.+]] = fadd double [[B_BUF_IDX_VAL]], [[B_BUF_IDX_SUB_K2POW_VAL]]
    // CHECK: store double [[RED]], ptr [[B_BUF_IDX]],

    // --i;
    // CHECK: [[I_PREV:%.+]] = sub nuw i64 [[I]], 1
    // CHECK: [[CMP:%.+]] = icmp uge i64 [[I_PREV]], [[K2POW]]
    // CHECK: br i1 [[CMP]], label %[[INNER_BODY]], label %[[INNER_EXIT]]
    // CHECK: [[INNER_EXIT]]:

    // ++k;
    // CHECK: [[K_NEXT]] = add nuw i32 [[K]], 1
    // k2pow <<= 1;
    // CHECK: [[K2POW_NEXT]] = shl nuw i64 [[K2POW]], 1
    // CHECK: [[CMP:%.+]] = icmp ne i32 [[K_NEXT]], [[CEIL_LOG2_10_INT]]
    // CHECK: br i1 [[CMP]], label %[[OUTER_BODY]], label %[[OUTER_EXIT:[^,]+]]
    // CHECK: [[OUTER_EXIT]]:
    bar();
    // CHECK: call void @__kmpc_for_static_init_4(
    // CHECK: call ptr @llvm.stacksave()
    // CHECK: store float 0.000000e+00, ptr %
    // CHECK: store double 0.000000e+00, ptr [[B_PRIV_ADDR:%.+]],
    // CHECK: br label %[[DISPATCH:[^,]+]]

    // Skip the before scan body.
    // CHECK: call void @{{.+}}foo{{.+}}()

    // CHECK: [[EXIT_INSCAN:[^,]+]]:
    // CHECK: br label %[[LOOP_CONTINUE:[^,]+]]

    // CHECK: [[DISPATCH]]:
    // a_priv[[0..n] = a_buffer[i][0..n];
    // CHECK: [[BASE_IDX_I:%.+]] = load i32, ptr [[IV_ADDR:%.+]],
    // CHECK: [[BASE_IDX:%.+]] = zext i32 [[BASE_IDX_I]] to i64
    // CHECK: [[IDX:%.+]] = mul nsw i64 [[BASE_IDX]], [[NUM_ELEMS]]
    // CHECK: [[A_BUF_IDX:%.+]] = getelementptr inbounds float, ptr [[A_BUF]], i64 [[IDX]]
    // CHECK: [[A_PRIV:%.+]] = getelementptr inbounds [10 x float], ptr [[A_PRIV_ADDR:%.+]], i64 0, i64 0
    // CHECK: [[BYTES:%.+]] = mul nuw i64 [[NUM_ELEMS:%.+]], 4
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}[[A_PRIV]], ptr {{.*}}[[A_BUF_IDX]], i64 [[BYTES]], i1 false)

    // b_priv = b_buffer[i];
    // CHECK: [[B_BUF_IDX:%.+]] = getelementptr inbounds double, ptr [[B_BUF]], i64 [[BASE_IDX]]
    // CHECK: [[B_BUF_IDX_VAL:%.+]] = load double, ptr [[B_BUF_IDX]],
    // CHECK: store double [[B_BUF_IDX_VAL]], ptr [[B_PRIV_ADDR]],
    // CHECK: br label %[[SCAN_PHASE:[^,]+]]

    // CHECK: [[SCAN_PHASE]]:
    // CHECK: call void @{{.+}}bar{{.+}}()
    // CHECK: br label %[[EXIT_INSCAN]]

    // CHECK: [[LOOP_CONTINUE]]:
    // CHECK: call void @llvm.stackrestore(ptr %
    // CHECK: call void @__kmpc_for_static_fini(
  }

#pragma omp parallel for simd reduction(inscan, +:a[:n], b)
  for (int i = 0; i < 10; ++i) {
    // CHECK: call void @__kmpc_for_static_init_4(
    // CHECK: call ptr @llvm.stacksave()
    // CHECK: store float 0.000000e+00, ptr %
    // CHECK: store double 0.000000e+00, ptr [[B_PRIV_ADDR:%.+]],
    // CHECK: br label %[[DISPATCH:[^,]+]]

    // Skip the before scan body.
    // CHECK: call void @{{.+}}foo{{.+}}()

    // CHECK: [[EXIT_INSCAN:[^,]+]]:

    // a_buffer[i][0..n] = a_priv[[0..n];
    // CHECK: [[BASE_IDX_I:%.+]] = load i32, ptr [[IV_ADDR:%.+]],
    // CHECK: [[BASE_IDX:%.+]] = zext i32 [[BASE_IDX_I]] to i64
    // CHECK: [[IDX:%.+]] = mul nsw i64 [[BASE_IDX]], [[NUM_ELEMS:%.+]]
    // CHECK: [[A_BUF_IDX:%.+]] = getelementptr inbounds float, ptr [[A_BUF:%.+]], i64 [[IDX]]
    // CHECK: [[A_PRIV:%.+]] = getelementptr inbounds [10 x float], ptr [[A_PRIV_ADDR:%.+]], i64 0, i64 0
    // CHECK: [[BYTES:%.+]] = mul nuw i64 [[NUM_ELEMS:%.+]], 4
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}[[A_BUF_IDX]], ptr {{.*}}[[A_PRIV]], i64 [[BYTES]], i1 false)

    // b_buffer[i] = b_priv;
    // CHECK: [[B_BUF_IDX:%.+]] = getelementptr inbounds double, ptr [[B_BUF:%.+]], i64 [[BASE_IDX]]
    // CHECK: [[B_PRIV:%.+]] = load double, ptr [[B_PRIV_ADDR]],
    // CHECK: store double [[B_PRIV]], ptr [[B_BUF_IDX]],
    // CHECK: br label %[[LOOP_CONTINUE:[^,]+]]

    // CHECK: [[DISPATCH]]:
    // CHECK: br label %[[INPUT_PHASE:[^,]+]]

    // CHECK: [[INPUT_PHASE]]:
    // CHECK: call void @{{.+}}bar{{.+}}()
    // CHECK: br label %[[EXIT_INSCAN]]

    // CHECK: [[LOOP_CONTINUE]]:
    // CHECK: call void @llvm.stackrestore(ptr %
    // CHECK: call void @__kmpc_for_static_fini(
    // CHECK: call void @__kmpc_barrier(
    foo();
#pragma omp scan exclusive(a[:n], b)
    // CHECK: [[LOG2_10:%.+]] = call double @llvm.log2.f64(double 1.000000e+01)
    // CHECK: [[CEIL_LOG2_10:%.+]] = call double @llvm.ceil.f64(double [[LOG2_10]])
    // CHECK: [[CEIL_LOG2_10_INT:%.+]] = fptoui double [[CEIL_LOG2_10]] to i32
    // CHECK: br label %[[OUTER_BODY:[^,]+]]
    // CHECK: [[OUTER_BODY]]:
    // CHECK: [[K:%.+]] = phi i32 [ 0, %{{.+}} ], [ [[K_NEXT:%.+]], %{{.+}} ]
    // CHECK: [[K2POW:%.+]] = phi i64 [ 1, %{{.+}} ], [ [[K2POW_NEXT:%.+]], %{{.+}} ]
    // CHECK: [[CMP:%.+]] = icmp uge i64 9, [[K2POW]]
    // CHECK: br i1 [[CMP]], label %[[INNER_BODY:[^,]+]], label %[[INNER_EXIT:[^,]+]]
    // CHECK: [[INNER_BODY]]:
    // CHECK: [[I:%.+]] = phi i64 [ 9, %[[OUTER_BODY]] ], [ [[I_PREV:%.+]], %{{.+}} ]

    // a_buffer[i] += a_buffer[i-pow(2, k)];
    // CHECK: [[IDX:%.+]] = mul nsw i64 [[I]], [[NUM_ELEMS]]
    // CHECK: [[A_BUF_IDX:%.+]] = getelementptr inbounds float, ptr [[A_BUF]], i64 [[IDX]]
    // CHECK: [[IDX_SUB_K2POW:%.+]] = sub nuw i64 [[I]], [[K2POW]]
    // CHECK: [[IDX:%.+]] = mul nsw i64 [[IDX_SUB_K2POW]], [[NUM_ELEMS]]
    // CHECK: [[A_BUF_IDX_SUB_K2POW:%.+]] = getelementptr inbounds float, ptr [[A_BUF]], i64 [[IDX]]
    // CHECK: [[B_BUF_IDX:%.+]] = getelementptr inbounds double, ptr [[B_BUF]], i64 [[I]]
    // CHECK: [[IDX_SUB_K2POW:%.+]] = sub nuw i64 [[I]], [[K2POW]]
    // CHECK: [[B_BUF_IDX_SUB_K2POW:%.+]] = getelementptr inbounds double, ptr [[B_BUF]], i64 [[IDX_SUB_K2POW]]
    // CHECK: [[A_BUF_END:%.+]] = getelementptr float, ptr [[A_BUF_IDX]], i64 [[NUM_ELEMS]]
    // CHECK: [[ISEMPTY:%.+]] = icmp eq ptr [[A_BUF_IDX]], [[A_BUF_END]]
    // CHECK: br i1 [[ISEMPTY]], label %[[RED_DONE:[^,]+]], label %[[RED_BODY:[^,]+]]
    // CHECK: [[RED_BODY]]:
    // CHECK: [[A_BUF_IDX_SUB_K2POW_ELEM:%.+]] = phi ptr [ [[A_BUF_IDX_SUB_K2POW]], %[[INNER_BODY]] ], [ [[A_BUF_IDX_SUB_K2POW_NEXT:%.+]], %[[RED_BODY]] ]
    // CHECK: [[A_BUF_IDX_ELEM:%.+]] = phi ptr [ [[A_BUF_IDX]], %[[INNER_BODY]] ], [ [[A_BUF_IDX_NEXT:%.+]], %[[RED_BODY]] ]
    // CHECK: [[A_BUF_IDX_VAL:%.+]] = load float, ptr [[A_BUF_IDX_ELEM]],
    // CHECK: [[A_BUF_IDX_SUB_K2POW_VAL:%.+]] = load float, ptr [[A_BUF_IDX_SUB_K2POW_ELEM]],
    // CHECK: [[RED:%.+]] = fadd float [[A_BUF_IDX_VAL]], [[A_BUF_IDX_SUB_K2POW_VAL]]
    // CHECK: store float [[RED]], ptr [[A_BUF_IDX_ELEM]],
    // CHECK: [[A_BUF_IDX_NEXT]] = getelementptr float, ptr [[A_BUF_IDX_ELEM]], i32 1
    // CHECK: [[A_BUF_IDX_SUB_K2POW_NEXT]] = getelementptr float, ptr [[A_BUF_IDX_SUB_K2POW_ELEM]], i32 1
    // CHECK: [[DONE:%.+]] = icmp eq ptr [[A_BUF_IDX_NEXT]], [[A_BUF_END]]
    // CHECK: br i1 [[DONE]], label %[[RED_DONE]], label %[[RED_BODY]]
    // CHECK: [[RED_DONE]]:

    // b_buffer[i] += b_buffer[i-pow(2, k)];
    // CHECK: [[B_BUF_IDX_VAL:%.+]] = load double, ptr [[B_BUF_IDX]],
    // CHECK: [[B_BUF_IDX_SUB_K2POW_VAL:%.+]] = load double, ptr [[B_BUF_IDX_SUB_K2POW]],
    // CHECK: [[RED:%.+]] = fadd double [[B_BUF_IDX_VAL]], [[B_BUF_IDX_SUB_K2POW_VAL]]
    // CHECK: store double [[RED]], ptr [[B_BUF_IDX]],

    // --i;
    // CHECK: [[I_PREV:%.+]] = sub nuw i64 [[I]], 1
    // CHECK: [[CMP:%.+]] = icmp uge i64 [[I_PREV]], [[K2POW]]
    // CHECK: br i1 [[CMP]], label %[[INNER_BODY]], label %[[INNER_EXIT]]
    // CHECK: [[INNER_EXIT]]:

    // ++k;
    // CHECK: [[K_NEXT]] = add nuw i32 [[K]], 1
    // k2pow <<= 1;
    // CHECK: [[K2POW_NEXT]] = shl nuw i64 [[K2POW]], 1
    // CHECK: [[CMP:%.+]] = icmp ne i32 [[K_NEXT]], [[CEIL_LOG2_10_INT]]
    // CHECK: br i1 [[CMP]], label %[[OUTER_BODY]], label %[[OUTER_EXIT:[^,]+]]
    // CHECK: [[OUTER_EXIT]]:
    bar();
    // CHECK: call void @__kmpc_for_static_init_4(
    // CHECK: call ptr @llvm.stacksave()
    // CHECK: store float 0.000000e+00, ptr %
    // CHECK: store double 0.000000e+00, ptr [[B_PRIV_ADDR:%.+]],
    // CHECK: br label %[[DISPATCH:[^,]+]]

    // CHECK: [[SCAN_PHASE:.+]]:
    // CHECK: call void @{{.+}}foo{{.+}}()
    // CHECK: br label %[[LOOP_CONTINUE:.+]]

    // CHECK: [[DISPATCH]]:
    // if (i >0)
    //   a_priv[[0..n] = a_buffer[i-1][0..n];
    // CHECK: [[BASE_IDX_I:%.+]] = load i32, ptr [[IV_ADDR:%.+]],
    // CHECK: [[BASE_IDX:%.+]] = zext i32 [[BASE_IDX_I]] to i64
    // CHECK: [[CMP:%.+]] = icmp eq i64 [[BASE_IDX]], 0
    // CHECK: br i1 [[CMP]], label %[[IF_DONE:[^,]+]], label %[[IF_THEN:[^,]+]]
    // CHECK: [[IF_THEN]]:
    // CHECK: [[BASE_IDX_SUB_1:%.+]] = sub nuw i64 [[BASE_IDX]], 1
    // CHECK: [[IDX:%.+]] = mul nsw i64 [[BASE_IDX_SUB_1]], [[NUM_ELEMS]]
    // CHECK: [[A_BUF_IDX:%.+]] = getelementptr inbounds float, ptr [[A_BUF]], i64 [[IDX]]
    // CHECK: [[A_PRIV:%.+]] = getelementptr inbounds [10 x float], ptr [[A_PRIV_ADDR:%.+]], i64 0, i64 0
    // CHECK: [[BYTES:%.+]] = mul nuw i64 [[NUM_ELEMS:%.+]], 4
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}[[A_PRIV]], ptr {{.*}}[[A_BUF_IDX]], i64 [[BYTES]], i1 false)

    // b_priv = b_buffer[i];
    // CHECK: [[B_BUF_IDX:%.+]] = getelementptr inbounds double, ptr [[B_BUF]], i64 [[BASE_IDX_SUB_1]]
    // CHECK: [[B_BUF_IDX_VAL:%.+]] = load double, ptr [[B_BUF_IDX]],
    // CHECK: store double [[B_BUF_IDX_VAL]], ptr [[B_PRIV_ADDR]],
    // CHECK: br label %[[SCAN_PHASE]]

    // CHECK: [[LOOP_CONTINUE]]:
    // CHECK: call void @llvm.stackrestore(ptr %
    // CHECK: call void @__kmpc_for_static_fini(
  }
}

// CHECK: !{!"llvm.loop.vectorize.enable", i1 true}

#endif

