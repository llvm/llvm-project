// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=SIMD-ONLY
// expected-no-diagnostics

// SIMD-ONLY-NOT: {{__kmpc|__tgt}}

// A 'uval' item is read and written back through the reference itself, not
// through its initializer (absent for a parameter, uncaptured in a region).

// CHECK-LABEL: define {{.*}}void @{{.*}}ref_param
// CHECK-SAME: (ptr noundef nonnull align 4 dereferenceable(4) [[X:%.+]], ptr noundef nonnull align 4 dereferenceable(4) [[Y:%.+]])
void ref_param(int &x, int &y) {
// CHECK: store ptr [[X]], ptr [[X_ADDR:%[^,]+]],
// CHECK: [[X_REF:%.+]] = load ptr, ptr [[X_ADDR]],
// CHECK-NEXT: store ptr [[X_REF]], ptr [[X_TMP:%[^,]+]],
// CHECK: store i32 0, ptr [[OMP_IV:%[^,]+]],
// CHECK-NEXT: [[X_REF1:%.+]] = load ptr, ptr [[X_TMP]],
// CHECK-NEXT: [[X_START:%.+]] = load i32, ptr [[X_REF1]],
// CHECK-NEXT: store i32 [[X_START]], ptr [[LINEAR_START:%[^,]+]],
// CHECK-NEXT: [[Y_REF:%.+]] = load ptr, ptr
// CHECK-NEXT: [[Y_VAL:%.+]] = load i32, ptr [[Y_REF]],
// CHECK-NEXT: [[STEP:%.+]] = add nsw i32 [[Y_VAL]], 1
// CHECK-NEXT: store i32 [[STEP]], ptr [[LINEAR_STEP:%[^,]+]],
#pragma omp simd linear(uval(x) : y + 1)
  for (int i = 0; i < 10; ++i)
    x += y + 1;
// CHECK: [[START:%.+]] = load i32, ptr [[LINEAR_START]],
// CHECK-NEXT: [[IV:%.+]] = load i32, ptr [[OMP_IV]],
// CHECK-NEXT: [[STEP_VAL:%.+]] = load i32, ptr [[LINEAR_STEP]],
// CHECK-NEXT: [[MUL:%.+]] = mul nsw i32 [[IV]], [[STEP_VAL]]
// CHECK-NEXT: [[ADD:%.+]] = add nsw i32 [[START]], [[MUL]]
// CHECK-NEXT: store i32 [[ADD]], ptr [[X_PRIV:%[^,]+]],
// CHECK: omp.inner.for.end:
// CHECK: [[X_REF2:%.+]] = load ptr, ptr [[X_TMP]],
// CHECK-NEXT: store ptr [[X_REF2]], ptr [[X_TMP1:%[^,]+]],
// CHECK-NEXT: [[X_FINAL:%.+]] = load i32, ptr [[X_PRIV]],
// CHECK-NEXT: [[X_REF3:%.+]] = load ptr, ptr [[X_TMP1]],
// CHECK-NEXT: store i32 [[X_FINAL]], ptr [[X_REF3]],
// CHECK-NEXT: ret void
}

// CHECK-LABEL: define {{.*}}void @{{.*}}local_ref{{.*}}()
void local_ref() {
  int a = 0;
  int &r = a;
// CHECK: [[R_PTR:%.+]] = load ptr, ptr %r,
// CHECK-NEXT: call void {{.*}}@__kmpc_fork_call({{.*}}, ptr @[[OUTLINED:.+]], ptr [[R_PTR]])
#pragma omp parallel for simd linear(uval(r))
  for (int i = 0; i < 10; ++i)
    r += 1;
}

// CHECK: define internal void @[[OUTLINED]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}}, ptr noundef nonnull align 4 dereferenceable(4) [[R:%.+]])
// CHECK: store ptr [[R]], ptr [[R_ADDR:%[^,]+]],
// CHECK-NEXT: [[R_REF:%.+]] = load ptr, ptr [[R_ADDR]],
// CHECK-NEXT: store ptr [[R_REF]], ptr [[R_TMP:%[^,]+]],
// CHECK-NEXT: [[R_REF1:%.+]] = load ptr, ptr [[R_TMP]],
// CHECK-NEXT: [[R_START:%.+]] = load i32, ptr [[R_REF1]],
// CHECK-NEXT: store i32 [[R_START]], ptr [[LINEAR_START:%[^,]+]],
// CHECK: [[START:%.+]] = load i32, ptr [[LINEAR_START]],
// CHECK-NEXT: [[IV:%.+]] = load i32, ptr
// CHECK-NEXT: [[MUL:%.+]] = mul nsw i32 [[IV]], 1
// CHECK-NEXT: [[ADD:%.+]] = add nsw i32 [[START]], [[MUL]]
// CHECK-NEXT: store i32 [[ADD]], ptr [[R_PRIV:%[^,]+]],
// CHECK: .omp.linear.pu:
// CHECK-NEXT: [[R_REF2:%.+]] = load ptr, ptr [[R_TMP]],
// CHECK-NEXT: store ptr [[R_REF2]], ptr [[R_TMP1:%[^,]+]],
// CHECK-NEXT: [[R_FINAL:%.+]] = load i32, ptr [[R_PRIV]],
// CHECK-NEXT: [[R_REF3:%.+]] = load ptr, ptr [[R_TMP1]],
// CHECK-NEXT: store i32 [[R_FINAL]], ptr [[R_REF3]],
