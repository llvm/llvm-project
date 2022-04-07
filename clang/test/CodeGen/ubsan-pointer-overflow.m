// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -w -emit-llvm -o - %s -fsanitize=pointer-overflow | FileCheck %s

// CHECK-LABEL: define{{.*}} void @variable_len_array_arith
void variable_len_array_arith(int n, int k) {
  int vla[n];
  int (*p)[n] = &vla;

  // CHECK: getelementptr inbounds i32, i32* {{.*}}, i64 [[INC:%.*]]
  // CHECK: @llvm.smul.with.overflow.i64(i64 4, i64 [[INC]]), !nosanitize
  // CHECK-NOT: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  ++p;

  // CHECK: getelementptr inbounds i32, i32* {{.*}}, i64 [[IDXPROM:%.*]]
  // CHECK: @llvm.smul.with.overflow.i64(i64 4, i64 [[IDXPROM]]), !nosanitize
  // CHECK: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  p + k;
}

// CHECK-LABEL: define{{.*}} void @objc_id
void objc_id(id *p) {
  // CHECK: add i64 {{.*}}, 8, !nosanitize
  // CHECK-NOT: select
  // CHECK: @__ubsan_handle_pointer_overflow{{.*}}
  p++;
}
