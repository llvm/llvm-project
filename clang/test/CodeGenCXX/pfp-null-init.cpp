// RUN: %clang_cc1 -triple aarch64-linux -fexperimental-pointer-field-protection-abi -fexperimental-pointer-field-protection-tagged -emit-llvm -o - %s | FileCheck %s

struct S {
  void *p;
private:
  int private_data;
};

// CHECK-LABEL: null_init
void null_init() {
  // Check that null initialization was correctly applied to the pointer field.
  // CHECK: %s = alloca %struct.S, align 8
  // CHECK: call void @llvm.memset.p0.i64(ptr align 8 %s, i8 0, i64 16, i1 false)
  // CHECK: %0 = getelementptr inbounds i8, ptr %s, i64 0
  // CHECK: %1 = call ptr @llvm.protected.field.ptr.p0(ptr %0, i64 29832, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS1S.p) ]
  // CHECK: store ptr null, ptr %1, align 8
  S s{};
}

