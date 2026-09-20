// RUN: %clang_cc1 -triple aarch64-linux -fexperimental-pointer-field-protection-abi -fexperimental-pointer-field-protection-tagged -emit-llvm -o - %s | FileCheck %s

struct S {
  int* ptr;
private:
  int private_data;
};  // Not Standard-layout, mixed access

// CHECK-LABEL: load_pointers
int* load_pointers(S *t) {
  // CHECK: %t.addr = alloca ptr, align 8
  // CHECK: store ptr %t, ptr %t.addr, align 8
  // CHECK: %0 = load ptr, ptr %t.addr, align 8
  // CHECK: %ptr = getelementptr inbounds nuw %struct.S, ptr %0, i32 0, i32 0
  // CHECK: %1 = call ptr @llvm.protected.field.ptr.p0(ptr %ptr, i64 63261, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS1S.ptr) ]
  // CHECK: %2 = load ptr, ptr %1, align 8
  // CHECK: ret ptr %2
   return t->ptr;
}

// CHECK-LABEL: store_pointers
void store_pointers(S* t, int* p) {
  // CHECK: %t.addr = alloca ptr, align 8
  // CHECK: %p.addr = alloca ptr, align 8
  // CHECK: store ptr %t, ptr %t.addr, align 8
  // CHECK: store ptr %p, ptr %p.addr, align 8
  // CHECK: %0 = load ptr, ptr %p.addr, align 8
  // CHECK: %1 = load ptr, ptr %t.addr, align 8
  // CHECK: %ptr = getelementptr inbounds nuw %struct.S, ptr %1, i32 0, i32 0
  // CHECK: %2 = call ptr @llvm.protected.field.ptr.p0(ptr %ptr, i64 63261, i1 true) [ "deactivation-symbol"(ptr @__pfp_ds__ZTS1S.ptr) ]
  // CHECK: store ptr %0, ptr %2, align 8
  t->ptr = p;
}
