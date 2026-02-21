// RUN: %clang_cc1 -triple aarch64-linux -fexperimental-pointer-field-protection-abi -fexperimental-pointer-field-protection-tagged -emit-llvm -o - %s | FileCheck %s


struct S {
  int* ptr1;
  __attribute__((no_field_protection)) int* ptr2;
private:
  int private_data;
};  // Not Standard-layout, mixed access

// CHECK-LABEL: load_pointers_without_no_field_protection
int* load_pointers_without_no_field_protection(S *t) {
  return t->ptr1;
}
// CHECK: call {{.*}} @llvm.protected.field.ptr.p0{{.*}}

// CHECK-LABEL: load_pointers_with_no_field_protection
int* load_pointers_with_no_field_protection(S *t) {
  return t->ptr2;
}
// CHECK-NOT: call {{.*}} @llvm.protected.field.ptr.p0{{.*}}

// CHECK-LABEL: store_pointers_without_no_field_protection
void store_pointers_without_no_field_protection(S *t, int *input) {
  t->ptr1 = input;
}
// CHECK: call {{.*}} @llvm.protected.field.ptr.p0{{.*}}

// CHECK-LABEL: store_pointers_with_no_field_protection
void store_pointers_with_no_field_protection(S *t, int *input) {
  t->ptr2 = input;
}
// CHECK-NOT: call {{.*}} @llvm.protected.field.ptr.p0{{.*}}
