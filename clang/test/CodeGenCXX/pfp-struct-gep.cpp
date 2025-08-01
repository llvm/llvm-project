// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-NOPFP
// RUN: %clang_cc1  -fexperimental-pointer-field-protection=tagged -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-PFP


struct S {
  int* ptr;
private:
  int private_data;
};  // Not Standard-layout, mixed access

// CHECK-LABEL: load_pointers
int* load_pointers(S *t) {
   return t->ptr;
}
// CHECK-PFP: call {{.*}} @llvm.protected.field.ptr({{.*}}, i64 0, metadata !"_ZTS1S.ptr", i1 false)
// CHECK-NOPFP: getelementptr

// CHECK-LABEL: store_pointers
void store_pointers(S* t, int* p) {
  t->ptr = p;
}
// CHECK-PFP: call {{.*}} @llvm.protected.field.ptr({{.*}}, i64 0, metadata !"_ZTS1S.ptr", i1 false)
// CHECK-NOPFP: getelementptr


