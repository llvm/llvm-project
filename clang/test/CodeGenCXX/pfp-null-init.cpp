// RUN: %clang_cc1  -fexperimental-pointer-field-protection=tagged -emit-llvm -o - %s | FileCheck %s

struct S {
  void *p;
private:
  int private_data;
};

// CHECK-LABEL: null_init
void null_init() {
  S s{};
}

// Check that the constructor was applied
// CHECK: call void @llvm.memset.{{.*}}
// CHECK: call {{.*}} @llvm.protected.field.ptr({{.*}}, i64 0, metadata !"_ZTS1S.p", i1 false)