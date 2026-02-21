// RUN: %clang_cc1 -triple aarch64-linux -fexperimental-pointer-field-protection-abi -fexperimental-pointer-field-protection-tagged -emit-llvm -O1 -o - %s | FileCheck %s

int val;

struct Pointer {
  int* ptr;
private:
  int private_data;
};

struct ArrayType {
  int* array[3];
private:
  int private_data;
};

struct Array {
  ArrayType array;
private:
  int private_data;
};

struct Struct {
  Pointer ptr;
};

// CHECK-LABEL: test_pointer
Pointer test_pointer(Pointer t) {
    t.ptr = &val;
    return t;
}
// CHECK: call {{.*}} @llvm.protected.field.ptr.p0{{.*}}



// CHECK-LABEL: test_struct
int* test_struct(Struct *t) {
  return (t->ptr).ptr;
}
// CHECK: call {{.*}} @llvm.protected.field.ptr.p0{{.*}}
