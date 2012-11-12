// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

// CHECK-NOT: i32 6
struct QVectorTypedData {
    int size;
    unsigned int sharable : 1;
    unsigned short array[1];
};

void foo(QVectorTypedData *X) {
  X->array[0] = 123;
}
