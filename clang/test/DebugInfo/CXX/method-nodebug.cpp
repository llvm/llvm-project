// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

class C {
  void present();
  void absent() __attribute__((nodebug));
};

C c;

// CHECK-NOT: name: "absent"
// CHECK:     name: "present"
// CHECK-NOT: name: "absent"
