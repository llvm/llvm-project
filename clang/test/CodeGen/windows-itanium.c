// RUN: %clang_cc1 -triple i686-windows-itanium -emit-llvm %s -o - \
// RUN:    | FileCheck %s -check-prefix CHECK-C -check-prefix CHECK

// RUN: %clang_cc1 -triple i686-windows-itanium -emit-llvm -x c++ %s -o - \
// RUN:    | FileCheck %s -check-prefix CHECK-CXX -check-prefix CHECK

int function() {
  return 32;
}

// CHECK-C: define dso_local i32 @function() {{.*}} {
// CHECK-CXX: define dso_local noundef i32 @_Z8functionv() {{.*}} {
// CHECK:   ret i32 32
// CHECK: }

