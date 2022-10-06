// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

unsigned char cxxstaticcast_0(unsigned int x) {
  return static_cast<unsigned char>(x);
}

// CHECK: cir.func @_Z15cxxstaticcast_0j
// CHECK:    %0 = cir.alloca i32, cir.ptr <i32>, ["x", init] {alignment = 4 : i64}
// CHECK:    %1 = cir.alloca i8, cir.ptr <i8>, ["__retval"] {alignment = 1 : i64}
// CHECK:    cir.store %arg0, %0 : i32, cir.ptr <i32>
// CHECK:    %2 = cir.load %0 : cir.ptr <i32>, i32
// CHECK:    %3 = cir.cast(integral, %2 : i32), i8
// CHECK:    cir.store %3, %1 : i8, cir.ptr <i8>
// CHECK:    %4 = cir.load %1 : cir.ptr <i8>, i8
// CHECK:    cir.return %4 : i8
// CHECK:  }
