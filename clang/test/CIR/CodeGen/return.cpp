// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

int &ret0(int &x) { 
  return x;
}

// CHECK: cir.func @_Z4ret0Ri
// CHECK:   %0 = cir.alloca !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>, ["x", init] {alignment = 8 : i64}
// CHECK:   %1 = cir.alloca !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>, ["__retval"] {alignment = 8 : i64}
// CHECK:   cir.store %arg0, %0 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>
// CHECK:   %2 = cir.load %0 : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK:   cir.store %2, %1 : !cir.ptr<i32>, cir.ptr <!cir.ptr<i32>>
// CHECK:   %3 = cir.load %1 : cir.ptr <!cir.ptr<i32>>, !cir.ptr<i32>
// CHECK:   cir.return %3 : !cir.ptr<i32>
