// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int x(int y) {
  return y > 0 ? 3 : 5;
}

// CHECK: cir.func @_Z1xi
// CHECK:     %0 = cir.alloca i32, cir.ptr <i32>, ["y", init] {alignment = 4 : i64}
// CHECK:     %1 = cir.alloca i32, cir.ptr <i32>, ["__retval"] {alignment = 4 : i64}
// CHECK:     cir.store %arg0, %0 : i32, cir.ptr <i32>
// CHECK:     %2 = cir.load %0 : cir.ptr <i32>, i32
// CHECK:     %3 = cir.const(0 : i32) : i32
// CHECK:     %4 = cir.cmp(gt, %2, %3) : i32, !cir.bool
// CHECK:     %5 = cir.ternary(%4, true {
// CHECK:       %7 = cir.const(3 : i32) : i32
// CHECK:       cir.yield %7 : i32
// CHECK:     }, false {
// CHECK:       %7 = cir.const(5 : i32) : i32
// CHECK:       cir.yield %7 : i32
// CHECK:     }) : i32
// CHECK:     cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK:     %6 = cir.load %1 : cir.ptr <i32>, i32
// CHECK:     cir.return %6 : i32
// CHECK:   }