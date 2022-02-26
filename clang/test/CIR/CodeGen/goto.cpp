// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void g0(int a) {
  int b = a;
  goto end;
  b = b + 1;
end:
  b = b + 2;
}

// CHECK:   func @g0
// CHECK:     %0 = cir.alloca i32, cir.ptr <i32>, ["a", paraminit] {alignment = 4 : i64}
// CHECK:     %1 = cir.alloca i32, cir.ptr <i32>, ["b", cinit] {alignment = 4 : i64}
// CHECK:     cir.store %arg0, %0 : i32, cir.ptr <i32>
// CHECK:     %2 = cir.load %0 : cir.ptr <i32>, i32
// CHECK:     cir.store %2, %1 : i32, cir.ptr <i32>
// CHECK:     cir.br ^bb2
// CHECK:   ^bb1:  // no predecessors
// CHECK:     %3 = cir.load %1 : cir.ptr <i32>, i32
// CHECK:     %4 = cir.cst(1 : i32) : i32
// CHECK:     %5 = cir.binop(add, %3, %4) : i32
// CHECK:     cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK:     cir.br ^bb2
// CHECK:   ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:     %6 = cir.load %1 : cir.ptr <i32>, i32
// CHECK:     %7 = cir.cst(2 : i32) : i32
// CHECK:     %8 = cir.binop(add, %6, %7) : i32
// CHECK:     cir.store %8, %1 : i32, cir.ptr <i32>
// CHECK:     cir.br ^bb3
// CHECK:   ^bb3:  // pred: ^bb2
// CHECK:     cir.return