// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void g0(int a) {
  int b = a;
  goto end;
  b = b + 1;
end:
  b = b + 2;
}

// CHECK:   cir.func @_Z2g0i
// CHECK-NEXT  %0 = cir.alloca i32, cir.ptr <i32>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT  %1 = cir.alloca i32, cir.ptr <i32>, ["b", init] {alignment = 4 : i64}
// CHECK-NEXT  cir.store %arg0, %0 : i32, cir.ptr <i32>
// CHECK-NEXT  %2 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT  cir.store %2, %1 : i32, cir.ptr <i32>
// CHECK-NEXT  cir.br ^bb2
// CHECK-NEXT ^bb1:  // no predecessors
// CHECK-NEXT   %3 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT   %4 = cir.cst(1 : i32) : i32
// CHECK-NEXT   %5 = cir.binop(add, %3, %4) : i32
// CHECK-NEXT   cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT   cir.br ^bb2
// CHECK-NEXT ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT   %6 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT   %7 = cir.cst(2 : i32) : i32
// CHECK-NEXT   %8 = cir.binop(add, %6, %7) : i32
// CHECK-NEXT   cir.store %8, %1 : i32, cir.ptr <i32>
// CHECK-NEXT   cir.return

void g1(int a) {
  int x = 0;
  goto end;
end:
  int y = a + 2;
}

// Make sure alloca for "y" shows up in the entry block
// CHECK: cir.func @_Z2g1i(%arg0: i32
// CHECK-NEXT: %0 = cir.alloca i32, cir.ptr <i32>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT: %1 = cir.alloca i32, cir.ptr <i32>, ["x", init] {alignment = 4 : i64}
// CHECK-NEXT: %2 = cir.alloca i32, cir.ptr <i32>, ["y", init] {alignment = 4 : i64}
// CHECK-NEXT: cir.store %arg0, %0 : i32, cir.ptr <i32>

int g2() {
  int b = 1;
  goto end;
  b = b + 1;
end:
  b = b + 2;
  return 1;
}

// Make sure (1) we don't get dangling unused cleanup blocks
//           (2) generated returns consider the function type

// CHECK: cir.func @_Z2g2v() -> i32 {

// CHECK:     cir.br ^bb2
// CHECK-NEXT:   ^bb1:  // no predecessors
// CHECK:   ^bb2:  // 2 preds: ^bb0, ^bb1

// CHECK:     [[R:%[0-9]+]] = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT:     [[R]] : i32
// CHECK-NEXT:   }