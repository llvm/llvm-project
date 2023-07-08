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
// CHECK-NEXT  %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT  %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["b", init] {alignment = 4 : i64}
// CHECK-NEXT  cir.store %arg0, %0 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT  %2 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT  cir.store %2, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT  cir.br ^bb2
// CHECK-NEXT ^bb1:  // no predecessors
// CHECK-NEXT   %3 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT   %4 = cir.const(1 : !s32i) : !s32i
// CHECK-NEXT   %5 = cir.binop(add, %3, %4) : !s32i
// CHECK-NEXT   cir.store %5, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT   cir.br ^bb2
// CHECK-NEXT ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT   %6 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT   %7 = cir.const(2 : !s32i) : !s32i
// CHECK-NEXT   %8 = cir.binop(add, %6, %7) : !s32i
// CHECK-NEXT   cir.store %8, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT   cir.return

void g1(int a) {
  int x = 0;
  goto end;
end:
  int y = a + 2;
}

// Make sure alloca for "y" shows up in the entry block
// CHECK: cir.func @_Z2g1i(%arg0: !s32i
// CHECK-NEXT: %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT: %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {alignment = 4 : i64}
// CHECK-NEXT: %2 = cir.alloca !s32i, cir.ptr <!s32i>, ["y", init] {alignment = 4 : i64}
// CHECK-NEXT: cir.store %arg0, %0 : !s32i, cir.ptr <!s32i>

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

// CHECK: cir.func @_Z2g2v() -> !s32i

// CHECK:     cir.br ^bb2
// CHECK-NEXT:   ^bb1:  // no predecessors
// CHECK:   ^bb2:  // 2 preds: ^bb0, ^bb1

// CHECK:     [[R:%[0-9]+]] = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:     [[R]] : !s32i
// CHECK-NEXT:   }
