// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

void foo(int i) {}

int test(void) {
  foo(2);
  return 0;
}

// CHECK-LABEL: func.func @test() -> i32 {
//       CHECK:   %[[ARG:.+]] = arith.constant 2 : i32
//  CHECK-NEXT:   call @foo(%[[ARG]]) : (i32) -> ()
//       CHECK: }
