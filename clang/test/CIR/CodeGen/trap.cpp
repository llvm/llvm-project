// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void foo();

void basic() {
  foo();
  __builtin_trap();
}

//      CHECK: cir.func @_Z5basicv()
// CHECK-NEXT:   cir.call @_Z3foov() : () -> ()
// CHECK-NEXT:   cir.trap
// CHECK-NEXT: }

void code_after_unreachable() {
  foo();
  __builtin_trap();
  foo();
}

//      CHECK: cir.func @_Z22code_after_unreachablev()
// CHECK-NEXT:   cir.call @_Z3foov() : () -> ()
// CHECK-NEXT:   cir.trap
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   cir.call @_Z3foov() : () -> ()
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
