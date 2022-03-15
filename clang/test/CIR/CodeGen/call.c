// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

void a(void) {}
void b(int a) {}

void c(void) {
  a();
  b(0);
}

// CHECK: module  {
// CHECK:   func @a() {
// CHECK:     cir.return
// CHECK:   }
// CHECK:   func @b(%arg0: i32 {{.*}} {
// CHECK:     %0 = cir.alloca i32, cir.ptr <i32>, ["a", paraminit]
// CHECK:     cir.store %arg0, %0 : i32, cir.ptr <i32>
// CHECK:     cir.return
// CHECK:   }
// CHECK:   func @c() {
// CHECK:     call @a() : () -> ()
// CHECK:     %0 = cir.cst(0 : i32) : i32
// CHECK:     call @b(%0) : (i32) -> ()
// CHECK:     cir.return
// CHECK:   }
