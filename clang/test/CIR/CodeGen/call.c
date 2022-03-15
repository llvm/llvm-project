// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void a(void) {}

void c(void) {
  a();
}

// CHECK: module  {
// CHECK:   func @a() {
// CHECK:     cir.return
// CHECK:   }
// CHECK:   func @c() {
// CHECK:     call @a() : () -> ()
// CHECK:     cir.return
// CHECK:   }

