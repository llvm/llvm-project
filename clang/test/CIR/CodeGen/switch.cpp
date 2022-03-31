// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void sw0(int a) {
  switch (a) {}
}

// CHECK: cir.scope {
// CHECK-NEXT:   %1 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT:   cir.switch (%1 : i32) [
// CHECK-NEXT:   ]
// CHECK-NEXT: }
