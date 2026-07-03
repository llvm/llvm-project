// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

void callee(void);

void caller(void) { callee(); }

// Plain C functions carry no identity attribute. Their symbol already is
// the plain name, since C names have no mangling.
// CHECK: cir.func{{.*}} @caller
// CHECK-NOT: func_info
