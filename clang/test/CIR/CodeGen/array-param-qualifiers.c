// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// Qualifiers inside the brackets of an array parameter don't affect the
// lowered type; the parameter is just a pointer to the element type.

int vla_const(int n, int a[const n]) { return a[0]; }
// CIR: cir.func {{.*}} @vla_const({{.*}}: !s32i {{.*}}, {{.*}}: !cir.ptr<!s32i> {{.*}})
// LLVM: define {{.*}} i32 @vla_const(i32 {{.*}}, ptr {{.*}})

int incomplete_const(int a[const]) { return a[0]; }
// CIR: cir.func {{.*}} @incomplete_const({{.*}}: !cir.ptr<!s32i> {{.*}})
// LLVM: define {{.*}} i32 @incomplete_const(ptr {{.*}})

int incomplete_volatile_restrict(int a[volatile restrict]) { return a[0]; }
// CIR: cir.func {{.*}} @incomplete_volatile_restrict({{.*}}: !cir.ptr<!s32i> {{.*}})
// LLVM: define {{.*}} i32 @incomplete_volatile_restrict(ptr {{.*}})
