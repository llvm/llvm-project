// NOTE: Do not autogenerate. Check the shared constructor list rather than unrelated function bodies.
// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-linux-gnu -x hip -fgpu-rdc -cuid=abc -fprofile-instrument=clang -emit-llvm -o - %s | FileCheck %s --check-prefix=INIT --implicit-check-not=@llvm.global_ctors.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -x hip -fgpu-rdc -cuid=abc -fprofile-instrument=clang -DNO_INIT -emit-llvm -o - %s | FileCheck %s --check-prefix=NO-INIT --implicit-check-not=@llvm.global_ctors.
// Object emission must accept the combined constructor list.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -x hip -fgpu-rdc -cuid=abc -fprofile-instrument=clang -emit-obj -o %t %s

#ifndef NO_INIT
int init();
int g = init();
#endif

// INIT: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] {{.*}}ptr @_GLOBAL__sub_I_{{[^,]+}}, ptr null{{.*}}ptr @__llvm_profile_register_shadow.{{[^,]+}}, ptr null
// NO-INIT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] {{.*}}ptr @__llvm_profile_register_shadow.{{[^,]+}}, ptr null
