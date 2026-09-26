// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-module-interface %S/Inputs/modules-import-iface.cppm -o %t.pcm
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-cir -fmodule-file=m=%t.pcm %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-llvm -fmodule-file=m=%t.pcm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm -fmodule-file=m=%t.pcm %s -o %t.og.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.og.ll %s

module m;

int g() { return f() + 1; }

// CIR-LABEL: cir.func {{.*}} @_ZW1m1gv(
// CIR:         {{%.+}} = cir.call @_ZW1m1fv() : () -> (!s32i {{.*}})

// LLVM-LABEL: define {{.*}} i32 @_ZW1m1gv()
// LLVM:         {{%.+}} = call {{.*}} i32 @_ZW1m1fv()
