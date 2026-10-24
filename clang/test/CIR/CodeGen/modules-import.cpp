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

import m;

int use_it() { return f(); }

// CIR-LABEL: cir.func {{.*}} @_Z6use_itv(
// CIR:         {{%.+}} = cir.call @_ZW1m1fv() : () -> (!s32i {{.*}})

// LLVM-LABEL: define {{.*}} i32 @_Z6use_itv()
// LLVM:         {{%.+}} = call {{.*}} i32 @_ZW1m1fv()
