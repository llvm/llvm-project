// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

inline const int &ref = 42;

const int *use() { return &ref; }

// CIR: cir.global constant linkonce_odr comdat @_ZGR3ref_ = #cir.int<42> : !s32i
// CIR: cir.func{{.*}} @_Z3usev()
// CIR: cir.const #cir.global_view<@_ZGR3ref_> : !cir.ptr<!s32i>

// LLVM: $_ZGR3ref_ = comdat any
// LLVM: @_ZGR3ref_ = linkonce_odr constant i32 42, comdat, align 4
