// RUN: %clang_cc1  -triple=x86_64-linux-gnu -std=gnu++14 -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck -check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -std=gnu++14 -fclangir -emit-llvm -o %t-cir.ll %s
// RUN: FileCheck -check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -std=gnu++14 -emit-llvm -o %t.ll %s
// RUN: FileCheck -check-prefix=OGCG --input-file=%t.ll %s

struct B {
  virtual void f() {}
  virtual ~B() {}
};
B x;

// CIR: cir.global external @x = #cir.const_record<{#cir.global_view<@_ZTV1B, [0 : i32, 2 : i32]> : !cir.vptr}> : !rec_B
// LLVM: @x = global %struct.B { ptr getelementptr inbounds nuw (i8, ptr @_ZTV1B, i64 16) }, align 8
// OGCG: @x = global %struct.B { ptr getelementptr inbounds inrange(-16, 24) ({ [5 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 2) }, align 8
