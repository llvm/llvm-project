// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct V1 { virtual void v1(); };
struct V2 { virtual void v2(); };
struct V3 { virtual void v3(); };

struct B : V1, V2 {};

struct D : V3, B {};

constinit D d{};
// CIR: cir.global external @d = #cir.const_record<{#cir.const_record<{#cir.global_view<@_ZTV1D, [0 : i32, 2 : i32]> : !cir.vptr}> : !rec_V3, #cir.const_record<{#cir.const_record<{#cir.global_view<@_ZTV1D, [1 : i32, 2 : i32]> : !cir.vptr}> : !rec_V1, #cir.const_record<{#cir.global_view<@_ZTV1D, [2 : i32, 2 : i32]> : !cir.vptr}> : !rec_V2}> : !rec_B}> : !rec_D
// LLVM: @d = global %struct.D { %struct.V3 { ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 16) }, %struct.B { %struct.V1 { ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 40) }, %struct.V2 { ptr getelementptr inbounds nuw (i8, ptr @_ZTV1D, i64 64) } } }
// OGCG: @d = global { ptr, ptr, ptr } { ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr], [3 x ptr], [3 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 2), ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr], [3 x ptr], [3 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 2), ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr], [3 x ptr], [3 x ptr] }, ptr @_ZTV1D, i32 0, i32 2, i32 2) }


struct Base {
  int i;
  char c;
  constexpr Base(int i, char c) : i(i), c(c) {}
};
struct Derived : Base {
  char d;
  constexpr Derived(int i, char c, char d) : Base(i, c), d(d) {}
};

Derived gd {1, 2, 3};
// CIR: cir.global external @gd = #cir.const_record<{#cir.const_record<{#cir.int<1> : !s32i, #cir.int<2> : !s8i}> : !rec_Base2Ebase, #cir.int<3> : !s8i, #cir.zero : !cir.array<!u8i x 2>}> : !rec_Derived
// LLVM: @gd = global %struct.Derived { %struct.Base.base <{ i32 1, i8 2 }>, i8 3, [2 x i8] zeroinitializer }
// OGCG: @gd = global { i32, i8, i8 } { i32 1, i8 2, i8 3 }

