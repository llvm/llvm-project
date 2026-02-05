// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-cir -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=DEFAULT-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-cir -o %t-wrapv.cir -fwrapv
// RUN: FileCheck --input-file=%t-wrapv.cir %s -check-prefix=WRAPV-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-llvm -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=DEFAULT-LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-llvm -o %t-wrapv-cir.ll -fwrapv
// RUN: FileCheck --input-file=%t-wrapv-cir.ll %s -check-prefix=WRAPV-LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=DEFAULT-OGCG
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o %t-wrapv.ll -fwrapv
// RUN: FileCheck --input-file=%t-wrapv.ll %s -check-prefix=WRAPV-OGCG

// TODO(cir): These tests are copied from clang/test/CodeGen/integer-overflow.c.
// When support for sanitizers is implemented, it this test should be updated
// to add sanitizer checks.

// Tests for signed integer overflow stuff.
void test1(void) {
  extern volatile int f11G, a, b;
  
  // DEFAULT-CIR:  cir.binop(add, {{.*}}, {{.*}}) nsw : !s32i
  // DEFAULT-LLVM: add nsw i32
  // DEFAULT-OGCG: add nsw i32
  // WRAPV-CIR:  cir.binop(add, {{.*}}, {{.*}}) : !s32i
  // WRAPV-LLVM: add i32
  // WRAPV-OGCG: add i32
  f11G = a + b;
  
  // DEFAULT-CIR:  cir.binop(sub, {{.*}}, {{.*}}) nsw : !s32i
  // DEFAULT-LLVM: sub nsw i32
  // DEFAULT-OGCG: sub nsw i32
  // WRAPV-CIR:  cir.binop(sub, {{.*}}, {{.*}}) : !s32i
  // WRAPV-LLVM: sub i32
  // WRAPV-OGCG: sub i32
  f11G = a - b;
  
  // DEFAULT-CIR:  cir.binop(mul, {{.*}}, {{.*}}) nsw : !s32i
  // DEFAULT-LLVM: mul nsw i32
  // DEFAULT-OGCG: mul nsw i32
  // WRAPV-CIR:  cir.binop(mul, {{.*}}, {{.*}}) : !s32i
  // WRAPV-LLVM: mul i32
  // WRAPV-OGCG: mul i32
  f11G = a * b;

  // DEFAULT-CIR:  cir.unary(minus, {{.*}}) nsw : !s32i
  // DEFAULT-LLVM: sub nsw i32 0, 
  // DEFAULT-OGCG: sub nsw i32 0, 
  // WRAPV-CIR:  cir.unary(minus, {{.*}}) : !s32i
  // WRAPV-LLVM: sub i32 0, 
  // WRAPV-OGCG: sub i32 0, 
  f11G = -a;
  
  // PR7426 - Overflow checking for increments.
  
  // DEFAULT-CIR:  cir.unary(inc, {{.*}}) nsw : !s32i
  // DEFAULT-LLVM: add nsw i32 {{.*}}, 1
  // DEFAULT-OGCG: add nsw i32 {{.*}}, 1
  // WRAPV-CIR:  cir.unary(inc, {{.*}}) : !s32i
  // WRAPV-LLVM: add i32 {{.*}}, 1
  // WRAPV-OGCG: add i32 {{.*}}, 1
  ++a;
  
  // DEFAULT-CIR:  cir.unary(dec, {{.*}}) nsw : !s32i
  // DEFAULT-LLVM: sub nsw i32 {{.*}}, 1
  // DEFAULT-OGCG: add nsw i32 {{.*}}, -1
  // WRAPV-CIR:  cir.unary(dec, {{.*}}) : !s32i
  // WRAPV-LLVM: sub i32 {{.*}}, 1
  // WRAPV-OGCG: add i32 {{.*}}, -1
  --a;
  
  // -fwrapv does not affect inbounds for GEP's.
  // This is controlled by -fwrapv-pointer instead.
  extern int* P;
  ++P;
  // DEFAULT-CIR:  %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // DEFAULT-CIR:  cir.ptr_stride {{.*}}, %[[ONE]]
  // DEFAULT-LLVM: getelementptr i32, ptr %{{.*}}, i64 1
  // DEFAULT-OGCG: getelementptr inbounds nuw i32, ptr
  // WRAPV-CIR:  %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // WRAPV-CIR:  cir.ptr_stride {{.*}}, %[[ONE]]
  // WRAPV-LLVM: getelementptr i32, ptr %{{.*}}, i64 1
  // WRAPV-OGCG: getelementptr inbounds nuw i32, ptr

  // PR9350: char pre-increment never overflows.
  extern volatile signed char PR9350_char_inc;
  // DEFAULT-CIR:  cir.unary(inc, {{.*}}) : !s8i
  // DEFAULT-LLVM: add i8 {{.*}}, 1
  // DEFAULT-OGCG: add i8 {{.*}}, 1
  // WRAPV-CIR:  cir.unary(inc, {{.*}}) : !s8i
  // WRAPV-LLVM: add i8 {{.*}}, 1
  // WRAPV-OGCG: add i8 {{.*}}, 1
  ++PR9350_char_inc;

  // PR9350: char pre-decrement never overflows.
  extern volatile signed char PR9350_char_dec;
  // DEFAULT-CIR:  cir.unary(dec, {{.*}}) : !s8i
  // DEFAULT-LLVM: sub i8 {{.*}}, 1
  // DEFAULT-OGCG: add i8 {{.*}}, -1
  // WRAPV-CIR:  cir.unary(dec, {{.*}}) : !s8i
  // WRAPV-LLVM: sub i8 {{.*}}, 1
  // WRAPV-OGCG: add i8 {{.*}}, -1
  --PR9350_char_dec;

  // PR9350: short pre-increment never overflows.
  extern volatile signed short PR9350_short_inc;
  // DEFAULT-CIR:  cir.unary(inc, {{.*}}) : !s16i
  // DEFAULT-LLVM: add i16 {{.*}}, 1
  // DEFAULT-OGCG: add i16 {{.*}}, 1
  // WRAPV-CIR:  cir.unary(inc, {{.*}}) : !s16i
  // WRAPV-LLVM: add i16 {{.*}}, 1
  // WRAPV-OGCG: add i16 {{.*}}, 1
  ++PR9350_short_inc;

  // PR9350: short pre-decrement never overflows.
  extern volatile signed short PR9350_short_dec;
  // DEFAULT-CIR:  cir.unary(dec, {{.*}}) : !s16i
  // DEFAULT-LLVM: sub i16 {{.*}}, 1
  // DEFAULT-OGCG: add i16 {{.*}}, -1
  // WRAPV-CIR:  cir.unary(dec, {{.*}}) : !s16i
  // WRAPV-LLVM: sub i16 {{.*}}, 1
  // WRAPV-OGCG: add i16 {{.*}}, -1
  --PR9350_short_dec;
}
