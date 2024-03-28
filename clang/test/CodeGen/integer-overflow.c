// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - -fwrapv | FileCheck %s --check-prefix=WRAPV
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - -ftrapv | FileCheck %s --check-prefix=TRAPV
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - -fsanitize=signed-integer-overflow | FileCheck %s --check-prefixes=CATCH_UB,CATCH_UB_POINTER
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - -fsanitize=signed-integer-overflow -fwrapv | FileCheck %s --check-prefixes=CATCH_UB,NOCATCH_UB_POINTER
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - -ftrapv -ftrapv-handler foo | FileCheck %s --check-prefix=TRAPV_HANDLER


// Tests for signed integer overflow stuff.
void test1(void) {
  // DEFAULT-LABEL: define{{.*}} void @test1
  // WRAPV-LABEL: define{{.*}} void @test1
  // TRAPV-LABEL: define{{.*}} void @test1
  extern volatile int f11G, a, b;
  
  // DEFAULT: add nsw i32
  // WRAPV: add i32
  // TRAPV: llvm.sadd.with.overflow.i32
  // CATCH_UB: llvm.sadd.with.overflow.i32
  // TRAPV_HANDLER: foo(
  f11G = a + b;
  
  // DEFAULT: sub nsw i32
  // WRAPV: sub i32
  // TRAPV: llvm.ssub.with.overflow.i32
  // CATCH_UB: llvm.ssub.with.overflow.i32
  // TRAPV_HANDLER: foo(
  f11G = a - b;
  
  // DEFAULT: mul nsw i32
  // WRAPV: mul i32
  // TRAPV: llvm.smul.with.overflow.i32
  // CATCH_UB: llvm.smul.with.overflow.i32
  // TRAPV_HANDLER: foo(
  f11G = a * b;

  // DEFAULT: sub nsw i32 0, 
  // WRAPV: sub i32 0, 
  // TRAPV: llvm.ssub.with.overflow.i32(i32 0
  // CATCH_UB: llvm.ssub.with.overflow.i32(i32 0
  // TRAPV_HANDLER: foo(
  f11G = -a;
  
  // PR7426 - Overflow checking for increments.
  
  // DEFAULT: add nsw i32 {{.*}}, 1
  // WRAPV: add i32 {{.*}}, 1
  // TRAPV: llvm.sadd.with.overflow.i32({{.*}}, i32 1)
  // CATCH_UB: llvm.sadd.with.overflow.i32({{.*}}, i32 1)
  // TRAPV_HANDLER: foo(
  ++a;
  
  // DEFAULT: add nsw i32 {{.*}}, -1
  // WRAPV: add i32 {{.*}}, -1
  // TRAPV: llvm.ssub.with.overflow.i32({{.*}}, i32 1)
  // CATCH_UB: llvm.ssub.with.overflow.i32({{.*}}, i32 1)
  // TRAPV_HANDLER: foo(
  --a;
  
  // -fwrapv should turn off inbounds for GEP's, PR9256
  extern int* P;
  ++P;
  // DEFAULT: getelementptr inbounds i32, ptr
  // WRAPV: getelementptr i32, ptr
  // TRAPV: getelementptr inbounds i32, ptr
  // CATCH_UB_POINTER: getelementptr inbounds i32, ptr
  // NOCATCH_UB_POINTER: getelementptr i32, ptr

  // PR9350: char pre-increment never overflows.
  extern volatile signed char PR9350_char_inc;
  // DEFAULT: add i8 {{.*}}, 1
  // WRAPV: add i8 {{.*}}, 1
  // TRAPV: add i8 {{.*}}, 1
  // CATCH_UB: add i8 {{.*}}, 1
  ++PR9350_char_inc;

  // PR9350: char pre-decrement never overflows.
  extern volatile signed char PR9350_char_dec;
  // DEFAULT: add i8 {{.*}}, -1
  // WRAPV: add i8 {{.*}}, -1
  // TRAPV: add i8 {{.*}}, -1
  // CATCH_UB: add i8 {{.*}}, -1
  --PR9350_char_dec;

  // PR9350: short pre-increment never overflows.
  extern volatile signed short PR9350_short_inc;
  // DEFAULT: add i16 {{.*}}, 1
  // WRAPV: add i16 {{.*}}, 1
  // TRAPV: add i16 {{.*}}, 1
  // CATCH_UB: add i16 {{.*}}, 1
  ++PR9350_short_inc;

  // PR9350: short pre-decrement never overflows.
  extern volatile signed short PR9350_short_dec;
  // DEFAULT: add i16 {{.*}}, -1
  // WRAPV: add i16 {{.*}}, -1
  // TRAPV: add i16 {{.*}}, -1
  // CATCH_UB: add i16 {{.*}}, -1
  --PR9350_short_dec;

  // PR24256: don't instrument __builtin_frame_address.
  __builtin_frame_address(0 + 0);
  // DEFAULT:  call ptr @llvm.frameaddress.p0(i32 0)
  // WRAPV:    call ptr @llvm.frameaddress.p0(i32 0)
  // TRAPV:    call ptr @llvm.frameaddress.p0(i32 0)
  // CATCH_UB: call ptr @llvm.frameaddress.p0(i32 0)
}

// Tests for integer overflow using __attribute__((wraps))
typedef int __attribute__((wraps)) wrapping_int;

void test2(void) {
  // DEFAULT-LABEL: define{{.*}} void @test2
  // WRAPV-LABEL: define{{.*}} void @test2
  // TRAPV-LABEL: define{{.*}} void @test2
  extern volatile wrapping_int a, b, c;

  // Basically, all cases should match the WRAPV case since this attribute
  // effectively enables wrapv for expressions containing wrapping types.

  // DEFAULT,WRAPV,TRAPV,CATCH_UB,TRAPV_HANDLER: add i32
  a = b + c;

  // DEFAULT,WRAPV,TRAPV,CATCH_UB,TRAPV_HANDLER: sub i32
  a = b - c;

  // DEFAULT,WRAPV,TRAPV,CATCH_UB,TRAPV_HANDLER: mul i32
  a = b * c;

  // DEFAULT,WRAPV,TRAPV,CATCH_UB,TRAPV_HANDLER: sub i32 0,
  a = -b;

  // DEFAULT,WRAPV,TRAPV,CATCH_UB,TRAPV_HANDLER: add i32 {{.*}}, 1
  ++b;

  // DEFAULT,WRAPV,TRAPV,CATCH_UB,TRAPV_HANDLER: add i32 {{.*}}, -1
  --b;

  // Less trivial cases
  extern volatile wrapping_int u, v;
  extern volatile int w;

  // DEFAULT,WRAPV,TRAPV,CATCH_UB,TRAPV_HANDLER: add i32
  if (u + v < u) {}

  // DEFAULT,WRAPV,TRAPV,CATCH_UB,TRAPV_HANDLER: add i32
  for (;u + v < u;) {}

  // this (w+1) should have instrumentation
  // DEFAULT,WRAPV,TRAPV,CATCH_UB,TRAPV_HANDLER: call {{.*}} @llvm.sadd.with.overflow.i32
  u = (w+1) + v;

  // no parts of this expression should have instrumentation
  // DEFAULT,WRAPV,TRAPV,CATCH_UB,TRAPV_HANDLER: add i32 {{.*}}, 1
  u = (v+1) + w;

  // downcast off the wraps attribute
  // DEFAULT,WRAPV,TRAPV,CATCH_UB,TRAPV_HANDLER: call { i32, i1 } @llvm.sadd.with.overflow.i32
  u = (int) u + (int) v;

  // DEFAULT,WRAPV,TRAPV,CATCH_UB,TRAPV_HANDLER: call { i32, i1 } @llvm.sadd.with.overflow.i32
  u = (int) u + w;
}
