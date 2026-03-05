// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -o %t.cir %s
// FileCheck --input-file=%t.cir --check-prefix=CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -o %t.ll %s
// FileCheck --input-file=%t.ll --check-prefix=CHECK-LLVM %s

// TODO: once we have support for targets that does not have native fp16
//       support but have fp16 conversion intrinsic support, add tests for
//       these targets.

volatile unsigned test;
volatile int i0;
volatile __fp16 h0 = 0.0, h1 = 1.0, h2;
volatile float f0, f1, f2;
volatile double d0;
short s0;

void foo(void) {
  test = (h0);
  // CHECK: %{{.+}} = cir.cast float_to_int %{{.+}} : !cir.f16 -> !u32i

  // CHECK-LLVM: %{{.+}} = fptoui half %{{.+}} to i32

  h0 = (test);
  // CHECK: %{{.+}} = cir.cast int_to_float %{{.+}} : !u32i -> !cir.f16

  // CHECK-LLVM: %{{.+}} = uitofp i32 %{{.+}} to half

  test = (!h1);
  //      CHECK: %[[#A:]] = cir.cast float_to_bool %{{.+}} : !cir.f16 -> !cir.bool
  // CHECK-NEXT: %[[#B:]] = cir.unary(not, %[[#A]]) : !cir.bool, !cir.bool
  // CHECK-NEXT: %[[#C:]] = cir.cast bool_to_int %[[#B]] : !cir.bool -> !s32i
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#C]] : !s32i -> !u32i

  //      CHECK-LLVM: %[[#A:]] = fcmp une half %{{.+}}, 0xH0000
  // CHECK-LLVM-NEXT: %[[#B:]] = zext i1 %[[#A]] to i8
  // CHECK-LLVM-NEXT: %[[#C:]] = xor i8 %[[#B]], 1
  // CHECK-LLVM-NEXT: %{{.+}} = zext i8 %[[#C]] to i32

  h1 = -h1;
  //  CHECK-NOT: %{{.+}} = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  //  CHECK-NOT: %{{.+}} = cir.cast floating %{{.+}} : !cir.float -> !cir.f16
  //      CHECK: %{{.+}} = cir.unary(minus, %{{.+}}) : !cir.f16, !cir.f16

  // CHECK-LLVM: %{{.+}} = fneg half %{{.+}}

  h1 = +h1;
  //  CHECK-NOT: %{{.+}} = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  //  CHECK-NOT: %{{.+}} = cir.cast floating %{{.+}} : !cir.float -> !cir.f16
  //      CHECK: %{{.+}} = cir.unary(plus, %{{.+}}) : !cir.f16, !cir.f16

  //      CHECK-LLVM: %[[#A:]] = load volatile half, ptr @h1, align 2
  // CHECK-LLVM-NEXT: store volatile half %[[#A]], ptr @h1, align 2

  h1++;
  //      CHECK: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.f16
  // CHECK-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  // CHECK-LLVM: %{.+} = fadd half %{.+}, 0xH3C00

  ++h1;
  //      CHECK: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.f16
  // CHECK-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  // CHECK-LLVM: %{.+} = fadd half %{.+}, 0xH3C00

  --h1;
  //      CHECK: %[[#A:]] = cir.const #cir.fp<-1.000000e+00> : !cir.f16
  // CHECK-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  // CHECK-LLVM: %{.+} = fadd half %{.+}, 0xHBC00

  h1--;
  //      CHECK: %[[#A:]] = cir.const #cir.fp<-1.000000e+00> : !cir.f16
  // CHECK-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  // CHECK-LLVM: %{.+} = fadd half %{.+}, 0xHBC00

  h1 = h0 * h2;
  // CHECK: %{{.+}} = cir.binop(mul, %{{.+}}, %{{.+}}) : !cir.f16

  // CHECK-LLVM: %{{.+}} = fmul half %{{.+}}, %{{.+}}

  h1 = h0 * (__fp16) -2.0f;
  //      CHECK: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.float -> !cir.f16
  // CHECK-NEXT: %{{.+}} = cir.binop(mul, %{{.+}}, %[[#C]]) : !cir.f16

  // CHECK-LLVM: %{{.+}} = fmul half %{{.+}}, 0xHC000

  h1 = h0 * f2;
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  //      CHECK: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.float
  // CHECK-NEXT: %{{.+}} = cir.cast floating %[[#B]] : !cir.float -> !cir.f16

  //      CHECK-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      CHECK-LLVM: %[[#RES:]] = fmul float %[[#LHS]], %{{.+}}
  // CHECK-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = f0 * h2;
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.float
  // CHECK-NEXT: %{{.+}} = cir.cast floating %[[#B]] : !cir.float -> !cir.f16

  //      CHECK-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // CHECK-LLVM-NEXT: %[[#RES:]] = fmul float %{{.+}}, %[[#RHS]]
  // CHECK-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = h0 * i0;
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  // CHECK-NEXT: %{{.+}} = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.f16

  //      CHECK-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM-NEXT: %{{.+}} = fmul half %{{.+}}, %[[#A]]

  h1 = (h0 / h2);
  // CHECK: %{{.+}} = cir.binop(div, %{{.+}}, %{{.+}}) : !cir.f16

  // CHECK-LLVM: %{{.+}} = fdiv half %{{.+}}, %{{.+}}

  h1 = (h0 / (__fp16) -2.0f);
  //      CHECK: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.float -> !cir.f16
  // CHECK-NEXT: %{{.+}} = cir.binop(div, %{{.+}}, %[[#C]]) : !cir.f16

  // CHECK-LLVM: %{{.+}} = fdiv half %{{.+}}, 0xHC000

  h1 = (h0 / f2);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  //      CHECK: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.float
  // CHECK-NEXT: %{{.+}} = cir.cast floating %[[#B]] : !cir.float -> !cir.f16

  //      CHECK-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      CHECK-LLVM: %[[#RES:]] = fdiv float %[[#LHS]], %{{.+}}
  // CHECK-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = (f0 / h2);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.float
  // CHECK-NEXT: %{{.+}} = cir.cast floating %[[#B]] : !cir.float -> !cir.f16

  //      CHECK-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // CHECK-LLVM-NEXT: %[[#RES:]] = fdiv float %{{.+}}, %[[#RHS]]
  // CHECK-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = (h0 / i0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  // CHECK-NEXT: %{{.+}} = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.f16

  //      CHECK-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM-NEXT: %{{.+}} = fdiv half %{{.+}}, %[[#A]]

  h1 = (h2 + h0);
  // CHECK: %{{.+}} = cir.binop(add, %{{.+}}, %{{.+}}) : !cir.f16

  // CHECK-LLVM: %{{.+}} = fadd half %{{.+}}, %{{.+}}

  h1 = ((__fp16)-2.0 + h0);
  //      CHECK: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // CHECK-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.double, !cir.double
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.double -> !cir.f16
  //      CHECK: %{{.+}} = cir.binop(add, %[[#C]], %{{.+}}) : !cir.f16

  // CHECK-LLVM: %{{.+}} = fadd half 0xHC000, %{{.+}}

  h1 = (h2 + f0);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  //      CHECK: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.float
  // CHECK-NEXT: %{{.+}} = cir.cast floating %[[#B]] : !cir.float -> !cir.f16

  //      CHECK-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      CHECK-LLVM: %[[#RES:]] = fadd float %[[#LHS]], %{{.+}}
  // CHECK-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = (f2 + h0);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.float
  // CHECK-NEXT: %{{.+}} = cir.cast floating %[[#B]] : !cir.float -> !cir.f16

  //      CHECK-LLVM: %[[#RHS:]] = fpext half %{{.=}} to float
  // CHECK-LLVM-NEXT: %[[#RES:]] = fadd float %{{.+}}, %[[#RHS]]
  // CHECK-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = (h0 + i0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  // CHECK-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  //      CHECK-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM-NEXT: %{{.+}} = fadd half %{{.+}}, %[[#A]]

  h1 = (h2 - h0);
  // CHECK: %{{.+}} = cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.f16

  // CHECK-LLVM: %{{.+}} = fsub half %{{.+}}, %{{.+}}

  h1 = ((__fp16)-2.0f - h0);
  //      CHECK: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.float -> !cir.f16
  //      CHECK: %{{.+}} = cir.binop(sub, %[[#C]], %{{.+}}) : !cir.f16

  // CHECK-LLVM: %{{.+}} = fsub half 0xHC000, %{{.+}}

  h1 = (h2 - f0);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  //      CHECK: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.float
  // CHECK-NEXT: %{{.+}} = cir.cast floating %[[#B]] : !cir.float -> !cir.f16

  //      CHECK-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      CHECK-LLVM: %[[#RES:]] = fsub float %[[#LHS]], %{{.+}}
  // CHECK-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = (f2 - h0);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.float
  // CHECK-NEXT: %{{.+}} = cir.cast floating %[[#B]] : !cir.float -> !cir.f16

  //      CHECK-LLVM: %[[#RHS:]] = fpext half %{{.=}} to float
  // CHECK-LLVM-NEXT: %[[#RES:]] = fsub float %{{.+}}, %[[#RHS]]
  // CHECK-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = (h0 - i0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  // CHECK-NEXT: %{{.+}} = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.f16

  //      CHECK-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM-NEXT: %{{.+}} = fsub half %{{.+}}, %[[#A]]

  test = (h2 < h0);
  //      CHECK: %[[#A:]] = cir.cmp(lt, %{{.+}}, %{{.+}}) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#A]] : !s32i -> !u32i

  // CHECK-LLVM: %{{.+}} = fcmp olt half %{{.+}}, %{{.+}}

  test = (h2 < (__fp16)42.0);
  //      CHECK: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // CHECK-NEXT: %[[#B:]] = cir.cast floating %[[#A]] : !cir.double -> !cir.f16
  // CHECK-NEXT: %[[#C:]] = cir.cmp(lt, %{{.+}}, %[[#B]]) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#C]] : !s32i -> !u32i

  // CHECK-LLVM: %{{.+}} = fcmp olt half %{{.+}}, 0xH5140

  test = (h2 < f0);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  //      CHECK: %[[#B:]] = cir.cmp(lt, %[[#A]], %{{.+}}) : !cir.float, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  // CHECK-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // CHECK-LLVM: %{{.+}} = fcmp olt float %[[#A]], %{{.+}}

  test = (f2 < h0);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.cmp(lt, %{{.+}}, %[[#A]]) : !cir.float, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  //      CHECK-LLVM: %[[#A:]] = fpext half %{{.=}} to float
  // CHECK-LLVM-NEXT: %{{.+}} = fcmp olt float %{{.+}}, %[[#A]]

  test = (i0 < h0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  //      CHECK: %[[#B:]] = cir.cmp(lt, %[[#A]], %{{.+}}) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  // CHECK-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM: %{{.+}} = fcmp olt half %[[#A]], %{{.+}}

  test = (h0 < i0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  // CHECK-NEXT: %[[#B:]] = cir.cmp(lt, %{{.+}}, %[[#A]]) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  //      CHECK-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM-NEXT: %{{.+}} = fcmp olt half %{{.+}}, %[[#A]]

  test = (h0 > h2);
  //      CHECK: %[[#A:]] = cir.cmp(gt, %{{.+}}, %{{.+}}) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#A]] : !s32i -> !u32i

  // CHECK-LLVM: %{{.+}} = fcmp ogt half %{{.+}}, %{{.+}}

  test = ((__fp16)42.0 > h2);
  //      CHECK: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // CHECK-NEXT: %[[#B:]] = cir.cast floating %[[#A]] : !cir.double -> !cir.f16
  //      CHECK: %[[#C:]] = cir.cmp(gt, %[[#B]], %{{.+}}) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#C]] : !s32i -> !u32i

  // CHECK-LLVM: %{{.+}} = fcmp ogt half 0xH5140, %{{.+}}

  test = (h0 > f2);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  //      CHECK: %[[#B:]] = cir.cmp(gt, %[[#A]], %{{.+}}) : !cir.float, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  // CHECK-LLVM: %[[#LHS:]] = fpext half %{{.=}} to float
  // CHECK-LLVM: %{{.+}} = fcmp ogt float %[[#LHS]], %{{.+}}

  test = (f0 > h2);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.cmp(gt, %{{.+}}, %[[#A]]) : !cir.float, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  //      CHECK-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // CHECK-LLVM-NEXT: %{{.+}} = fcmp ogt float %{{.+}}, %[[#RHS]]

  test = (i0 > h0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  //      CHECK: %[[#B:]] = cir.cmp(gt, %[[#A]], %{{.+}}) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  // CHECK-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM: %{{.+}} = fcmp ogt half %[[#LHS]], %{{.+}}

  test = (h0 > i0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  // CHECK-NEXT: %[[#B:]] = cir.cmp(gt, %{{.+}}, %[[#A]]) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  //      CHECK-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM-NEXT: %{{.+}} = fcmp ogt half %{{.+}}, %[[#RHS]]

  test = (h2 <= h0);
  //      CHECK: %[[#A:]] = cir.cmp(le, %{{.+}}, %{{.+}}) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#A]] : !s32i -> !u32i

  // CHECK-LLVM: %{{.+}} = fcmp ole half %{{.+}}, %{{.+}}

  test = (h2 <= (__fp16)42.0);
  //      CHECK: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // CHECK-NEXT: %[[#B:]] = cir.cast floating %[[#A]] : !cir.double -> !cir.f16
  // CHECK-NEXT: %[[#C:]] = cir.cmp(le, %{{.+}}, %[[#B]]) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#C]] : !s32i -> !u32i

  // CHECK-LLVM: %{{.+}} = fcmp ole half %{{.+}}, 0xH5140

  test = (h2 <= f0);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  //      CHECK: %[[#B:]] = cir.cmp(le, %[[#A]], %{{.+}}) : !cir.float, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  // CHECK-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // CHECK-LLVM: %{{.+}} = fcmp ole float %[[#LHS]], %{{.+}}

  test = (f2 <= h0);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.cmp(le, %{{.+}}, %[[#A]]) : !cir.float, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  //      CHECK-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // CHECK-LLVM-NEXT: %{{.+}} = fcmp ole float %{{.+}}, %[[#RHS]]

  test = (i0 <= h0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  //      CHECK: %[[#B:]] = cir.cmp(le, %[[#A]], %{{.+}}) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  // CHECK-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM: %{{.+}} = fcmp ole half %[[#LHS]], %{{.+}}

  test = (h0 <= i0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  // CHECK-NEXT: %[[#B:]] = cir.cmp(le, %{{.+}}, %[[#A]]) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  //      CHECK-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM-NEXT: %{{.+}} = fcmp ole half %{{.+}}, %[[#RHS]]

  test = (h0 >= h2);
  //      CHECK: %[[#A:]] = cir.cmp(ge, %{{.+}}, %{{.+}}) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#A]] : !s32i -> !u32i

  // CHECK-LLVM: %{{.+}} = fcmp oge half %{{.+}}, %{{.+}}

  test = (h0 >= (__fp16)-2.0);
  //      CHECK: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // CHECK-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.double, !cir.double
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.double -> !cir.f16
  // CHECK-NEXT: %[[#D:]] = cir.cmp(ge, %{{.+}}, %[[#C]]) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#D]] : !s32i -> !u32i

  // CHECK-LLVM: %{{.+}} = fcmp oge half %{{.+}}, 0xHC000

  test = (h0 >= f2);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  //      CHECK: %[[#B:]] = cir.cmp(ge, %[[#A]], %{{.+}}) : !cir.float, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  // CHECK-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // CHECK-LLVM: %{{.+}} = fcmp oge float %[[#LHS]], %{{.+}}

  test = (f0 >= h2);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.cmp(ge, %{{.+}}, %[[#A]]) : !cir.float, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  //      CHECK-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // CHECK-LLVM-NEXT: %{{.+}} = fcmp oge float %{{.+}}, %[[#RHS]]

  test = (i0 >= h0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  //      CHECK: %[[#B:]] = cir.cmp(ge, %[[#A]], %{{.+}}) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  // CHECK-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM: %{{.+}} = fcmp oge half %[[#LHS]], %{{.+}}

  test = (h0 >= i0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  // CHECK-NEXT: %[[#B:]] = cir.cmp(ge, %{{.+}}, %[[#A]]) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  //      CHECK-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM-NEXT: %{{.+}} = fcmp oge half %{{.+}}, %[[#RHS]]

  test = (h1 == h2);
  //      CHECK: %[[#A:]] = cir.cmp(eq, %{{.+}}, %{{.+}}) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#A]] : !s32i -> !u32i

  // CHECK-LLVM: %{{.+}} = fcmp oeq half %{{.+}}, %{{.+}}

  test = (h1 == (__fp16)1.0);
  //      CHECK: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // CHECK-NEXT: %[[#B:]] = cir.cast floating %[[#A]] : !cir.double -> !cir.f16
  // CHECK-NEXT: %[[#C:]] = cir.cmp(eq, %{{.+}}, %[[#B]]) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#C]] : !s32i -> !u32i

  // CHECK-LLVM: %{{.+}} = fcmp oeq half %{{.+}}, 0xH3C00

  test = (h1 == f1);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  //      CHECK: %[[#B:]] = cir.cmp(eq, %[[#A]], %{{.+}}) : !cir.float, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  // CHECK-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // CHECK-LLVM: %{{.+}} = fcmp oeq float %[[#LHS]], %{{.+}}

  test = (f1 == h1);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.cmp(eq, %{{.+}}, %[[#A]]) : !cir.float, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  //      CHECK-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // CHECK-LLVM-NEXT: %{{.+}} = fcmp oeq float %{{.+}}, %[[#RHS]]

  test = (i0 == h0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  //      CHECK: %[[#B:]] = cir.cmp(eq, %[[#A]], %{{.+}}) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  // CHECK-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM: %{{.+}} = fcmp oeq half %[[#LHS]], %{{.+}}

  test = (h0 == i0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  // CHECK-NEXT: %[[#B:]] = cir.cmp(eq, %{{.+}}, %[[#A]]) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  //      CHECK-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM-NEXT: %{{.=}} = fcmp oeq half %{{.+}}, %[[#RHS]]

  test = (h1 != h2);
  //      CHECK: %[[#A:]] = cir.cmp(ne, %{{.+}}, %{{.+}}) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#A]] : !s32i -> !u32i

  // CHECK-LLVM: %{{.+}} = fcmp une half %{{.+}}, %{{.+}}

  test = (h1 != (__fp16)1.0);
  //      CHECK: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // CHECK-NEXT: %[[#B:]] = cir.cast floating %[[#A]] : !cir.double -> !cir.f16
  // CHECK-NEXT: %[[#C:]] = cir.cmp(ne, %{{.+}}, %[[#B]]) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#C]] : !s32i -> !u32i

  // CHECK-LLVM: %{{.+}} = fcmp une half %{{.+}}, 0xH3C00

  test = (h1 != f1);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  //      CHECK: %[[#B:]] = cir.cmp(ne, %[[#A]], %{{.+}}) : !cir.float, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  // CHECK-LLVM: %[[#LHS:]] = fpext half %{{.=}} to float
  // CHECK-LLVM: %{{.+}} = fcmp une float %[[#LHS]], %{{.+}}

  test = (f1 != h1);
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.float, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  //      CHECK-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // CHECK-LLVM-NEXT: %{{.+}} = fcmp une float %{{.+}}, %[[#A]]

  test = (i0 != h0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  //      CHECK: %[[#B:]] = cir.cmp(ne, %[[#A]], %{{.+}}) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  // CHECK-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM: %{{.+}} = fcmp une half %[[#LHS]], %{{.+}}

  test = (h0 != i0);
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  // CHECK-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.f16, !cir.bool
  // CHECK-NEXT: %{{.+}} = cir.cast integral %[[#B]] : !s32i -> !u32i

  //      CHECK-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM-NEXT: %{{.+}} = fcmp une half %{{.+}}, %[[#RHS]]

  h1 = (h1 ? h2 : h0);
  //      CHECK: %[[#A:]] = cir.cast float_to_bool %{{.+}} : !cir.f16 -> !cir.bool
  // CHECK-NEXT: %[[#B:]] = cir.ternary(%[[#A]], true {
  //      CHECK:   cir.yield %{{.+}} : !cir.f16
  // CHECK-NEXT: }, false {
  //      CHECK:   cir.yield %{{.+}} : !cir.f16
  // CHECK-NEXT: }) : (!cir.bool) -> !cir.f16
  // CHECK-NEXT: %[[#C:]] = cir.get_global @h1 : !cir.ptr<!cir.f16>
  // CHECK-NEXT: cir.store volatile %[[#B]], %[[#C]] : !cir.f16, !cir.ptr<!cir.f16>

  //      CHECK-LLVM:   %[[#A:]] = fcmp une half %{{.+}}, 0xH0000
  // CHECK-LLVM-NEXT:   br i1 %[[#A]], label %[[#LABEL_A:]], label %[[#LABEL_B:]]
  //      CHECK-LLVM: [[#LABEL_A]]:
  // CHECK-LLVM-NEXT:   %[[#B:]] = load volatile half, ptr @h2, align 2
  // CHECK-LLVM-NEXT:   br label %[[#LABEL_C:]]
  //      CHECK-LLVM: [[#LABEL_B]]:
  // CHECK-LLVM-NEXT:   %[[#C:]] = load volatile half, ptr @h0, align 2
  // CHECK-LLVM-NEXT:   br label %[[#LABEL_C]]
  //      CHECK-LLVM: [[#LABEL_C]]:
  // CHECK-LLVM-NEXT:   %8 = phi half [ %[[#C]], %[[#LABEL_B]] ], [ %[[#B]], %[[#LABEL_A]] ]

  h0 = h1;
  //      CHECK: %[[#A:]] = cir.get_global @h1 : !cir.ptr<!cir.f16>
  // CHECK-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // CHECK-NEXT: %[[#C:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // CHECK-NEXT: cir.store volatile %[[#B]], %[[#C]] : !cir.f16, !cir.ptr<!cir.f16>

  //      CHECK-LLVM: %[[#A:]] = load volatile half, ptr @h1, align 2
  // CHECK-LLVM-NEXT: store volatile half %[[#A]], ptr @h0, align 2

  h0 = (__fp16)-2.0f;
  //      CHECK: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.float -> !cir.f16
  // CHECK-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // CHECK-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  // CHECK-LLVM: store volatile half 0xHC000, ptr @h0, align 2

  h0 = f0;
  //      CHECK: %[[#A:]] = cir.get_global @f0 : !cir.ptr<!cir.float>
  // CHECK-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.float -> !cir.f16
  // CHECK-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // CHECK-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      CHECK-LLVM: %[[#A:]] = load volatile float, ptr @f0, align 4
  // CHECK-LLVM-NEXT: %[[#B:]] = fptrunc float %[[#A]] to half
  // CHECK-LLVM-NEXT: store volatile half %[[#B]], ptr @h0, align 2

  h0 = i0;
  //      CHECK: %[[#A:]] = cir.get_global @i0 : !cir.ptr<!s32i>
  // CHECK-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[#C:]] = cir.cast int_to_float %[[#B]] : !s32i -> !cir.f16
  // CHECK-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // CHECK-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      CHECK-LLVM: %[[#A:]] = load volatile i32, ptr @i0, align 4
  // CHECK-LLVM-NEXT: %[[#B:]] = sitofp i32 %[[#A]] to half
  // CHECK-LLVM-NEXT: store volatile half %[[#B]], ptr @h0, align 2

  i0 = h0;
  //      CHECK: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // CHECK-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // CHECK-NEXT: %[[#C:]] = cir.cast float_to_int %[[#B]] : !cir.f16 -> !s32i
  // CHECK-NEXT: %[[#D:]] = cir.get_global @i0 : !cir.ptr<!s32i>
  // CHECK-NEXT: cir.store volatile %[[#C]], %[[#D]] : !s32i, !cir.ptr<!s32i>

  //      CHECK-LLVM: %[[#A:]] = load volatile half, ptr @h0, align 2
  // CHECK-LLVM-NEXT: %[[#B:]] = fptosi half %[[#A]] to i32
  // CHECK-LLVM-NEXT: store volatile i32 %[[#B]], ptr @i0, align 4

  h0 += h1;
  //      CHECK: %[[#A:]] = cir.binop(add, %{{.+}}, %{{.+}}) : !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  // CHECK-LLVM: %{{.+}} = fadd half %{{.+}}, %{{.+}}

  h0 += (__fp16)1.0f;
  //      CHECK: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.cast floating %[[#A]] : !cir.float -> !cir.f16
  //      CHECK: %[[#C:]] = cir.binop(add, %{{.+}}, %[[#B]]) : !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  // CHECK-LLVM: %{{.+}} = fadd half %{{.+}}, 0xH3C00

  h0 += f2;
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.float
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.float -> !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      CHECK-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // CHECK-LLVM-NEXT: %[[#RES:]] = fadd float %[[#LHS]], %{{.+}}
  // CHECK-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  i0 += h0;
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  // CHECK-NEXT: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.f16
  // CHECK-NEXT: %[[#C:]] = cir.cast float_to_int %[[#B]] : !cir.f16 -> !s32i
  // CHECK-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      CHECK-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM-NEXT: %[[#B:]] = fadd half %[[#A]], %{{.+}}
  // CHECK-LLVM-NEXT: %{{.+}} = fptosi half %[[#B]] to i32

  h0 += i0;
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  //      CHECK: %[[#B:]] = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  // CHECK-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM: %{{.+}} = fadd half %{{.+}}, %[[#A]]

  h0 -= h1;
  //      CHECK: %[[#A:]] = cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  // CHECK-LLVM: %{{.+}} = fsub half %{{.+}}, %{{.+}}

  h0 -= (__fp16)1.0;
  //      CHECK: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // CHECK-NEXT: %[[#B:]] = cir.cast floating %[[#A]] : !cir.double -> !cir.f16
  //      CHECK: %[[#C:]] = cir.binop(sub, %{{.+}}, %[[#B]]) : !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  // CHECK-LLVM: %{{.+}} = fsub half %{{.+}}, 0xH3C00

  h0 -= f2;
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.float
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.float -> !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      CHECK-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // CHECK-LLVM-NEXT: %[[#RES:]] = fsub float %[[#LHS]], %{{.+}}
  // CHECK-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  i0 -= h0;
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  // CHECK-NEXT: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.f16
  // CHECK-NEXT: %[[#C:]] = cir.cast float_to_int %[[#B]] : !cir.f16 -> !s32i
  // CHECK-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      CHECK-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM-NEXT: %[[#B:]] = fsub half %[[#A]], %{{.+}}
  // CHECK-LLVM-NEXT: %{{.+}} = fptosi half %[[#B]] to i32

  h0 -= i0;
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  //      CHECK: %[[#B:]] = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  // CHECK-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM: %{{.+}} = fsub half %{{.+}}, %[[#A]]

  h0 *= h1;
  //      CHECK: %[[#A:]] = cir.binop(mul, %{{.+}}, %{{.+}}) : !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  // CHECK-LLVM: %{{.+}} = fmul half %{{.+}}, %{{.+}}

  h0 *= (__fp16)1.0;
  //      CHECK: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // CHECK-NEXT: %[[#B:]] = cir.cast floating %[[#A]] : !cir.double -> !cir.f16
  //      CHECK: %[[#C:]] = cir.binop(mul, %{{.+}}, %[[#B]]) : !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  // CHECK-LLVM: %{{.+}} = fmul half %{{.+}}, 0xH3C00

  h0 *= f2;
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.float
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.float -> !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      CHECK-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // CHECK-LLVM-NEXT: %[[#RES:]] = fmul float %[[#LHS]], %{{.+}}
  // CHECK-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  i0 *= h0;
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  // CHECK-NEXT: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.f16
  // CHECK-NEXT: %[[#C:]] = cir.cast float_to_int %[[#B]] : !cir.f16 -> !s32i
  // CHECK-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      CHECK-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM-NEXT: %[[#B:]] = fmul half %[[#A]], %{{.+}}
  // CHECK-LLVM-NEXT: %{{.+}} = fptosi half %[[#B]] to i32

  h0 *= i0;
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  //      CHECK: %[[#B:]] = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  // CHECK-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM: %{{.+}} = fmul half %{{.+}}, %[[#A]]

  h0 /= h1;
  //      CHECK: %[[#A:]] = cir.binop(div, %{{.+}}, %{{.+}}) : !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  // CHECK-LLVM: %{{.+}} = fdiv half %{{.+}}, %{{.+}}

  h0 /= (__fp16)1.0;
  //      CHECK: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // CHECK-NEXT: %[[#B:]] = cir.cast floating %[[#A]] : !cir.double -> !cir.f16
  //      CHECK: %[[#C:]] = cir.binop(div, %{{.+}}, %[[#B]]) : !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  // CHECK-LLVM: %{{.+}} = fdiv half %{{.+}}, 0xH3C00

  h0 /= f2;
  //      CHECK: %[[#A:]] = cir.cast floating %{{.+}} : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.float
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.float -> !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      CHECK-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // CHECK-LLVM-NEXT: %[[#RES:]] = fdiv float %[[#LHS]], %{{.+}}
  // CHECK-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  i0 /= h0;
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  // CHECK-NEXT: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.f16
  // CHECK-NEXT: %[[#C:]] = cir.cast float_to_int %[[#B]] : !cir.f16 -> !s32i
  // CHECK-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      CHECK-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM-NEXT: %[[#B:]] = fdiv half %[[#A]], %{{.+}}
  // CHECK-LLVM-NEXT: %{{.+}} = fptosi half %[[#B]] to i32

  h0 /= i0;
  //      CHECK: %[[#A:]] = cir.cast int_to_float %{{.+}} : !s32i -> !cir.f16
  //      CHECK: %[[#B:]] = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.f16
  // CHECK-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  // CHECK-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // CHECK-LLVM: %{{.+}} = fdiv half %{{.+}}, %[[#A]]

  h0 = d0;
  //      CHECK: %[[#A:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // CHECK-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.double>, !cir.double
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.double -> !cir.f16
  // CHECK-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // CHECK-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      CHECK-LLVM: %[[#A:]] = load volatile double, ptr @d0, align 8
  // CHECK-LLVM-NEXT: %[[#B:]] = fptrunc double %[[#A]] to half
  // CHECK-LLVM-NEXT: store volatile half %[[#B]], ptr @h0, align 2

  h0 = (float)d0;
  //      CHECK: %[[#A:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // CHECK-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.double>, !cir.double
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.double -> !cir.float
  // CHECK-NEXT: %[[#D:]] = cir.cast floating %[[#C]] : !cir.float -> !cir.f16
  // CHECK-NEXT: %[[#E:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // CHECK-NEXT: cir.store volatile %[[#D]], %[[#E]] : !cir.f16, !cir.ptr<!cir.f16>

  //      CHECK-LLVM: %[[#A:]] = load volatile double, ptr @d0, align 8
  // CHECK-LLVM-NEXT: %[[#B:]] = fptrunc double %[[#A]] to float
  // CHECK-LLVM-NEXT: %[[#C:]] = fptrunc float %[[#B]] to half
  // CHECK-LLVM-NEXT: store volatile half %[[#C]], ptr @h0, align 2

  d0 = h0;
  //      CHECK: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // CHECK-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.f16 -> !cir.double
  // CHECK-NEXT: %[[#D:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // CHECK-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.double, !cir.ptr<!cir.double>

  //      CHECK-LLVM: %[[#A:]] = load volatile half, ptr @h0, align 2
  // CHECK-LLVM-NEXT: %[[#B:]] = fpext half %[[#A]] to double
  // CHECK-LLVM-NEXT: store volatile double %[[#B]], ptr @d0, align 8

  d0 = (float)h0;
  //      CHECK: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // CHECK-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // CHECK-NEXT: %[[#C:]] = cir.cast floating %[[#B]] : !cir.f16 -> !cir.float
  // CHECK-NEXT: %[[#D:]] = cir.cast floating %[[#C]] : !cir.float -> !cir.double
  // CHECK-NEXT: %[[#E:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // CHECK-NEXT: cir.store volatile %[[#D]], %[[#E]] : !cir.double, !cir.ptr<!cir.double>

  //      CHECK-LLVM: %[[#A:]] = load volatile half, ptr @h0, align 2
  // CHECK-LLVM-NEXT: %[[#B:]] = fpext half %[[#A]] to float
  // CHECK-LLVM-NEXT: %[[#C:]] = fpext float %[[#B]] to double
  // CHECK-LLVM-NEXT: store volatile double %[[#C]], ptr @d0, align 8

  h0 = s0;
  //      CHECK: %[[#A:]] = cir.get_global @s0 : !cir.ptr<!s16i>
  // CHECK-NEXT: %[[#B:]] = cir.load %[[#A]] : !cir.ptr<!s16i>, !s16i
  // CHECK-NEXT: %[[#C:]] = cir.cast int_to_float %[[#B]] : !s16i -> !cir.f16
  // CHECK-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // CHECK-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      CHECK-LLVM: %[[#A:]] = load i16, ptr @s0, align 2
  // CHECK-LLVM-NEXT: %[[#B:]] = sitofp i16 %[[#A]] to half
  // CHECK-LLVM-NEXT: store volatile half %[[#B]], ptr @h0, align 2
}
