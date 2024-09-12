// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -o %t.cir %s
// FileCheck --input-file=%t.cir --check-prefix=NONATIVE %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fnative-half-type -fclangir -emit-cir -o %t.cir %s
// FileCheck --input-file=%t.cir --check-prefix=NATIVE %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -o %t.ll %s
// FileCheck --input-file=%t.ll --check-prefix=NONATIVE-LLVM %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fnative-half-type -fclangir -emit-llvm -o %t.ll %s
// FileCheck --input-file=%t.ll --check-prefix=NATIVE-LLVM %s

volatile unsigned test;
volatile int i0;
volatile _Float16 h0 = 0.0, h1 = 1.0, h2;
volatile float f0, f1, f2;
volatile double d0;
short s0;

void foo(void) {
  test = (h0);
  // NONATIVE: %{{.+}} = cir.cast(float_to_int, %{{.+}} : !cir.f16), !u32i
  // NATIVE: %{{.+}} = cir.cast(float_to_int, %{{.+}} : !cir.f16), !u32i

  // NONATIVE-LLVM: %{{.+}} = fptoui half %{{.+}} to i32
  // NATIVE-LLVM: %{{.+}} = fptoui half %{{.+}} to i32

  h0 = (test);
  // NONATIVE: %{{.+}} = cir.cast(int_to_float, %{{.+}} : !u32i), !cir.f16
  // NATIVE: %{{.+}} = cir.cast(int_to_float, %{{.+}} : !u32i), !cir.f16

  // NONATIVE-LLVM: %{{.+}} = uitofp i32 %{{.+}} to half
  // NATIVE-LLVM: %{{.+}} = uitofp i32 %{{.+}} to half

  test = (!h1);
  //      NONATIVE: %[[#A:]] = cir.cast(float_to_bool, %{{.+}} : !cir.f16), !cir.bool
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(not, %[[#A]]) : !cir.bool, !cir.bool
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(bool_to_int, %[[#B]] : !cir.bool), !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(float_to_bool, %{{.+}} : !cir.f16), !cir.bool
  // NATIVE-NEXT: %[[#B:]] = cir.unary(not, %[[#A]]) : !cir.bool, !cir.bool
  // NATIVE-NEXT: %[[#C:]] = cir.cast(bool_to_int, %[[#B]] : !cir.bool), !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#A:]] = fcmp une half %{{.+}}, 0xH0000
  // NONATIVE-LLVM-NEXT: %[[#B:]] = zext i1 %[[#A]] to i8
  // NONATIVE-LLVM-NEXT: %[[#C:]] = xor i8 %[[#B]], 1
  // NONATIVE-LLVM-NEXT: %{{.+}} = zext i8 %[[#C]] to i32

  //      NATIVE-LLVM: %[[#A:]] = fcmp une half %{{.+}}, 0xH0000
  // NATIVE-LLVM-NEXT: %[[#B:]] = zext i1 %[[#A]] to i8
  // NATIVE-LLVM-NEXT: %[[#C:]] = xor i8 %[[#B]], 1
  // NATIVE-LLVM-NEXT: %{{.+}} = zext i8 %[[#C]] to i32

  h1 = -h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //  NATIVE-NOT: %{{.+}} = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //  NATIVE-NOT: %{{.+}} = cir.cast(floating, %{{.+}} : !cir.float), !cir.f16
  //      NATIVE: %{{.+}} = cir.unary(minus, %{{.+}}) : !cir.f16, !cir.f16

  //      NONATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fneg float %[[#A]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to half

  // NATIVE-LLVM: %{{.+}} = fneg half %{{.+}}

  h1 = +h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(plus, %[[#A]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //  NATIVE-NOT: %{{.+}} = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //  NATIVE-NOT: %{{.+}} = cir.cast(floating, %{{.+}} : !cir.float), !cir.f16
  //      NATIVE: %{{.+}} = cir.unary(plus, %{{.+}}) : !cir.f16, !cir.f16

  //      NONATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fptrunc float %[[#A]] to half

  //      NATIVE-LLVM: %[[#A:]] = load volatile half, ptr @h1, align 2
  // NATIVE-LLVM-NEXT: store volatile half %[[#A]], ptr @h1, align 2

  h1++;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.f16
  // NONATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  // NONATIVE-LLVM: %{.+} = fadd half %{.+}, 0xH3C00

  // NATIVE-LLVM: %{.+} = fadd half %{.+}, 0xH3C00

  ++h1;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.f16
  // NONATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  // NONATIVE-LLVM: %{.+} = fadd half %{.+}, 0xH3C00

  // NATIVE-LLVM: %{.+} = fadd half %{.+}, 0xH3C00

  --h1;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<-1.000000e+00> : !cir.f16
  // NONATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<-1.000000e+00> : !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  // NONATIVE-LLVM: %{.+} = fadd half %{.+}, 0xHBC00

  // NATIVE-LLVM: %{.+} = fadd half %{.+}, 0xHBC00

  h1--;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<-1.000000e+00> : !cir.f16
  // NONATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<-1.000000e+00> : !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  // NONATIVE-LLVM: %{.+} = fadd half %{.+}, 0xHBC00

  // NATIVE-LLVM: %{.+} = fadd half %{.+}, 0xHBC00

  h1 = h0 * h2;
  //      NONATIVE: %[[#LHS:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#RHS:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#A:]] = cir.binop(mul, %[[#LHS]], %[[#RHS]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#A]] : !cir.float), !cir.f16

  // NATIVE: %{{.+}} = cir.binop(mul, %{{.+}}, %{{.+}}) : !cir.f16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#SUM:]] = fmul float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#SUM]] to half

  // NATIVE-LLVM: %{{.+}} = fmul half %{{.+}}, %{{.+}}

  h1 = h0 * (_Float16) -2.0f;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.unary(minus, %[[#B]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.binop(mul, %[[#A]], %[[#E]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#F]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(mul, %{{.+}}, %[[#C]]) : !cir.f16

  //      NONATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fmul float %[[#A]], -2.000000e+00
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to half

  // NATIVE-LLVM: %{{.+}} = fmul half %{{.+}}, 0xHC000

  h1 = h0 * f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RES:]] = fmul float %[[#LHS]], %{{.+}}
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NATIVE-LLVM: %[[#RES:]] = fmul float %[[#LHS]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = f0 * h2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fmul float %{{.+}}, %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#RES:]] = fmul float %{{.+}}, %[[#RHS]]
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = h0 * i0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(mul, %[[#A]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.f16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM-NEXT: %[[#A:]] = fpext half %[[#RHS]] to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fmul float %[[#LHS]], %[[#A]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM-NEXT: %{{.+}} = fmul half %{{.+}}, %[[#A]]

  h1 = (h0 / h2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(div, %[[#A]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16

  // NATIVE: %{{.+}} = cir.binop(div, %{{.+}}, %{{.+}}) : !cir.f16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fdiv float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  // NATIVE-LLVM: %{{.+}} = fdiv half %{{.+}}, %{{.+}}

  h1 = (h0 / (_Float16) -2.0f);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.unary(minus, %[[#B]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.binop(div, %[[#A]], %[[#E]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#F]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(div, %{{.+}}, %[[#C]]) : !cir.f16

  //      NONATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fdiv float %[[#A]], -2.000000e+00
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to half

  // NATIVE-LLVM: %{{.+}} = fdiv half %{{.+}}, 0xHC000

  h1 = (h0 / f2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RES:]] = fdiv float %[[#LHS]], %{{.+}}
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NATIVE-LLVM: %[[#RES:]] = fdiv float %[[#LHS]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = (f0 / h2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fdiv float %{{.+}}, %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#RES:]] = fdiv float %{{.+}}, %[[#RHS]]
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = (h0 / i0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(div, %[[#A]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.f16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM-NEXT: %[[#A:]] = fpext half %[[#RHS]] to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fdiv float %[[#LHS]], %[[#A]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM-NEXT: %{{.+}} = fdiv half %{{.+}}, %[[#A]]

  h1 = (h2 + h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(add, %[[#A]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16

  // NATIVE: %{{.+}} = cir.binop(add, %{{.+}}, %{{.+}}) : !cir.f16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fadd float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  // NATIVE-LLVM: %{{.+}} = fadd half %{{.+}}, %{{.+}}

  h1 = ((_Float16)-2.0 + h0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.double, !cir.double
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#E:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.binop(add, %[[#D]], %[[#E]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#F]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.double, !cir.double
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.f16
  //      NATIVE: %{{.+}} = cir.binop(add, %[[#C]], %{{.+}}) : !cir.f16

  //      NONATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fadd float -2.000000e+00, %[[#A]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to half

  // NATIVE-LLVM: %{{.+}} = fadd half 0xHC000, %{{.+}}

  h1 = (h2 + f0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RES:]] = fadd float %[[#LHS]], %{{.+}}
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NATIVE-LLVM: %[[#RES:]] = fadd float %[[#LHS]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = (f2 + h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NONATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fadd float %{{.+}}, %[[#A]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to half

  //      NATIVE-LLVM: %[[#RHS:]] = fpext half %{{.=}} to float
  // NATIVE-LLVM-NEXT: %[[#RES:]] = fadd float %{{.+}}, %[[#RHS]]
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = (h0 + i0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(add, %[[#A]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM-NEXT: %[[#A:]] = fpext half %[[#RHS]] to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fadd float %[[#LHS]], %[[#A]]
  // NONATIVE-LLVM-NEXT: %{{.=}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM-NEXT: %{{.+}} = fadd half %{{.+}}, %[[#A]]

  h1 = (h2 - h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(sub, %[[#A]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16

  // NATIVE: %{{.+}} = cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.f16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  // NATIVE-LLVM: %{{.+}} = fsub half %{{.+}}, %{{.+}}

  h1 = ((_Float16)-2.0f - h0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#E:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.binop(sub, %[[#D]], %[[#E]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#F]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  //      NATIVE: %{{.+}} = cir.binop(sub, %[[#C]], %{{.+}}) : !cir.f16

  //      NONATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fsub float -2.000000e+00, %[[#A]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to half

  // NATIVE-LLVM: %{{.+}} = fsub half 0xHC000, %{{.+}}

  h1 = (h2 - f0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RES:]] = fsub float %[[#LHS]], %{{.+}}
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NATIVE-LLVM: %[[#RES:]] = fsub float %[[#LHS]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = (f2 - h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %{{.+}}, %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#RHS:]] = fpext half %{{.=}} to float
  // NATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %{{.+}}, %[[#RHS]]
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  h1 = (h0 - i0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(sub, %[[#A]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.f16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM-NEXT: %[[#A:]] = fpext half %[[#RHS]] to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %[[#LHS]], %[[#A]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM-NEXT: %{{.+}} = fsub half %{{.+}}, %[[#A]]

  test = (h2 < h0);
  //      NONATIVE: %[[#A:]] = cir.cmp(lt, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(lt, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp olt half %{{.+}}, %{{.+}}

  // NATIVE-LLVM: %{{.+}} = fcmp olt half %{{.+}}, %{{.+}}

  test = (h2 < (_Float16)42.0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cmp(lt, %{{.+}}, %[[#B]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cmp(lt, %{{.+}}, %[[#B]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp olt half %{{.+}}, 0xH5140

  // NATIVE-LLVM: %{{.+}} = fcmp olt half %{{.+}}, 0xH5140

  test = (h2 < f0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(lt, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(lt, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM: %{{.+}} = fcmp olt float %[[#A]], %{{.+}}

  // NATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM: %{{.+}} = fcmp olt float %[[#A]], %{{.+}}

  test = (f2 < h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(lt, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(lt, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#A:]] = fpext half %{{.=}} to float
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp olt float %{{.+}}, %[[#A]]

  //      NATIVE-LLVM: %[[#A:]] = fpext half %{{.=}} to float
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp olt float %{{.+}}, %[[#A]]

  test = (i0 < h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NONATIVE: %[[#B:]] = cir.cmp(lt, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.cmp(lt, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM: %{{.+}} = fcmp olt half %[[#A]], %{{.+}}

  // NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM: %{{.+}} = fcmp olt half %[[#A]], %{{.+}}

  test = (h0 < i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(lt, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(lt, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp olt half %{{.+}}, %[[#A]]

  //      NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp olt half %{{.+}}, %[[#A]]

  test = (h0 > h2);
  //      NONATIVE: %[[#A:]] = cir.cmp(gt, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(gt, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp ogt half %{{.+}}, %{{.+}}

  // NATIVE-LLVM: %{{.+}} = fcmp ogt half %{{.+}}, %{{.+}}

  test = ((_Float16)42.0 > h2);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  //      NONATIVE: %[[#C:]] = cir.cmp(gt, %[[#B]], %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  //      NATIVE: %[[#C:]] = cir.cmp(gt, %[[#B]], %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp ogt half 0xH5140, %{{.+}}

  // NATIVE-LLVM: %{{.+}} = fcmp ogt half 0xH5140, %{{.+}}

  test = (h0 > f2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(gt, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(gt, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.=}} to float
  // NONATIVE-LLVM: %{{.+}} = fcmp ogt float %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = fpext half %{{.=}} to float
  // NATIVE-LLVM: %{{.+}} = fcmp ogt float %[[#LHS]], %{{.+}}

  test = (f0 > h2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(gt, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(gt, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp ogt float %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp ogt float %{{.+}}, %[[#RHS]]

  test = (i0 > h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NONATIVE: %[[#B:]] = cir.cmp(gt, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.cmp(gt, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM: %{{.+}} = fcmp ogt half %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM: %{{.+}} = fcmp ogt half %[[#LHS]], %{{.+}}

  test = (h0 > i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NONATIVE: %[[#B:]] = cir.cmp(gt, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(gt, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp ogt half %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp ogt half %{{.+}}, %[[#RHS]]

  test = (h2 <= h0);
  //      NONATIVE: %[[#A:]] = cir.cmp(le, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(le, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp ole half %{{.+}}, %{{.+}}

  // NATIVE-LLVM: %{{.+}} = fcmp ole half %{{.+}}, %{{.+}}

  test = (h2 <= (_Float16)42.0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cmp(le, %{{.+}}, %[[#B]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cmp(le, %{{.+}}, %[[#B]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp ole half %{{.+}}, 0xH5140

  // NATIVE-LLVM: %{{.+}} = fcmp ole half %{{.+}}, 0xH5140

  test = (h2 <= f0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(le, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(le, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM: %{{.+}} = fcmp ole float %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM: %{{.+}} = fcmp ole float %[[#LHS]], %{{.+}}

  test = (f2 <= h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(le, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(le, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp ole float %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp ole float %{{.+}}, %[[#RHS]]

  test = (i0 <= h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NONATIVE: %[[#B:]] = cir.cmp(le, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.cmp(le, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM: %{{.+}} = fcmp ole half %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM: %{{.+}} = fcmp ole half %[[#LHS]], %{{.+}}

  test = (h0 <= i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(le, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(le, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp ole half %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp ole half %{{.+}}, %[[#RHS]]

  test = (h0 >= h2);
  //      NONATIVE: %[[#A:]] = cir.cmp(ge, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i
  // NONATIVE-NEXT: %{{.+}} = cir.get_global @test : !cir.ptr<!u32i>

  //      NATIVE: %[[#A:]] = cir.cmp(ge, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp oge half %{{.+}}, %{{.+}}

  // NATIVE-LLVM: %{{.+}} = fcmp oge half %{{.+}}, %{{.+}}

  test = (h0 >= (_Float16)-2.0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.double, !cir.double
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.cmp(ge, %{{.+}}, %[[#C]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#D]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.double, !cir.double
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.f16
  // NATIVE-NEXT: %[[#D:]] = cir.cmp(ge, %{{.+}}, %[[#C]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#D]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp oge half %{{.+}}, 0xHC000

  // NATIVE-LLVM: %{{.+}} = fcmp oge half %{{.+}}, 0xHC000

  test = (h0 >= f2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(ge, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(ge, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM: %{{.+}} = fcmp oge float %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM: %{{.+}} = fcmp oge float %[[#LHS]], %{{.+}}

  test = (f0 >= h2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ge, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(ge, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp oge float %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp oge float %{{.+}}, %[[#RHS]]

  test = (i0 >= h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NONATIVE: %[[#B:]] = cir.cmp(ge, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.cmp(ge, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM: %{{.+}} = fcmp oge half %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM: %{{.+}} = fcmp oge half %[[#LHS]], %{{.+}}

  test = (h0 >= i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ge, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(ge, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp oge half %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp oge half %{{.+}}, %[[#RHS]]

  test = (h1 == h2);
  //      NONATIVE: %[[#A:]] = cir.cmp(eq, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(eq, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp oeq half %{{.+}}, %{{.+}}

  // NATIVE-LLVM: %{{.+}} = fcmp oeq half %{{.+}}, %{{.+}}

  test = (h1 == (_Float16)1.0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cmp(eq, %{{.+}}, %[[#B]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cmp(eq, %{{.+}}, %[[#B]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp oeq half %{{.+}}, 0xH3C00

  // NATIVE-LLVM: %{{.+}} = fcmp oeq half %{{.+}}, 0xH3C00

  test = (h1 == f1);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(eq, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(eq, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM: %{{.+}} = fcmp oeq float %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM: %{{.+}} = fcmp oeq float %[[#LHS]], %{{.+}}

  test = (f1 == h1);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(eq, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(eq, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp oeq float %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp oeq float %{{.+}}, %[[#RHS]]

  test = (i0 == h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NONATIVE: %[[#B:]] = cir.cmp(eq, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.cmp(eq, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM: %{{.+}} = fcmp oeq half %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM: %{{.+}} = fcmp oeq half %[[#LHS]], %{{.+}}

  test = (h0 == i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(eq, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(eq, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM-NEXT: %{{.=}} = fcmp oeq half %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM-NEXT: %{{.=}} = fcmp oeq half %{{.+}}, %[[#RHS]]

  test = (h1 != h2);
  //      NONATIVE: %[[#A:]] = cir.cmp(ne, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(ne, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp une half %{{.+}}, %{{.+}}

  // NATIVE-LLVM: %{{.+}} = fcmp une half %{{.+}}, %{{.+}}

  test = (h1 != (_Float16)1.0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cmp(ne, %{{.+}}, %[[#B]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp une half %{{.+}}, 0xH3C00

  // NATIVE-LLVM: %{{.+}} = fcmp une half %{{.+}}, 0xH3C00

  test = (h1 != f1);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(ne, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(ne, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM: %{{.+}} = fcmp une float %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = fpext half %{{.=}} to float
  // NATIVE-LLVM: %{{.+}} = fcmp une float %[[#LHS]], %{{.+}}

  test = (f1 != h1);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp une float %{{.+}}, %[[#A]]

  //      NATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp une float %{{.+}}, %[[#A]]

  test = (i0 != h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NONATIVE: %[[#B:]] = cir.cmp(ne, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.cmp(ne, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM: %{{.+}} = fcmp une half %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM: %{{.+}} = fcmp une half %[[#LHS]], %{{.+}}

  test = (h0 != i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp une half %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp une half %{{.+}}, %[[#RHS]]

  h1 = (h1 ? h2 : h0);
  //      NONATIVE: %[[#A:]] = cir.cast(float_to_bool, %{{.+}} : !cir.f16), !cir.bool
  // NONATIVE-NEXT: %{{.+}} = cir.ternary(%[[#A]], true {
  //      NONATIVE:   cir.yield %{{.+}} : !cir.f16
  // NONATIVE-NEXT: }, false {
  //      NONATIVE:   cir.yield %{{.+}} : !cir.f16
  // NONATIVE-NEXT: }) : (!cir.bool) -> !cir.f16
  //      NONATIVE: %{{.+}} = cir.get_global @h1 : !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(float_to_bool, %{{.+}} : !cir.f16), !cir.bool
  // NATIVE-NEXT: %[[#B:]] = cir.ternary(%[[#A]], true {
  //      NATIVE:   cir.yield %{{.+}} : !cir.f16
  // NATIVE-NEXT: }, false {
  //      NATIVE:   cir.yield %{{.+}} : !cir.f16
  // NATIVE-NEXT: }) : (!cir.bool) -> !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.get_global @h1 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#B]], %[[#C]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM:   %[[#A:]] = fcmp une half %{{.+}}, 0xH0000
  // NONATIVE-LLVM-NEXT:   br i1 %[[#A]], label %[[#LABEL_A:]], label %[[#LABEL_B:]]
  //      NONATIVE-LLVM: [[#LABEL_A]]:
  // NONATIVE-LLVM-NEXT:   %[[#B:]] = load volatile half, ptr @h2, align 2
  // NONATIVE-LLVM-NEXT:   br label %[[#LABEL_C:]]
  //      NONATIVE-LLVM: [[#LABEL_B]]:
  // NONATIVE-LLVM-NEXT:   %[[#C:]] = load volatile half, ptr @h0, align 2
  // NONATIVE-LLVM-NEXT:   br label %[[#LABEL_C]]
  //      NONATIVE-LLVM: [[#LABEL_C]]:
  // NONATIVE-LLVM-NEXT:   %8 = phi half [ %[[#C]], %[[#LABEL_B]] ], [ %[[#B]], %[[#LABEL_A]] ]

  //      NATIVE-LLVM:   %[[#A:]] = fcmp une half %{{.+}}, 0xH0000
  // NATIVE-LLVM-NEXT:   br i1 %[[#A]], label %[[#LABEL_A:]], label %[[#LABEL_B:]]
  //      NATIVE-LLVM: [[#LABEL_A]]:
  // NATIVE-LLVM-NEXT:   %[[#B:]] = load volatile half, ptr @h2, align 2
  // NATIVE-LLVM-NEXT:   br label %[[#LABEL_C:]]
  //      NATIVE-LLVM: [[#LABEL_B]]:
  // NATIVE-LLVM-NEXT:   %[[#C:]] = load volatile half, ptr @h0, align 2
  // NATIVE-LLVM-NEXT:   br label %[[#LABEL_C]]
  //      NATIVE-LLVM: [[#LABEL_C]]:
  // NATIVE-LLVM-NEXT:   %8 = phi half [ %[[#C]], %[[#LABEL_B]] ], [ %[[#B]], %[[#LABEL_A]] ]

  h0 = h1;
  //      NONATIVE: %[[#A:]] = cir.get_global @h1 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: cir.store volatile %[[#B]], %[[#C]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.get_global @h1 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#B]], %[[#C]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#A:]] = load volatile half, ptr @h1, align 2
  // NONATIVE-LLVM-NEXT: store volatile half %[[#A]], ptr @h0, align 2

  //      NATIVE-LLVM: %[[#A:]] = load volatile half, ptr @h1, align 2
  // NATIVE-LLVM-NEXT: store volatile half %[[#A]], ptr @h0, align 2

  h0 = (_Float16)-2.0f;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  // NONATIVE-LLVM: store volatile half 0xHC000, ptr @h0, align 2

  // NATIVE-LLVM: store volatile half 0xHC000, ptr @h0, align 2

  h0 = f0;
  //      NONATIVE: %[[#A:]] = cir.get_global @f0 : !cir.ptr<!cir.float>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.float>, !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.get_global @f0 : !cir.ptr<!cir.float>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.float>, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#A:]] = load volatile float, ptr @f0, align 4
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fptrunc float %[[#A]] to half
  // NONATIVE-LLVM-NEXT: store volatile half %[[#B]], ptr @h0, align 2

  //      NATIVE-LLVM: %[[#A:]] = load volatile float, ptr @f0, align 4
  // NATIVE-LLVM-NEXT: %[[#B:]] = fptrunc float %[[#A]] to half
  // NATIVE-LLVM-NEXT: store volatile half %[[#B]], ptr @h0, align 2

  h0 = i0;
  //      NONATIVE: %[[#A:]] = cir.get_global @i0 : !cir.ptr<!s32i>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!s32i>, !s32i
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(int_to_float, %[[#B]] : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.get_global @i0 : !cir.ptr<!s32i>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!s32i>, !s32i
  // NATIVE-NEXT: %[[#C:]] = cir.cast(int_to_float, %[[#B]] : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#A:]] = load volatile i32, ptr @i0, align 4
  // NONATIVE-LLVM-NEXT: %[[#B:]] = sitofp i32 %[[#A]] to half
  // NONATIVE-LLVM-NEXT: store volatile half %[[#B]], ptr @h0, align 2

  //      NATIVE-LLVM: %[[#A:]] = load volatile i32, ptr @i0, align 4
  // NATIVE-LLVM-NEXT: %[[#B:]] = sitofp i32 %[[#A]] to half
  // NATIVE-LLVM-NEXT: store volatile half %[[#B]], ptr @h0, align 2

  i0 = h0;
  //      NONATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.f16), !s32i
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @i0 : !cir.ptr<!s32i>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.f16), !s32i
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @i0 : !cir.ptr<!s32i>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !s32i, !cir.ptr<!s32i>

  //      NONATIVE-LLVM: %[[#A:]] = load volatile half, ptr @h0, align 2
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fptosi half %[[#A]] to i32
  // NONATIVE-LLVM-NEXT: store volatile i32 %[[#B]], ptr @i0, align 4

  //      NATIVE-LLVM: %[[#A:]] = load volatile half, ptr @h0, align 2
  // NATIVE-LLVM-NEXT: %[[#B:]] = fptosi half %[[#A]] to i32
  // NATIVE-LLVM-NEXT: store volatile i32 %[[#B]], ptr @i0, align 4

  h0 += h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(add, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.binop(add, %{{.+}}, %{{.+}}) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fadd float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  // NATIVE-LLVM: %{{.+}} = fadd half %{{.+}}, %{{.+}}

  h0 += (_Float16)1.0f;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#D:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.binop(add, %[[#D]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.cast(floating, %[[#E]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#F]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.float), !cir.f16
  //      NATIVE: %[[#C:]] = cir.binop(add, %{{.+}}, %[[#B]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fadd float %[[#A]], 1.000000e+00
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to half

  // NATIVE-LLVM: %{{.+}} = fadd half %{{.+}}, 0xH3C00

  h0 += f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fadd float %[[#LHS]], %{{.+}}
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#RES:]] = fadd float %[[#LHS]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  i0 += h0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(add, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(float_to_int, %[[#C]] : !cir.float), !s32i
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.f16), !s32i
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NONATVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  //      NONATVE-LLVM: %[[#LHS:]] = sitofp i32 %3 to float
  // NONATVE-LLVM-NEXT: %[[#RES:]] = fadd float %[[#LHS]], %[[#RHS]]
  // NONATVE-LLVM-NEXT: %{{.+}} = fptosi float %[[#RES]] to i32

  //      NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM-NEXT: %[[#B:]] = fadd half %[[#A]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptosi half %[[#B]] to i32

  h0 += i0;
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(add, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fpext half %[[#A]] to float
  //      NONATIVE-LLVM: %[[#C:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#D:]] = fadd float %[[#C]], %[[#B]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#D]] to half

  // NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM: %{{.+}} = fadd half %{{.+}}, %[[#A]]

  h0 -= h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(sub, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  // NATIVE-LLVM: %{{.+}} = fsub half %{{.+}}, %{{.+}}

  h0 -= (_Float16)1.0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(sub, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  //      NATIVE: %[[#C:]] = cir.binop(sub, %{{.+}}, %[[#B]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fsub float %[[#A]], 1.000000e+00
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to half

  // NATIVE-LLVM: %{{.+}} = fsub half %{{.+}}, 0xH3C00

  h0 -= f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %[[#LHS]], %{{.+}}
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %[[#LHS]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  i0 -= h0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(sub, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(float_to_int, %[[#C]] : !cir.float), !s32i
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.f16), !s32i
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NONATVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  //      NONATVE-LLVM: %[[#LHS:]] = sitofp i32 %3 to float
  // NONATVE-LLVM-NEXT: %[[#RES:]] = fsub float %[[#LHS]], %[[#RHS]]
  // NONATVE-LLVM-NEXT: %{{.+}} = fptosi float %[[#RES]] to i32

  //      NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM-NEXT: %[[#B:]] = fsub half %[[#A]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptosi half %[[#B]] to i32

  h0 -= i0;
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(sub, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fpext half %[[#A]] to float
  //      NONATIVE-LLVM: %[[#C:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#D:]] = fsub float %[[#C]], %[[#B]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#D]] to half

  // NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM: %{{.+}} = fsub half %{{.+}}, %[[#A]]

  h0 *= h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(mul, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.binop(mul, %{{.+}}, %{{.+}}) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fmul float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  // NATIVE-LLVM: %{{.+}} = fmul half %{{.+}}, %{{.+}}

  h0 *= (_Float16)1.0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(mul, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  //      NATIVE: %[[#C:]] = cir.binop(mul, %{{.+}}, %[[#B]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fmul float %[[#A]], 1.000000e+00
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to half

  // NATIVE-LLVM: %{{.+}} = fmul half %{{.+}}, 0xH3C00

  h0 *= f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fmul float %[[#LHS]], %{{.+}}
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#RES:]] = fmul float %[[#LHS]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  i0 *= h0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(mul, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(float_to_int, %[[#C]] : !cir.float), !s32i
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.f16), !s32i
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NONATVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  //      NONATVE-LLVM: %[[#LHS:]] = sitofp i32 %3 to float
  // NONATVE-LLVM-NEXT: %[[#RES:]] = fmul float %[[#LHS]], %[[#RHS]]
  // NONATVE-LLVM-NEXT: %{{.+}} = fptosi float %[[#RES]] to i32

  //      NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM-NEXT: %[[#B:]] = fmul half %[[#A]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptosi half %[[#B]] to i32

  h0 *= i0;
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(mul, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fpext half %[[#A]] to float
  //      NONATIVE-LLVM: %[[#C:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#D:]] = fmul float %[[#C]], %[[#B]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#D]] to half

  // NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM: %{{.+}} = fmul half %{{.+}}, %[[#A]]

  h0 /= h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(div, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.binop(div, %{{.+}}, %{{.+}}) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fdiv float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  // NATIVE-LLVM: %{{.+}} = fdiv half %{{.+}}, %{{.+}}

  h0 /= (_Float16)1.0;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#D:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.binop(div, %[[#D]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.cast(floating, %[[#E]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#F]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  //      NATIVE: %[[#C:]] = cir.binop(div, %{{.+}}, %[[#B]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#A:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fdiv float %[[#A]], 1.000000e+00
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to half

  // NATIVE-LLVM: %{{.+}} = fdiv half %{{.+}}, 0xH3C00

  h0 /= f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fdiv float %[[#LHS]], %{{.+}}
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  //      NATIVE-LLVM: %[[#LHS:]] = fpext half %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#RES:]] = fdiv float %[[#LHS]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to half

  i0 /= h0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(div, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(float_to_int, %[[#C]] : !cir.float), !s32i
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.f16), !s32i
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NONATVE-LLVM: %[[#RHS:]] = fpext half %{{.+}} to float
  //      NONATVE-LLVM: %[[#LHS:]] = sitofp i32 %3 to float
  // NONATVE-LLVM-NEXT: %[[#RES:]] = fdiv float %[[#LHS]], %[[#RHS]]
  // NONATVE-LLVM-NEXT: %{{.+}} = fptosi float %[[#RES]] to i32

  //      NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM-NEXT: %[[#B:]] = fdiv half %[[#A]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptosi half %[[#B]] to i32

  h0 /= i0;
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(div, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fpext half %[[#A]] to float
  //      NONATIVE-LLVM: %[[#C:]] = fpext half %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#D:]] = fdiv float %[[#C]], %[[#B]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#D]] to half

  // NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to half
  // NATIVE-LLVM: %{{.+}} = fdiv half %{{.+}}, %[[#A]]

  h0 = d0;
  //      NONATIVE: %[[#A:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.double>, !cir.double
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.double>, !cir.double
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.f16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#A:]] = load volatile double, ptr @d0, align 8
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fptrunc double %[[#A]] to half
  // NONATIVE-LLVM-NEXT: store volatile half %[[#B]], ptr @h0, align 2

  //      NATIVE-LLVM: %[[#A:]] = load volatile double, ptr @d0, align 8
  // NATIVE-LLVM-NEXT: %[[#B:]] = fptrunc double %[[#A]] to half
  // NATIVE-LLVM-NEXT: store volatile half %[[#B]], ptr @h0, align 2

  h0 = (float)d0;
  //      NONATIVE: %[[#A:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.double>, !cir.double
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: %[[#E:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %[[#E]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.double>, !cir.double
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.float
  // NATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NATIVE-NEXT: %[[#E:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#D]], %[[#E]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#A:]] = load volatile double, ptr @d0, align 8
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fptrunc double %[[#A]] to float
  // NONATIVE-LLVM-NEXT: %[[#C:]] = fptrunc float %[[#B]] to half
  // NONATIVE-LLVM-NEXT: store volatile half %[[#C]], ptr @h0, align 2

  //      NATIVE-LLVM: %[[#A:]] = load volatile double, ptr @d0, align 8
  // NATIVE-LLVM-NEXT: %[[#B:]] = fptrunc double %[[#A]] to float
  // NATIVE-LLVM-NEXT: %[[#C:]] = fptrunc float %[[#B]] to half
  // NATIVE-LLVM-NEXT: store volatile half %[[#C]], ptr @h0, align 2

  d0 = h0;
  //      NONATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.double
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.double, !cir.ptr<!cir.double>

  //      NATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.double
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.double, !cir.ptr<!cir.double>

  //      NONATVE-LLVM: %[[#A:]] = load volatile half, ptr @h0, align 2
  // NONATVE-LLVM-NEXT: %[[#B:]] = fpext half %[[#A]] to double
  // NONATVE-LLVM-NEXT: store volatile double %[[#B]], ptr @d0, align 8

  //      NATIVE-LLVM: %[[#A:]] = load volatile half, ptr @h0, align 2
  // NATIVE-LLVM-NEXT: %[[#B:]] = fpext half %[[#A]] to double
  // NATIVE-LLVM-NEXT: store volatile double %[[#B]], ptr @d0, align 8

  d0 = (float)h0;
  //      NONATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.double
  // NONATIVE-NEXT: %[[#E:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %[[#E]] : !cir.double, !cir.ptr<!cir.double>

  //      NATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.double
  // NATIVE-NEXT: %[[#E:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NATIVE-NEXT: cir.store volatile %[[#D]], %[[#E]] : !cir.double, !cir.ptr<!cir.double>

  //      NONATVE-LLVM: %[[#A:]] = load volatile half, ptr @h0, align 2
  // NONATVE-LLVM-NEXT: %[[#B:]] = fpext half %[[#A]] to float
  // NONATVE-LLVM-NEXT: %[[#C:]] = fpext float %[[#B]] to double
  // NONATVE-LLVM-NEXT: store volatile double %[[#C]], ptr @d0, align 8

  //      NATIVE-LLVM: %[[#A:]] = load volatile half, ptr @h0, align 2
  // NATIVE-LLVM-NEXT: %[[#B:]] = fpext half %[[#A]] to float
  // NATIVE-LLVM-NEXT: %[[#C:]] = fpext float %[[#B]] to double
  // NATIVE-LLVM-NEXT: store volatile double %[[#C]], ptr @d0, align 8

  h0 = s0;
  //      NONATIVE: %[[#A:]] = cir.get_global @s0 : !cir.ptr<!s16i>
  // NONATIVE-NEXT: %[[#B:]] = cir.load %[[#A]] : !cir.ptr<!s16i>, !s16i
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(int_to_float, %[[#B]] : !s16i), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.get_global @s0 : !cir.ptr<!s16i>
  // NATIVE-NEXT: %[[#B:]] = cir.load %[[#A]] : !cir.ptr<!s16i>, !s16i
  // NATIVE-NEXT: %[[#C:]] = cir.cast(int_to_float, %[[#B]] : !s16i), !cir.f16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NONATIVE-LLVM: %[[#A:]] = load i16, ptr @s0, align 2
  // NONATIVE-LLVM-NEXT: %[[#B:]] = sitofp i16 %[[#A]] to half
  // NONATIVE-LLVM-NEXT: store volatile half %[[#B]], ptr @h0, align 2

  //      NATIVE-LLVM: %[[#A:]] = load i16, ptr @s0, align 2
  // NATIVE-LLVM-NEXT: %[[#B:]] = sitofp i16 %[[#A]] to half
  // NATIVE-LLVM-NEXT: store volatile half %[[#B]], ptr @h0, align 2
}
