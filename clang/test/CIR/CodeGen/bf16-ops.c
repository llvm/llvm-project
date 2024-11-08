// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefix=NONATIVE %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +fullbf16 -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefix=NATIVE %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll --check-prefix=NONATIVE-LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +fullbf16 -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll --check-prefix=NATIVE-LLVM %s

volatile unsigned test;
volatile int i0;
volatile __bf16 h0 = 0.0, h1 = 1.0, h2;
volatile float f0, f1, f2;
volatile double d0;
short s0;

void foo(void) {
  test = (h0);
  // NONATIVE: %{{.+}} = cir.cast(float_to_int, %{{.+}} : !cir.bf16), !u32i
  // NATIVE: %{{.+}} = cir.cast(float_to_int, %{{.+}} : !cir.bf16), !u32i

  // NONATIVE-LLVM: %{{.+}} = fptoui bfloat %{{.+}} to i32
  // NATIVE-LLVM: %{{.+}} = fptoui bfloat %{{.+}} to i32

  h0 = (test);
  // NONATIVE: %{{.+}} = cir.cast(int_to_float, %{{.+}} : !u32i), !cir.bf16
  // NATIVE: %{{.+}} = cir.cast(int_to_float, %{{.+}} : !u32i), !cir.bf16

  // NONATIVE-LLVM: %{{.+}} = uitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM: %{{.+}} = uitofp i32 %{{.+}} to bfloat

  test = (!h1);
  //      NONATIVE: %[[#A:]] = cir.cast(float_to_bool, %{{.+}} : !cir.bf16), !cir.bool
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(not, %[[#A]]) : !cir.bool, !cir.bool
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(bool_to_int, %[[#B]] : !cir.bool), !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(float_to_bool, %{{.+}} : !cir.bf16), !cir.bool
  // NATIVE-NEXT: %[[#B:]] = cir.unary(not, %[[#A]]) : !cir.bool, !cir.bool
  // NATIVE-NEXT: %[[#C:]] = cir.cast(bool_to_int, %[[#B]] : !cir.bool), !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#A:]] = fcmp une bfloat %{{.+}}, 0xR0000
  // NONATIVE-LLVM-NEXT: %[[#B:]] = zext i1 %[[#A]] to i8
  // NONATIVE-LLVM-NEXT: %[[#C:]] = xor i8 %[[#B]], 1
  // NONATIVE-LLVM-NEXT: %{{.+}} = zext i8 %[[#C]] to i32

  //      NATIVE-LLVM: %[[#A:]] = fcmp une bfloat %{{.+}}, 0xR0000
  // NATIVE-LLVM-NEXT: %[[#B:]] = zext i1 %[[#A]] to i8
  // NATIVE-LLVM-NEXT: %[[#C:]] = xor i8 %[[#B]], 1
  // NATIVE-LLVM-NEXT: %{{.+}} = zext i8 %[[#C]] to i32

  h1 = -h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //  NATIVE-NOT: %{{.+}} = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //  NATIVE-NOT: %{{.+}} = cir.cast(floating, %{{.+}} : !cir.float), !cir.bf16
  //      NATIVE: %{{.+}} = cir.unary(minus, %{{.+}}) : !cir.bf16, !cir.bf16

  //      NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fneg float %[[#A]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fneg bfloat %{{.+}}

  h1 = +h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(plus, %[[#A]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //  NATIVE-NOT: %{{.+}} = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //  NATIVE-NOT: %{{.+}} = cir.cast(floating, %{{.+}} : !cir.float), !cir.bf16
  //      NATIVE: %{{.+}} = cir.unary(plus, %{{.+}}) : !cir.bf16, !cir.bf16

  //      NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#A]] to bfloat

  //      NATIVE-LLVM: %[[#A:]] = load volatile bfloat, ptr @h1, align 2
  // NATIVE-LLVM-NEXT: store volatile bfloat %[[#A]], ptr @h1, align 2

  h1++;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.bf16
  // NONATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.bf16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.bf16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.bf16

  // NONATIVE-LLVM: %{{.+}} = fadd bfloat %{{.+}}, 0xR3F80

  // NATIVE-LLVM: %{{.+}} = fadd bfloat %{{.+}}, 0xR3F80

  ++h1;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.bf16
  // NONATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.bf16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.bf16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.bf16

  // NONATIVE-LLVM: %{{.+}} = fadd bfloat %{{.+}}, 0xR3F80

  // NATIVE-LLVM: %{{.+}} = fadd bfloat %{{.+}}, 0xR3F80

  --h1;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<-1.000000e+00> : !cir.bf16
  // NONATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.bf16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<-1.000000e+00> : !cir.bf16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.bf16

  // NONATIVE-LLVM: %{{.+}} = fadd bfloat %{{.+}}, 0xRBF80

  // NATIVE-LLVM: %{{.+}} = fadd bfloat %{{.+}}, 0xRBF80

  h1--;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<-1.000000e+00> : !cir.bf16
  // NONATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.bf16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<-1.000000e+00> : !cir.bf16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.bf16

  // NONATIVE-LLVM: %{{.+}} = fadd bfloat %{{.+}}, 0xRBF80

  // NATIVE-LLVM: %{{.+}} = fadd bfloat %{{.+}}, 0xRBF80

  h1 = h0 * h2;
  //      NONATIVE: %[[#LHS:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#RHS:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#A:]] = cir.binop(mul, %[[#LHS]], %[[#RHS]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#A]] : !cir.float), !cir.bf16

  // NATIVE: %{{.+}} = cir.binop(mul, %{{.+}}, %{{.+}}) : !cir.bf16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fmul float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fmul bfloat %{{.+}}, %{{.+}}

  h1 = h0 * (__bf16) -2.0f;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.unary(minus, %[[#B]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.binop(mul, %[[#A]], %[[#E]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#F]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NATIVE-NEXT: %{{.+}} = cir.binop(mul, %{{.+}}, %[[#C]]) : !cir.bf16

  //      NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fmul float %[[#A]], -2.000000e+00
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fmul bfloat %{{.+}}, 0xRC000

  h1 = h0 * f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NATIVE: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RES:]] = fmul float %[[#LHS]], %{{.+}}
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  //      NATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  //      NATIVE-LLVM: %[[#RES:]] = fmul float %[[#LHS]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  h1 = f0 * h2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fmul float %{{.+}}, %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  //      NATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#RES:]] = fmul float %{{.+}}, %[[#RHS]]
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  h1 = h0 * i0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(mul, %[[#A]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#D]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NATIVE-NEXT: %{{.+}} = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.bf16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM-NEXT: %[[#A:]] = fpext bfloat %[[#RHS]] to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fmul float %[[#LHS]], %[[#A]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  //      NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM-NEXT: %{{.+}} = fmul bfloat %{{.+}}, %[[#A]]

  h1 = (h0 / h2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(div, %[[#A]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#C]] : !cir.float), !cir.bf16

  // NATIVE: %{{.+}} = cir.binop(div, %{{.+}}, %{{.+}}) : !cir.bf16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fdiv float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fdiv bfloat %{{.+}}, %{{.+}}

  h1 = (h0 / (__bf16) -2.0f);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.unary(minus, %[[#B]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.binop(div, %[[#A]], %[[#E]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#F]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NATIVE-NEXT: %{{.+}} = cir.binop(div, %{{.+}}, %[[#C]]) : !cir.bf16

  //      NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fdiv float %[[#A]], -2.000000e+00
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fdiv bfloat %{{.+}}, 0xRC000

  h1 = (h0 / f2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NATIVE: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RES:]] = fdiv float %[[#LHS]], %{{.+}}
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  //      NATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  //      NATIVE-LLVM: %[[#RES:]] = fdiv float %[[#LHS]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  h1 = (f0 / h2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fdiv float %{{.+}}, %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  //      NATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#RES:]] = fdiv float %{{.+}}, %[[#RHS]]
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  h1 = (h0 / i0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(div, %[[#A]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#D]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NATIVE-NEXT: %{{.+}} = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.bf16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM-NEXT: %[[#A:]] = fpext bfloat %[[#RHS]] to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fdiv float %[[#LHS]], %[[#A]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  //      NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM-NEXT: %{{.+}} = fdiv bfloat %{{.+}}, %[[#A]]

  h1 = (h2 + h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(add, %[[#A]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#C]] : !cir.float), !cir.bf16

  // NATIVE: %{{.+}} = cir.binop(add, %{{.+}}, %{{.+}}) : !cir.bf16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fadd float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fadd bfloat %{{.+}}, %{{.+}}

  h1 = ((__bf16)-2.0 + h0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.double, !cir.double
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.bf16
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.bf16), !cir.float
  //      NONATIVE: %[[#E:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.binop(add, %[[#D]], %[[#E]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#F]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.double, !cir.double
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.bf16
  //      NATIVE: %{{.+}} = cir.binop(add, %[[#C]], %{{.+}}) : !cir.bf16

  //      NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fadd float -2.000000e+00, %[[#A]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fadd bfloat 0xRC000, %{{.+}}

  h1 = (h2 + f0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NATIVE: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RHS:]] = load volatile float, ptr @f0, align 4
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fadd float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  //      NATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#RHS:]] = load volatile float, ptr @f0, align 4
  // NATIVE-LLVM-NEXT: %[[#RES:]] = fadd float %[[#LHS]], %[[#RHS]]
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  h1 = (f2 + h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fadd float %{{.+}}, %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  //      NATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#RES:]] = fadd float %{{.+}}, %[[#RHS]]
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  h1 = (h0 + i0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(add, %[[#A]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#D]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.bf16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS_INT:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM-NEXT: %[[#RHS:]] = fpext bfloat %[[#RHS_INT]] to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fadd float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  //      NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM-NEXT: %{{.+}} = fadd bfloat %{{.+}}, %[[#A]]

  h1 = (h2 - h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(sub, %[[#A]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#C]] : !cir.float), !cir.bf16

  // NATIVE: %{{.+}} = cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.bf16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fsub bfloat %{{.+}}, %{{.+}}

  h1 = ((__bf16)-2.0f - h0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.bf16), !cir.float
  //      NONATIVE: %[[#E:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.binop(sub, %[[#D]], %[[#E]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#F]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  //      NATIVE: %{{.+}} = cir.binop(sub, %[[#C]], %{{.+}}) : !cir.bf16

  //      NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fsub float -2.000000e+00, %[[#A]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fsub bfloat 0xRC000, %{{.+}}

  h1 = (h2 - f0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NATIVE: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RHS:]] = load volatile float, ptr @f0, align 4
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  //      NATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#RHS:]] = load volatile float, ptr @f0, align 4
  // NATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %[[#LHS]], %[[#RHS]]
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  h1 = (f2 - h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %{{.+}}, %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  //      NATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %{{.+}}, %[[#RHS]]
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  h1 = (h0 - i0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(sub, %[[#A]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#D]] : !cir.float), !cir.bf16

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NATIVE-NEXT: %{{.+}} = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.bf16

  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#RHS_INT:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM-NEXT: %[[#RHS:]] = fpext bfloat %[[#RHS_INT]] to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  //      NATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM-NEXT: %{{.+}} = fsub bfloat %{{.+}}, %[[#A]]

  test = (h2 < h0);
  //      NONATIVE: %[[#A:]] = cir.cmp(lt, %{{.+}}, %{{.+}}) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(lt, %{{.+}}, %{{.+}}) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp olt bfloat %{{.+}}, %{{.+}}

  // NATIVE-LLVM: %{{.+}} = fcmp olt bfloat %{{.+}}, %{{.+}}

  test = (h2 < (__bf16)42.0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.bf16
  // NONATIVE-NEXT: %[[#C:]] = cir.cmp(lt, %{{.+}}, %[[#B]]) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.bf16
  // NATIVE-NEXT: %[[#C:]] = cir.cmp(lt, %{{.+}}, %[[#B]]) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp olt bfloat %{{.+}}, 0xR4228

  // NATIVE-LLVM: %{{.+}} = fcmp olt bfloat %{{.+}}, 0xR4228

  test = (h2 < f0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(lt, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(lt, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM: %{{.+}} = fcmp olt float %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM: %{{.+}} = fcmp olt float %[[#LHS]], %{{.+}}

  test = (f2 < h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(lt, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(lt, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp olt float %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp olt float %{{.+}}, %[[#RHS]]

  test = (i0 < h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NONATIVE: %[[#B:]] = cir.cmp(lt, %[[#A]], %{{.+}}) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NATIVE: %[[#B:]] = cir.cmp(lt, %[[#A]], %{{.+}}) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM: %{{.+}} = fcmp olt bfloat %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM: %{{.+}} = fcmp olt bfloat %[[#LHS]], %{{.+}}

  test = (h0 < i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(lt, %{{.+}}, %[[#A]]) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(lt, %{{.+}}, %[[#A]]) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp olt bfloat %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp olt bfloat %{{.+}}, %[[#RHS]]

  test = (h0 > h2);
  //      NONATIVE: %[[#A:]] = cir.cmp(gt, %{{.+}}, %{{.+}}) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(gt, %{{.+}}, %{{.+}}) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp ogt bfloat %{{.+}}, %{{.+}}

  // NATIVE-LLVM: %{{.+}} = fcmp ogt bfloat %{{.+}}, %{{.+}}

  test = ((__bf16)42.0 > h2);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.bf16
  //      NONATIVE: %[[#C:]] = cir.cmp(gt, %[[#B]], %{{.+}}) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.bf16
  //      NATIVE: %[[#C:]] = cir.cmp(gt, %[[#B]], %{{.+}}) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp ogt bfloat 0xR4228, %{{.+}}

  // NATIVE-LLVM: %{{.+}} = fcmp ogt bfloat 0xR4228, %{{.+}}

  test = (h0 > f2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(gt, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(gt, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM: %{{.+}} = fcmp ogt float %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM: %{{.+}} = fcmp ogt float %[[#LHS]], %{{.+}}

  test = (f0 > h2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(gt, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(gt, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM: %{{.+}} = fcmp ogt float %{{.+}}, %[[#RHS]]

  // NATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM: %{{.+}} = fcmp ogt float %{{.+}}, %[[#RHS]]

  test = (i0 > h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NONATIVE: %[[#B:]] = cir.cmp(gt, %[[#A]], %{{.+}}) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NATIVE: %[[#B:]] = cir.cmp(gt, %[[#A]], %{{.+}}) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM: %{{.+}} = fcmp ogt bfloat %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM: %{{.+}} = fcmp ogt bfloat %[[#LHS]], %{{.+}}

  test = (h0 > i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NONATIVE: %[[#B:]] = cir.cmp(gt, %{{.+}}, %[[#A]]) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(gt, %{{.+}}, %[[#A]]) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp ogt bfloat %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp ogt bfloat %{{.+}}, %[[#RHS]]

  test = (h2 <= h0);
  //      NONATIVE: %[[#A:]] = cir.cmp(le, %{{.+}}, %{{.+}}) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(le, %{{.+}}, %{{.+}}) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp ole bfloat %{{.+}}, %{{.+}}

  // NATIVE-LLVM: %{{.+}} = fcmp ole bfloat %{{.+}}, %{{.+}}

  test = (h2 <= (__bf16)42.0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.bf16
  // NONATIVE-NEXT: %[[#C:]] = cir.cmp(le, %{{.+}}, %[[#B]]) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.bf16
  // NATIVE-NEXT: %[[#C:]] = cir.cmp(le, %{{.+}}, %[[#B]]) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp ole bfloat %{{.+}}, 0xR4228

  // NATIVE-LLVM: %{{.+}} = fcmp ole bfloat %{{.+}}, 0xR4228

  test = (h2 <= f0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(le, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(le, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM: %{{.+}} = fcmp ole float %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM: %{{.+}} = fcmp ole float %[[#LHS]], %{{.+}}

  test = (f2 <= h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(le, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(le, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp ole float %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp ole float %{{.+}}, %[[#RHS]]

  test = (i0 <= h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NONATIVE: %[[#B:]] = cir.cmp(le, %[[#A]], %{{.+}}) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NATIVE: %[[#B:]] = cir.cmp(le, %[[#A]], %{{.+}}) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM: %{{.+}} = fcmp ole bfloat %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM: %{{.+}} = fcmp ole bfloat %[[#LHS]], %{{.+}}

  test = (h0 <= i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(le, %{{.+}}, %[[#A]]) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(le, %{{.+}}, %[[#A]]) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp ole bfloat %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp ole bfloat %{{.+}}, %[[#RHS]]

  test = (h0 >= h2);
  //      NONATIVE: %[[#A:]] = cir.cmp(ge, %{{.+}}, %{{.+}}) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i
  // NONATIVE-NEXT: %{{.+}} = cir.get_global @test : !cir.ptr<!u32i>

  //      NATIVE: %[[#A:]] = cir.cmp(ge, %{{.+}}, %{{.+}}) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp oge bfloat %{{.+}}, %{{.+}}

  // NATIVE-LLVM: %{{.+}} = fcmp oge bfloat %{{.+}}, %{{.+}}

  test = (h0 >= (__bf16)-2.0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.double, !cir.double
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.bf16
  // NONATIVE-NEXT: %[[#D:]] = cir.cmp(ge, %{{.+}}, %[[#C]]) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#D]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.double, !cir.double
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.bf16
  // NATIVE-NEXT: %[[#D:]] = cir.cmp(ge, %{{.+}}, %[[#C]]) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#D]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp oge bfloat %{{.+}}, 0xRC000

  // NATIVE-LLVM: %{{.+}} = fcmp oge bfloat %{{.+}}, 0xRC000

  test = (h0 >= f2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(ge, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(ge, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM: %{{.+}} = fcmp oge float %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM: %{{.+}} = fcmp oge float %[[#LHS]], %{{.+}}

  test = (f0 >= h2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ge, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(ge, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM: %{{.+}} = fcmp oge float %{{.+}}, %[[#RHS]]

  // NATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM: %{{.+}} = fcmp oge float %{{.+}}, %[[#RHS]]

  test = (i0 >= h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NONATIVE: %[[#B:]] = cir.cmp(ge, %[[#A]], %{{.+}}) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NATIVE: %[[#B:]] = cir.cmp(ge, %[[#A]], %{{.+}}) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM: %{{.+}} = fcmp oge bfloat %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM: %{{.+}} = fcmp oge bfloat %[[#LHS]], %{{.+}}

  test = (h0 >= i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ge, %{{.+}}, %[[#A]]) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(ge, %{{.+}}, %[[#A]]) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp oge bfloat %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp oge bfloat %{{.+}}, %[[#RHS]]

  test = (h1 == h2);
  //      NONATIVE: %[[#A:]] = cir.cmp(eq, %{{.+}}, %{{.+}}) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(eq, %{{.+}}, %{{.+}}) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp oeq bfloat %{{.+}}, %{{.+}}

  // NATIVE-LLVM: %{{.+}} = fcmp oeq bfloat %{{.+}}, %{{.+}}

  test = (h1 == (__bf16)1.0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.bf16
  // NONATIVE-NEXT: %[[#C:]] = cir.cmp(eq, %{{.+}}, %[[#B]]) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.bf16
  // NATIVE-NEXT: %[[#C:]] = cir.cmp(eq, %{{.+}}, %[[#B]]) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp oeq bfloat %{{.+}}, 0xR3F80

  // NATIVE-LLVM: %{{.+}} = fcmp oeq bfloat %{{.+}}, 0xR3F80

  test = (h1 == f1);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(eq, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(eq, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM: %{{.+}} = fcmp oeq float %[[#A]], %{{.+}}

  // NATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM: %{{.+}} = fcmp oeq float %[[#A]], %{{.+}}

  test = (f1 == h1);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(eq, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(eq, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp oeq float %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp oeq float %{{.+}}, %[[#RHS]]

  test = (i0 == h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NONATIVE: %[[#B:]] = cir.cmp(eq, %[[#A]], %{{.+}}) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NATIVE: %[[#B:]] = cir.cmp(eq, %[[#A]], %{{.+}}) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM: %{{.+}} = fcmp oeq bfloat %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM: %{{.+}} = fcmp oeq bfloat %[[#LHS]], %{{.+}}

  test = (h0 == i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(eq, %{{.+}}, %[[#A]]) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(eq, %{{.+}}, %[[#A]]) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp oeq bfloat %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp oeq bfloat %{{.+}}, %[[#RHS]]

  test = (h1 != h2);
  //      NONATIVE: %[[#A:]] = cir.cmp(ne, %{{.+}}, %{{.+}}) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(ne, %{{.+}}, %{{.+}}) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp une bfloat %{{.+}}, %{{.+}}

  // NATIVE-LLVM: %{{.+}} = fcmp une bfloat %{{.+}}, %{{.+}}

  test = (h1 != (__bf16)1.0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.double), !cir.bf16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.bf16
  // NATIVE-NEXT: %[[#C:]] = cir.cmp(ne, %{{.+}}, %[[#B]]) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  // NONATIVE-LLVM: %{{.+}} = fcmp une bfloat %{{.+}}, 0xR3F80

  // NATIVE-LLVM: %{{.+}} = fcmp une bfloat %{{.+}}, 0xR3F80

  test = (h1 != f1);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(ne, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(ne, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM: %{{.+}} = fcmp une float %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM: %{{.+}} = fcmp une float %[[#LHS]], %{{.+}}

  test = (f1 != h1);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp une float %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp une float %{{.+}}, %[[#RHS]]

  test = (i0 != h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NONATIVE: %[[#B:]] = cir.cmp(ne, %[[#A]], %{{.+}}) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NATIVE: %[[#B:]] = cir.cmp(ne, %[[#A]], %{{.+}}) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  // NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM: %{{.+}} = fcmp une bfloat %[[#LHS]], %{{.+}}

  // NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM: %{{.+}} = fcmp une bfloat %[[#LHS]], %{{.+}}

  test = (h0 != i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.bf16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.bf16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NONATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM-NEXT: %{{.+}} = fcmp une bfloat %{{.+}}, %[[#RHS]]

  //      NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM-NEXT: %{{.+}} = fcmp une bfloat %{{.+}}, %[[#RHS]]

  h1 = (h1 ? h2 : h0);
  //      NONATIVE: %[[#A:]] = cir.cast(float_to_bool, %{{.+}} : !cir.bf16), !cir.bool
  // NONATIVE-NEXT: %{{.+}} = cir.ternary(%[[#A]], true {
  //      NONATIVE:   cir.yield %{{.+}} : !cir.bf16
  // NONATIVE-NEXT: }, false {
  //      NONATIVE:   cir.yield %{{.+}} : !cir.bf16
  // NONATIVE-NEXT: }) : (!cir.bool) -> !cir.bf16
  //      NONATIVE: %{{.+}} = cir.get_global @h1 : !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.cast(float_to_bool, %{{.+}} : !cir.bf16), !cir.bool
  // NATIVE-NEXT: %[[#B:]] = cir.ternary(%[[#A]], true {
  //      NATIVE:   cir.yield %{{.+}} : !cir.bf16
  // NATIVE-NEXT: }, false {
  //      NATIVE:   cir.yield %{{.+}} : !cir.bf16
  // NATIVE-NEXT: }) : (!cir.bool) -> !cir.bf16
  // NATIVE-NEXT: %[[#C:]] = cir.get_global @h1 : !cir.ptr<!cir.bf16>
  // NATIVE-NEXT: cir.store volatile %[[#B]], %[[#C]] : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM:   %[[#A:]] = fcmp une bfloat %{{.+}}, 0xR0000
  // NONATIVE-LLVM-NEXT:   br i1 %[[#A]], label %[[#LABEL_A:]], label %[[#LABEL_B:]]
  //      NONATIVE-LLVM: [[#LABEL_A]]:
  // NONATIVE-LLVM-NEXT:   %[[#B:]] = load volatile bfloat, ptr @h2, align 2
  // NONATIVE-LLVM-NEXT:   br label %[[#LABEL_C:]]
  //      NONATIVE-LLVM: [[#LABEL_B]]:
  // NONATIVE-LLVM-NEXT:   %[[#C:]] = load volatile bfloat, ptr @h0, align 2
  // NONATIVE-LLVM-NEXT:   br label %[[#LABEL_C]]
  //      NONATIVE-LLVM: [[#LABEL_C]]:
  // NONATIVE-LLVM-NEXT:   %{{.+}} = phi bfloat [ %[[#C]], %[[#LABEL_B]] ], [ %[[#B]], %[[#LABEL_A]] ]

  //      NATIVE-LLVM:   %[[#A:]] = fcmp une bfloat %{{.+}}, 0xR0000
  // NATIVE-LLVM-NEXT:   br i1 %[[#A]], label %[[#LABEL_A:]], label %[[#LABEL_B:]]
  //      NATIVE-LLVM: [[#LABEL_A]]:
  // NATIVE-LLVM-NEXT:   %[[#B:]] = load volatile bfloat, ptr @h2, align 2
  // NATIVE-LLVM-NEXT:   br label %[[#LABEL_C:]]
  //      NATIVE-LLVM: [[#LABEL_B]]:
  // NATIVE-LLVM-NEXT:   %[[#C:]] = load volatile bfloat, ptr @h0, align 2
  // NATIVE-LLVM-NEXT:   br label %[[#LABEL_C]]
  //      NATIVE-LLVM: [[#LABEL_C]]:
  // NATIVE-LLVM-NEXT:   %{{.+}} = phi bfloat [ %[[#C]], %[[#LABEL_B]] ], [ %[[#B]], %[[#LABEL_A]] ]

  h0 = h1;
  //      NONATIVE: %[[#A:]] = cir.get_global @h1 : !cir.ptr<!cir.bf16>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.bf16>, !cir.bf16
  // NONATIVE-NEXT: %[[#C:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NONATIVE-NEXT: cir.store volatile %[[#B]], %[[#C]] : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.get_global @h1 : !cir.ptr<!cir.bf16>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.bf16>, !cir.bf16
  // NATIVE-NEXT: %[[#C:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NATIVE-NEXT: cir.store volatile %[[#B]], %[[#C]] : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = load volatile bfloat, ptr @h1, align 2
  // NONATIVE-LLVM-NEXT: store volatile bfloat %[[#A]], ptr @h0, align 2

  //      NATIVE-LLVM: %[[#A:]] = load volatile bfloat, ptr @h1, align 2
  // NATIVE-LLVM-NEXT: store volatile bfloat %[[#A]], ptr @h0, align 2

  h0 = (__bf16)-2.0f;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.bf16, !cir.ptr<!cir.bf16>

  // NONATIVE-LLVM: store volatile bfloat 0xRC000, ptr @h0, align 2

  // NATIVE-LLVM: store volatile bfloat 0xRC000, ptr @h0, align 2

  h0 = f0;
  //      NONATIVE: %[[#A:]] = cir.get_global @f0 : !cir.ptr<!cir.float>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.float>, !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.get_global @f0 : !cir.ptr<!cir.float>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.float>, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = load volatile float, ptr @f0, align 4
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fptrunc float %[[#A]] to bfloat
  // NONATIVE-LLVM-NEXT: store volatile bfloat %[[#B]], ptr @h0, align 2

  //      NATIVE-LLVM: %[[#A:]] = load volatile float, ptr @f0, align 4
  // NATIVE-LLVM-NEXT: %[[#B:]] = fptrunc float %[[#A]] to bfloat
  // NATIVE-LLVM-NEXT: store volatile bfloat %[[#B]], ptr @h0, align 2

  h0 = i0;
  //      NONATIVE: %[[#A:]] = cir.get_global @i0 : !cir.ptr<!s32i>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!s32i>, !s32i
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(int_to_float, %[[#B]] : !s32i), !cir.bf16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.get_global @i0 : !cir.ptr<!s32i>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!s32i>, !s32i
  // NATIVE-NEXT: %[[#C:]] = cir.cast(int_to_float, %[[#B]] : !s32i), !cir.bf16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = load volatile i32, ptr @i0, align 4
  // NONATIVE-LLVM-NEXT: %[[#B:]] = sitofp i32 %[[#A]] to bfloat
  // NONATIVE-LLVM-NEXT: store volatile bfloat %[[#B]], ptr @h0, align 2

  //      NATIVE-LLVM: %[[#A:]] = load volatile i32, ptr @i0, align 4
  // NATIVE-LLVM-NEXT: %[[#B:]] = sitofp i32 %[[#A]] to bfloat
  // NATIVE-LLVM-NEXT: store volatile bfloat %[[#B]], ptr @h0, align 2

  i0 = h0;
  //      NONATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.bf16>, !cir.bf16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.bf16), !s32i
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @i0 : !cir.ptr<!s32i>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.bf16>, !cir.bf16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.bf16), !s32i
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @i0 : !cir.ptr<!s32i>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !s32i, !cir.ptr<!s32i>

  //      NONATIVE-LLVM: %[[#A:]] = load volatile bfloat, ptr @h0, align 2
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fptosi bfloat %[[#A]] to i32
  // NONATIVE-LLVM-NEXT: store volatile i32 %[[#B]], ptr @i0, align 4

  //      NATIVE-LLVM: %[[#A:]] = load volatile bfloat, ptr @h0, align 2
  // NATIVE-LLVM-NEXT: %[[#B:]] = fptosi bfloat %[[#A]] to i32
  // NATIVE-LLVM-NEXT: store volatile i32 %[[#B]], ptr @i0, align 4

  h0 += h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(add, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.binop(add, %{{.+}}, %{{.+}}) : !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#A:]] = fadd float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#A]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fadd bfloat %{{.+}}, %{{.+}}

  h0 += (__bf16)1.0f;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.bf16), !cir.float
  //      NONATIVE: %[[#D:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.binop(add, %[[#D]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.cast(floating, %[[#E]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#F]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.float), !cir.bf16
  //      NATIVE: %[[#C:]] = cir.binop(add, %{{.+}}, %[[#B]]) : !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fadd float %[[#A]], 1.000000e+00
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fadd bfloat %{{.+}}, 0xR3F80

  h0 += f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fadd float %[[#A]], %{{.+}}
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  //      NATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#B:]] = fadd float %[[#A]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  i0 += h0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(add, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(float_to_int, %[[#C]] : !cir.float), !s32i
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NATIVE-NEXT: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.bf16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.bf16), !s32i
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fadd float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptosi float %[[#RES]] to i32

  //      NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM-NEXT: %[[#A:]] = fadd bfloat %[[#LHS]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptosi bfloat %[[#A]] to i32

  h0 += i0;
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.bf16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(add, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NATIVE: %[[#B:]] = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM-NEXT: %[[#RHS:]] = fpext bfloat %[[#A]] to float
  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fadd float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  // NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM: %{{.+}} = fadd bfloat %{{.+}}, %[[#RHS]]

  h0 -= h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(sub, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#A:]] = fsub float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#A]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fsub bfloat %{{.+}}, %{{.+}}

  h0 -= (__bf16)1.0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.double), !cir.bf16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.bf16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(sub, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.bf16
  //      NATIVE: %[[#C:]] = cir.binop(sub, %{{.+}}, %[[#B]]) : !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fsub float %[[#A]], 1.000000e+00
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fsub bfloat %{{.+}}, 0xR3F80

  h0 -= f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fsub float %[[#A]], %{{.+}}
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  //      NATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#B:]] = fsub float %[[#A]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  i0 -= h0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(sub, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(float_to_int, %[[#C]] : !cir.float), !s32i
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NATIVE-NEXT: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.bf16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.bf16), !s32i
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptosi float %[[#RES]] to i32

  //      NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM-NEXT: %[[#A:]] = fsub bfloat %[[#LHS]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptosi bfloat %[[#A]] to i32

  h0 -= i0;
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.bf16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(sub, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NATIVE: %[[#B:]] = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM-NEXT: %[[#RHS:]] = fpext bfloat %[[#A]] to float
  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fsub float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  // NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM: %{{.+}} = fsub bfloat %{{.+}}, %[[#RHS]]

  h0 *= h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(mul, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.binop(mul, %{{.+}}, %{{.+}}) : !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#A:]] = fmul float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#A]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fmul bfloat %{{.+}}, %{{.+}}

  h0 *= (__bf16)1.0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.double), !cir.bf16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.bf16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(mul, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.bf16
  //      NATIVE: %[[#C:]] = cir.binop(mul, %{{.+}}, %[[#B]]) : !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fmul float %[[#A]], 1.000000e+00
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fmul bfloat %{{.+}}, 0xR3F80

  h0 *= f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fmul float %[[#A]], %{{.+}}
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  //      NATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#B:]] = fmul float %[[#A]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  i0 *= h0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(mul, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(float_to_int, %[[#C]] : !cir.float), !s32i
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NATIVE-NEXT: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.bf16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.bf16), !s32i
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fmul float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptosi float %[[#RES]] to i32

  //      NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM-NEXT: %[[#A:]] = fmul bfloat %[[#LHS]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptosi bfloat %[[#A]] to i32

  h0 *= i0;
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.bf16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(mul, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NATIVE: %[[#B:]] = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM-NEXT: %[[#RHS:]] = fpext bfloat %[[#A]] to float
  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fmul float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  // NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM: %{{.+}} = fmul bfloat %{{.+}}, %[[#RHS]]

  h0 /= h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(div, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.binop(div, %{{.+}}, %{{.+}}) : !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#A:]] = fdiv float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#A]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fdiv bfloat %{{.+}}, %{{.+}}

  h0 /= (__bf16)1.0;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.bf16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.bf16), !cir.float
  //      NONATIVE: %[[#D:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.binop(div, %[[#D]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.cast(floating, %[[#E]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#F]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.bf16
  //      NATIVE: %[[#C:]] = cir.binop(div, %{{.+}}, %[[#B]]) : !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fdiv float %[[#A]], 1.000000e+00
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  // NATIVE-LLVM: %{{.+}} = fdiv bfloat %{{.+}}, 0xR3F80

  h0 /= f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fdiv float %[[#A]], %{{.+}}
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  //      NATIVE-LLVM: %[[#A:]] = fpext bfloat %{{.+}} to float
  // NATIVE-LLVM-NEXT: %[[#B:]] = fdiv float %[[#A]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#B]] to bfloat

  i0 /= h0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(div, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(float_to_int, %[[#C]] : !cir.float), !s32i
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NATIVE-NEXT: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.bf16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.bf16), !s32i
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NONATIVE-LLVM: %[[#RHS:]] = fpext bfloat %{{.+}} to float
  //      NONATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fdiv float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptosi float %[[#RES]] to i32

  //      NATIVE-LLVM: %[[#LHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM-NEXT: %[[#A:]] = fdiv bfloat %[[#LHS]], %{{.+}}
  // NATIVE-LLVM-NEXT: %{{.+}} = fptosi bfloat %[[#A]] to i32

  h0 /= i0;
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.bf16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(div, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.bf16
  //      NATIVE: %[[#B:]] = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.bf16
  // NATIVE-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = sitofp i32 %{{.+}} to bfloat
  // NONATIVE-LLVM-NEXT: %[[#RHS:]] = fpext bfloat %[[#A]] to float
  //      NONATIVE-LLVM: %[[#LHS:]] = fpext bfloat %{{.+}} to float
  // NONATIVE-LLVM-NEXT: %[[#RES:]] = fdiv float %[[#LHS]], %[[#RHS]]
  // NONATIVE-LLVM-NEXT: %{{.+}} = fptrunc float %[[#RES]] to bfloat

  // NATIVE-LLVM: %[[#RHS:]] = sitofp i32 %{{.+}} to bfloat
  // NATIVE-LLVM: %{{.+}} = fdiv bfloat %{{.+}}, %[[#RHS]]

  h0 = d0;
  //      NONATIVE: %[[#A:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.double>, !cir.double
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.bf16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.double>, !cir.double
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.bf16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = load volatile double, ptr @d0, align 8
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fptrunc double %[[#A]] to bfloat
  // NONATIVE-LLVM-NEXT: store volatile bfloat %[[#B]], ptr @h0, align 2

  //      NATIVE-LLVM: %[[#A:]] = load volatile double, ptr @d0, align 8
  // NATIVE-LLVM-NEXT: %[[#B:]] = fptrunc double %[[#A]] to bfloat
  // NATIVE-LLVM-NEXT: store volatile bfloat %[[#B]], ptr @h0, align 2

  h0 = (float)d0;
  //      NONATIVE: %[[#A:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.double>, !cir.double
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.bf16
  // NONATIVE-NEXT: %[[#E:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %[[#E]] : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.double>, !cir.double
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.float
  // NATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.bf16
  // NATIVE-NEXT: %[[#E:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NATIVE-NEXT: cir.store volatile %[[#D]], %[[#E]] : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = load volatile double, ptr @d0, align 8
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fptrunc double %[[#A]] to float
  // NONATIVE-LLVM-NEXT: %[[#C:]] = fptrunc float %[[#B]] to bfloat
  // NONATIVE-LLVM-NEXT: store volatile bfloat %[[#C]], ptr @h0, align 2

  //      NATIVE-LLVM: %[[#A:]] = load volatile double, ptr @d0, align 8
  // NATIVE-LLVM-NEXT: %[[#B:]] = fptrunc double %[[#A]] to float
  // NATIVE-LLVM-NEXT: %[[#C:]] = fptrunc float %[[#B]] to bfloat
  // NATIVE-LLVM-NEXT: store volatile bfloat %[[#C]], ptr @h0, align 2

  d0 = h0;
  //      NONATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.bf16>, !cir.bf16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.bf16), !cir.double
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.double, !cir.ptr<!cir.double>

  //      NATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.bf16>, !cir.bf16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.bf16), !cir.double
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.double, !cir.ptr<!cir.double>

  //      NONATIVE-LLVM: %[[#A:]] = load volatile bfloat, ptr @h0, align 2
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fpext bfloat %[[#A]] to double
  // NONATIVE-LLVM-NEXT: store volatile double %[[#B]], ptr @d0, align 8

  //      NATIVE-LLVM: %[[#A:]] = load volatile bfloat, ptr @h0, align 2
  // NATIVE-LLVM-NEXT: %[[#B:]] = fpext bfloat %[[#A]] to double
  // NATIVE-LLVM-NEXT: store volatile double %[[#B]], ptr @d0, align 8

  d0 = (float)h0;
  //      NONATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.bf16>, !cir.bf16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.bf16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.double
  // NONATIVE-NEXT: %[[#E:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %[[#E]] : !cir.double, !cir.ptr<!cir.double>

  //      NATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.bf16>, !cir.bf16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.bf16), !cir.float
  // NATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.double
  // NATIVE-NEXT: %[[#E:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NATIVE-NEXT: cir.store volatile %[[#D]], %[[#E]] : !cir.double, !cir.ptr<!cir.double>

  //      NONATIVE-LLVM: %[[#A:]] = load volatile bfloat, ptr @h0, align 2
  // NONATIVE-LLVM-NEXT: %[[#B:]] = fpext bfloat %[[#A]] to float
  // NONATIVE-LLVM-NEXT: %[[#C:]] = fpext float %[[#B]] to double
  // NONATIVE-LLVM-NEXT: store volatile double %[[#C]], ptr @d0, align 8

  //      NATIVE-LLVM: %[[#A:]] = load volatile bfloat, ptr @h0, align 2
  // NATIVE-LLVM-NEXT: %[[#B:]] = fpext bfloat %[[#A]] to float
  // NATIVE-LLVM-NEXT: %[[#C:]] = fpext float %[[#B]] to double
  // NATIVE-LLVM-NEXT: store volatile double %[[#C]], ptr @d0, align 8

  h0 = s0;
  //      NONATIVE: %[[#A:]] = cir.get_global @s0 : !cir.ptr<!s16i>
  // NONATIVE-NEXT: %[[#B:]] = cir.load %[[#A]] : !cir.ptr<!s16i>, !s16i
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(int_to_float, %[[#B]] : !s16i), !cir.bf16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NATIVE: %[[#A:]] = cir.get_global @s0 : !cir.ptr<!s16i>
  // NATIVE-NEXT: %[[#B:]] = cir.load %[[#A]] : !cir.ptr<!s16i>, !s16i
  // NATIVE-NEXT: %[[#C:]] = cir.cast(int_to_float, %[[#B]] : !s16i), !cir.bf16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.bf16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.bf16, !cir.ptr<!cir.bf16>

  //      NONATIVE-LLVM: %[[#A:]] = load i16, ptr @s0, align 2
  // NONATIVE-LLVM-NEXT: %[[#B:]] = sitofp i16 %[[#A]] to bfloat
  // NONATIVE-LLVM-NEXT: store volatile bfloat %[[#B]], ptr @h0, align 2

  //      NATIVE-LLVM: %[[#A:]] = load i16, ptr @s0, align 2
  // NATIVE-LLVM-NEXT: %[[#B:]] = sitofp i16 %[[#A]] to bfloat
  // NATIVE-LLVM-NEXT: store volatile bfloat %[[#B]], ptr @h0, align 2
}
