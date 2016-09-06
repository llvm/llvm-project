// RUN: %clang_cc1 -x c++ %s -O0 -triple=x86_64-apple-darwin -target-feature +avx2 -fmax-type-align=16 -emit-llvm -o - -Werror | FileCheck %s
// rdar://16254558

typedef float AVX2Float __attribute__((__vector_size__(32)));


volatile float TestAlign(void)
{
       volatile AVX2Float *p = new AVX2Float;
        *p = *p;
        AVX2Float r = *p;
        return r[0];
}

// CHECK: [[R:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:  [[CALL:%.*]] = call i8* @_Znwm(i64 32)
// CHECK-NEXT:  [[ZERO:%.*]] = bitcast i8* [[CALL]] to <8 x float>*
// CHECK-NEXT:  store <8 x float>* [[ZERO]], <8 x float>** [[P:%.*]], align 8
// CHECK-NEXT:  [[ONE:%.*]] = load <8 x float>*, <8 x float>** [[P]], align 8
// CHECK-NEXT:  [[TWO:%.*]] = load volatile <8 x float>, <8 x float>* [[ONE]], align 16
// CHECK-NEXT:  [[THREE:%.*]] = load <8 x float>*, <8 x float>** [[P]], align 8
// CHECK-NEXT:  store volatile <8 x float> [[TWO]], <8 x float>* [[THREE]], align 16
// CHECK-NEXT:  [[FOUR:%.*]] = load <8 x float>*, <8 x float>** [[P]], align 8
// CHECK-NEXT:  [[FIVE:%.*]] = load volatile <8 x float>, <8 x float>* [[FOUR]], align 16
// CHECK-NEXT:  store <8 x float> [[FIVE]], <8 x float>* [[R]], align 32
// CHECK-NEXT:  [[SIX:%.*]] = load <8 x float>, <8 x float>* [[R]], align 32
// CHECK-NEXT:  [[VECEXT:%.*]] = extractelement <8 x float> [[SIX]], i32 0
// CHECK-NEXT:  ret float [[VECEXT]]

typedef float AVX2Float_Explicitly_aligned __attribute__((__vector_size__(32))) __attribute__((aligned (32)));

typedef AVX2Float_Explicitly_aligned AVX2Float_indirect;

typedef AVX2Float_indirect AVX2Float_use_existing_align;

volatile float TestAlign2(void)
{
       volatile AVX2Float_use_existing_align *p = new AVX2Float_use_existing_align;
        *p = *p;
        AVX2Float_use_existing_align r = *p;
        return r[0];
}

// CHECK: [[R:%.*]] = alloca <8 x float>, align 32
// CHECK-NEXT:  [[CALL:%.*]] = call i8* @_Znwm(i64 32)
// CHECK-NEXT:  [[ZERO:%.*]] = bitcast i8* [[CALL]] to <8 x float>*
// CHECK-NEXT:  store <8 x float>* [[ZERO]], <8 x float>** [[P:%.*]], align 8
// CHECK-NEXT:  [[ONE:%.*]] = load <8 x float>*, <8 x float>** [[P]], align 8
// CHECK-NEXT:  [[TWO:%.*]] = load volatile <8 x float>, <8 x float>* [[ONE]], align 32
// CHECK-NEXT:  [[THREE:%.*]] = load <8 x float>*, <8 x float>** [[P]], align 8
// CHECK-NEXT:  store volatile <8 x float> [[TWO]], <8 x float>* [[THREE]], align 32
// CHECK-NEXT:  [[FOUR:%.*]] = load <8 x float>*, <8 x float>** [[P]], align 8
// CHECK-NEXT:  [[FIVE:%.*]] = load volatile <8 x float>, <8 x float>* [[FOUR]], align 32
// CHECK-NEXT:  store <8 x float> [[FIVE]], <8 x float>* [[R]], align 32
// CHECK-NEXT:  [[SIX:%.*]] = load <8 x float>, <8 x float>* [[R]], align 32
// CHECK-NEXT:  [[VECEXT:%.*]] = extractelement <8 x float> [[SIX]], i32 0
// CHECK-NEXT:  ret float [[VECEXT]]
