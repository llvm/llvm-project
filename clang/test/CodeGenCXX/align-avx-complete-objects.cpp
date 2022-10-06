// RUN: %clang_cc1 -no-opaque-pointers %std_cxx11-14 %s -O0 -triple=x86_64-apple-darwin -target-feature +avx2 -fmax-type-align=16 -emit-llvm -o - -Werror | FileCheck %s --check-prefixes=CHECK,PRE17
// RUN: %clang_cc1 -no-opaque-pointers %std_cxx17- %s -O0 -triple=x86_64-apple-darwin -target-feature +avx2 -fmax-type-align=16 -emit-llvm -o - -Werror | FileCheck %s --check-prefixes=CHECK,CXX17
// rdar://16254558

typedef float AVX2Float __attribute__((__vector_size__(32)));


#if __cplusplus < 202002L
volatile
#endif
float TestAlign(void)
{
       volatile AVX2Float *p = new AVX2Float;
        *p = *p;
        AVX2Float r = *p;
        return r[0];
}

// CHECK: [[R:%.*]] = alloca <8 x float>, align 32
// PRE17-NEXT:  [[CALL:%.*]] = call noalias noundef nonnull i8* @_Znwm(i64 noundef 32)
// CXX17-NEXT:  [[CALL:%.*]] = call noalias noundef nonnull align 32 i8* @_ZnwmSt11align_val_t(i64 noundef 32, i64 noundef 32)
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

#if __cplusplus < 202002L
volatile
#endif
float TestAlign2(void)
{
       volatile AVX2Float_use_existing_align *p = new AVX2Float_use_existing_align;
        *p = *p;
        AVX2Float_use_existing_align r = *p;
        return r[0];
}

// CHECK: [[R:%.*]] = alloca <8 x float>, align 32
// PRE17-NEXT:  [[CALL:%.*]] = call noalias noundef nonnull i8* @_Znwm(i64 noundef 32)
// CXX17-NEXT:  [[CALL:%.*]] = call noalias noundef nonnull align 32 i8* @_ZnwmSt11align_val_t(i64 noundef 32, i64 noundef 32)
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
