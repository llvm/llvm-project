// RUN: %clang_cc1 -O3 -triple %itanium_abi_triple \
// RUN:   -emit-llvm -o - %s \
// RUN:   | FileCheck -check-prefixes=COMMON,CHECK %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN:   -ffp-exception-behavior=strict -O3 \
// RUN:   -triple %itanium_abi_triple -emit-llvm -o - %s \
// RUN:   | FileCheck -check-prefixes=COMMON,STRICT %s

// Is FP_CONTRACT honored in a simple case?
float fp_contract_1(float a, float b, float c) {
// COMMON: _Z13fp_contract_1fff
// CHECK: %[[M:.+]] = fmul contract float %a, %b
// CHECK-NEXT: fadd contract float %[[M]], %c
// STRICT: %[[M:.+]] = tail call contract float @llvm.experimental.constrained.fmul.f32(float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT-NEXT: tail call contract float @llvm.experimental.constrained.fadd.f32(float %[[M]], float %c, metadata !"round.tonearest", metadata !"fpexcept.strict")

#pragma clang fp contract(fast)
  return a * b + c;
}

// Is FP_CONTRACT state cleared on exiting compound statements?
float fp_contract_2(float a, float b, float c) {
  // COMMON: _Z13fp_contract_2fff
  // CHECK: %[[M:.+]] = fmul float %a, %b
  // CHECK-NEXT: fadd float %[[M]], %c
  // STRICT: %[[M:.+]] = tail call float @llvm.experimental.constrained.fmul.f32(float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // STRICT-NEXT: tail call float @llvm.experimental.constrained.fadd.f32(float %[[M]], float %c, metadata !"round.tonearest", metadata !"fpexcept.strict")
  {
#pragma clang fp contract(fast)
  }
  return a * b + c;
}

// Does FP_CONTRACT survive template instantiation?
class Foo {};
Foo operator+(Foo, Foo);

template <typename T>
T template_muladd(T a, T b, T c) {
#pragma clang fp contract(fast)
  return a * b + c;
}

float fp_contract_3(float a, float b, float c) {
  // COMMON: _Z13fp_contract_3fff
  // CHECK: %[[M:.+]] = fmul contract float %a, %b
  // CHECK-NEXT: fadd contract float %[[M]], %c
  // STRICT: %[[M:.+]] = tail call contract float @llvm.experimental.constrained.fmul.f32(float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // STRICT-NEXT: tail call contract noundef float @llvm.experimental.constrained.fadd.f32(float %[[M]], float %c, metadata !"round.tonearest", metadata !"fpexcept.strict")
  return template_muladd<float>(a, b, c);
}

template <typename T>
class fp_contract_4 {
  float method(float a, float b, float c) {
#pragma clang fp contract(fast)
    return a * b + c;
  }
};

template class fp_contract_4<int>;
// COMMON: _ZN13fp_contract_4IiE6methodEfff
// CHECK: %[[M:.+]] = fmul contract float %a, %b
// CHECK-NEXT: fadd contract float %[[M]], %c
// STRICT: %[[M:.+]] = tail call contract float @llvm.experimental.constrained.fmul.f32(float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT-NEXT: tail call contract float @llvm.experimental.constrained.fadd.f32(float %[[M]], float %c, metadata !"round.tonearest", metadata !"fpexcept.strict")

// Check file-scoped FP_CONTRACT
#pragma clang fp contract(fast)
float fp_contract_5(float a, float b, float c) {
  // COMMON: _Z13fp_contract_5fff
  // CHECK: %[[M:.+]] = fmul contract float %a, %b
  // CHECK-NEXT: fadd contract float %[[M]], %c
  // STRICT: %[[M:.+]] = tail call contract float @llvm.experimental.constrained.fmul.f32(float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // STRICT-NEXT: tail call contract float @llvm.experimental.constrained.fadd.f32(float %[[M]], float %c, metadata !"round.tonearest", metadata !"fpexcept.strict")
  return a * b + c;
}

// Verify that we can handle multiple flags on the same pragma
#pragma clang fp contract(fast) contract(off)
float fp_contract_6(float a, float b, float c) {
  // COMMON: _Z13fp_contract_6fff
  // CHECK: %[[M:.+]] = fmul float %a, %b
  // CHECK-NEXT: fadd float %[[M]], %c
  // STRICT: %[[M:.+]] = tail call float @llvm.experimental.constrained.fmul.f32(float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // STRICT-NEXT: tail call float @llvm.experimental.constrained.fadd.f32(float %[[M]], float %c, metadata !"round.tonearest", metadata !"fpexcept.strict")
  return a * b + c;
}


#pragma clang fp contract(fast)
float fp_contract_7(float a, float b, float c) {
// COMMON: _Z13fp_contract_7f
// CHECK: tail call contract float @llvm.sqrt.f32(float %a)
// CHECK: tail call contract float @llvm.ceil.f32(float %a)
// CHECK: tail call contract float @llvm.sin.f32(float %a)
// CHECK: tail call contract float @llvm.pow.f32(float %a, float %b)
// CHECK: tail call contract float @llvm.exp.f32(float %a)
// CHECK: tail call contract float @llvm.fma.f32(float %a, float %b, float %c)
// CHECK: tail call contract float @llvm.nearbyint.f32(float %a)
// CHECK: tail call contract float @llvm.log2.f32(float %a)
// STRICT: tail call contract float @llvm.experimental.constrained.sqrt.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: tail call contract float @llvm.experimental.constrained.ceil.f32(float %a, metadata !"fpexcept.strict")
// STRICT: tail call contract float @llvm.experimental.constrained.sin.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: tail call contract float @llvm.experimental.constrained.pow.f32(float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: tail call contract float @llvm.experimental.constrained.exp.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: tail call contract float @llvm.experimental.constrained.fma.f32(float %a, float %b, float %c, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: tail call contract float @llvm.experimental.constrained.nearbyint.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: tail call contract float @llvm.experimental.constrained.log2.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  return __builtin_sqrtf(a) +
         __builtin_elementwise_ceil(a) +
         __builtin_elementwise_sin(a) +
         __builtin_elementwise_pow(a, b) +
         __builtin_elementwise_exp(a) +
         __builtin_elementwise_fma(a, b, c) +
         __builtin_elementwise_nearbyint(a) +
         __builtin_elementwise_log2(a);
}

float fp_contract_8(float a, float b, float c) {
// COMMON: _Z13fp_contract_8f
// CHECK: tail call float @llvm.sqrt.f32(float %a)
// CHECK: tail call float @llvm.ceil.f32(float %a)
// CHECK: tail call float @llvm.sin.f32(float %a)
// CHECK: tail call float @llvm.pow.f32(float %a, float %b)
// CHECK: tail call float @llvm.exp.f32(float %a)
// CHECK: tail call float @llvm.fma.f32(float %a, float %b, float %c)
// CHECK: tail call float @llvm.nearbyint.f32(float %a)
// CHECK: tail call float @llvm.log2.f32(float %a)
// STRICT: tail call float @llvm.experimental.constrained.sqrt.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: tail call float @llvm.experimental.constrained.ceil.f32(float %a, metadata !"fpexcept.strict")
// STRICT: tail call float @llvm.experimental.constrained.sin.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: tail call float @llvm.experimental.constrained.pow.f32(float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: tail call float @llvm.experimental.constrained.exp.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: tail call float @llvm.experimental.constrained.fma.f32(float %a, float %b, float %c, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: tail call float @llvm.experimental.constrained.nearbyint.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: tail call float @llvm.experimental.constrained.log2.f32(float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
#pragma clang fp contract(off)  
  return __builtin_sqrtf(a) +
         __builtin_elementwise_ceil(a) +
         __builtin_elementwise_sin(a) +
         __builtin_elementwise_pow(a, b) +
         __builtin_elementwise_exp(a) +
         __builtin_elementwise_fma(a, b, c) +
         __builtin_elementwise_nearbyint(a) +
         __builtin_elementwise_log2(a);
}
