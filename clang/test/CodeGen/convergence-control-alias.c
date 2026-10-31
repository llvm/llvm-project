// RUN: %clang_cc1 -triple spirv1.6-unknown-vulkan1.3-compute -fconvergent-functions -emit-llvm -o - %s | FileCheck --check-prefix=SPIRV %s
// RUN: %clang_cc1 -triple amdgpu9-amd-amdhsa -fconvergent-functions -emit-llvm -o - %s | FileCheck --check-prefix=AMDGPU %s

// A call through an alias or a function pointer is only convergent because of
// its call site attribute, which must match the convergence control token.
// With convergence tokens, [[clang::noconvergent]] does not change whether a
// call is convergent. The call site attribute is only left off when the callee
// is a convergent function when the call is emitted.

void target(void) {}
void alias_fn(void) __attribute__((alias("target")));

// SPIRV-LABEL: define spir_func void @call_alias()
// SPIRV: [[TOKEN:%.*]] = call token @llvm.experimental.convergence.entry()
// SPIRV: call spir_func void @alias_fn() [[CONVERGENT:#[0-9]+]] [ "convergencectrl"(token [[TOKEN]]) ]{{$}}
// AMDGPU-LABEL: define dso_local void @call_alias()
// AMDGPU: call void @alias_fn() [[CONVERGENT:#[0-9]+]]{{$}}
void call_alias(void) { alias_fn(); }

// SPIRV-LABEL: define spir_func void @call_alias_noconvergent()
// SPIRV: [[TOKEN:%.*]] = call token @llvm.experimental.convergence.entry()
// SPIRV: call spir_func void @alias_fn() [[CONVERGENT]] [ "convergencectrl"(token [[TOKEN]]) ]{{$}}
// AMDGPU-LABEL: define dso_local void @call_alias_noconvergent()
// AMDGPU: call void @alias_fn(){{$}}
void call_alias_noconvergent(void) {
  [[clang::noconvergent]] alias_fn();
}

// SPIRV-LABEL: define spir_func void @call_indirect_noconvergent(
// SPIRV: [[TOKEN:%.*]] = call token @llvm.experimental.convergence.entry()
// SPIRV: call spir_func void %{{[0-9]+}}() [[CONVERGENT]] [ "convergencectrl"(token [[TOKEN]]) ]{{$}}
// AMDGPU-LABEL: define dso_local void @call_indirect_noconvergent(
// AMDGPU: call void %{{[0-9]+}}(){{$}}
void call_indirect_noconvergent(void (*fp)(void)) {
  [[clang::noconvergent]] fp();
}

// SPIRV-LABEL: define spir_func void @call_direct_noconvergent()
// SPIRV: [[TOKEN:%.*]] = call token @llvm.experimental.convergence.entry()
// SPIRV: call spir_func void @target() [ "convergencectrl"(token [[TOKEN]]) ]{{$}}
// AMDGPU-LABEL: define dso_local void @call_direct_noconvergent()
// AMDGPU: call void @target(){{$}}
void call_direct_noconvergent(void) {
  [[clang::noconvergent]] target();
}

// SPIRV: attributes [[CONVERGENT]] = { convergent }
// AMDGPU: attributes [[CONVERGENT]] = { convergent }
