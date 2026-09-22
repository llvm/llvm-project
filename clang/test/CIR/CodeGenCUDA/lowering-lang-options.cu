// RUN: echo -n "GPU binary would be here." > %t.fatbin

// The lowering-relevant LangOptions are serialized onto the module as
// #cir.lowering_lang_options (see clang/test/CIR/CodeGen/lowering-lang-options.cpp),
// and post-CIRGen lowering consumes them from there via LowerModule rather than
// from a live clang::LangOptions. This test exercises both halves end-to-end:
// CIRGen records the CUDA host/device configuration in the attribute, and
// LoweringPrepare gates CUDA module-ctor synthesis on it
// (cuda && !cuda_is_device), so the registration ctor appears for the host
// compile and is suppressed for the device compile.

//===----------------------------------------------------------------------===//
// Host compilation: cuda = true, cuda_is_device = false.
//===----------------------------------------------------------------------===//

// The serialized attribute records the host configuration:
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-cir %s -x cuda \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t.fatbin -o - \
// RUN:   | FileCheck %s --check-prefix=HOST-ATTR
// HOST-ATTR: cir.lowering_lang_options = #cir.lowering_lang_options<
// HOST-ATTR-SAME: cuda = true
// HOST-ATTR-SAME: cuda_is_device = false

// -emit-cir stops before the passes, so the ctor only appears once lowering has
// consumed the attribute. cuda && !cuda_is_device holds, so it is synthesized:
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-llvm %s -x cuda \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t.fatbin -o - \
// RUN:   | FileCheck %s --check-prefix=HOST-LOWER
// HOST-LOWER: __cuda_module_ctor
// HOST-LOWER: __cudaRegisterFatBinary

//===----------------------------------------------------------------------===//
// Device compilation: cuda = true, cuda_is_device = true.
//===----------------------------------------------------------------------===//

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir -emit-cir %s -x cuda \
// RUN:   -fcuda-is-device -target-sdk-version=12.3 -o - \
// RUN:   | FileCheck %s --check-prefix=DEV-ATTR
// DEV-ATTR: cir.lowering_lang_options = #cir.lowering_lang_options<
// DEV-ATTR-SAME: cuda = true
// DEV-ATTR-SAME: cuda_is_device = true

// cuda_is_device flips the gate, so lowering must NOT synthesize the host-only
// CUDA module ctor:
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir -emit-llvm %s -x cuda \
// RUN:   -fcuda-is-device -target-sdk-version=12.3 -o - \
// RUN:   | FileCheck %s --check-prefix=DEV-LOWER
// DEV-LOWER: @_Z6kernelv
// DEV-LOWER-NOT: __cuda_module_ctor

// Minimal CUDA runtime declarations so the host-side launch stub can be built
// without the real CUDA headers.
typedef unsigned long size_t;
struct dim3 { unsigned x, y, z; };
extern "C" int cudaLaunchKernel(const void *, dim3, dim3, void **, size_t,
                                void *);
extern "C" int __cudaPopCallConfiguration(dim3 *, dim3 *, size_t *, void *);

__attribute__((global)) void kernel() {}
