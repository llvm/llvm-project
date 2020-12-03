// REQUIRES: x86-registered-target, nvptx-registered-target, amdgpu-registered-target

// By default CUDA uses -ffp-contract=fast, HIP uses -ffp-contract=fast-honor-pragmas.
// we should fuse multiply/add into fma instruction.
// In IR, fmul/fadd instructions with contract flag are emitted.
// In backend
//    nvptx -  assumes fast fp fuse option, which fuses
//             mult/add insts disregarding contract flag and
//             llvm.fmuladd intrinsics.
//    amdgcn - assumes standard fp fuse option, which only
//             fuses mult/add insts with contract flag and
//             llvm.fmuladd intrinsics.

// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -S \
// RUN:   -disable-llvm-passes -o - %s \
// RUN:   | FileCheck -check-prefixes=COMMON,NV-ON %s
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -S \
// RUN:   -target-cpu gfx906 -disable-llvm-passes -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-ON %s
// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -S \
// RUN:   -O3 -o - %s \
// RUN:   | FileCheck -check-prefixes=COMMON,NV-OPT-FAST %s
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -S \
// RUN:   -O3 -target-cpu gfx906 -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-OPT-FASTSTD %s

// Check separate compile/backend steps corresponding to -save-temps.

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -O3 -disable-llvm-passes -target-cpu gfx906 -o %t.ll -x hip %s
// RUN: cat %t.ll  | FileCheck -check-prefixes=COMMON,AMD-OPT-FAST-IR %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -S \
// RUN:   -O3 -target-cpu gfx906 -o - -x ir %t.ll \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-OPT-FASTSTD %s

// Explicit -ffp-contract=fast
// In IR, fmul/fadd instructions with contract flag are emitted.
// In backend
//    nvptx/amdgcn - assumes fast fp fuse option, which fuses
//                   mult/add insts disregarding contract flag and
//                   llvm.fmuladd intrinsics.

// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -S \
// RUN:   -ffp-contract=fast -disable-llvm-passes -o - %s \
// RUN:   | FileCheck -check-prefixes=COMMON,NV-ON %s
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -S \
// RUN:   -target-cpu gfx906 -disable-llvm-passes -o - -x hip %s \
// RUN:   -ffp-contract=fast \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-ON %s
// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -S \
// RUN:   -O3 -o - %s \
// RUN:   -ffp-contract=fast \
// RUN:   | FileCheck -check-prefixes=COMMON,NV-OPT-FAST %s
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -S \
// RUN:   -O3 -target-cpu gfx906 -o - -x hip %s \
// RUN:   -ffp-contract=fast \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-OPT-FAST %s

// Check separate compile/backend steps corresponding to -save-temps.
// When input is IR, -ffp-contract has no effect. Backend uses default
// default FP fuse option.

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -ffp-contract=fast \
// RUN:   -O3 -disable-llvm-passes -target-cpu gfx906 -o %t.ll -x hip %s
// RUN: cat %t.ll  | FileCheck -check-prefixes=COMMON,AMD-OPT-FAST-IR %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -S \
// RUN:   -O3 -target-cpu gfx906 -o - -x ir %t.ll \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-OPT-FASTSTD %s

// Explicit -ffp-contract=fast-honor-pragmas
// In IR, fmul/fadd instructions with contract flag are emitted.
// In backend
//    nvptx/amdgcn - assumes standard fp fuse option, which only
//                   fuses mult/add insts with contract flag or
//                   llvm.fmuladd intrinsics.

// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -S \
// RUN:   -ffp-contract=fast-honor-pragmas -disable-llvm-passes -o - %s \
// RUN:   | FileCheck -check-prefixes=COMMON,NV-ON %s
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -S \
// RUN:   -target-cpu gfx906 -disable-llvm-passes -o - -x hip %s \
// RUN:   -ffp-contract=fast-honor-pragmas \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-ON %s
// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -S \
// RUN:   -O3 -o - %s \
// RUN:   -ffp-contract=fast-honor-pragmas \
// RUN:   | FileCheck -check-prefixes=COMMON,NV-OPT-FASTSTD %s
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -S \
// RUN:   -O3 -target-cpu gfx906 -o - -x hip %s \
// RUN:   -ffp-contract=fast-honor-pragmas \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-OPT-FASTSTD %s

// Check separate compile/backend steps corresponding to -save-temps.
// When input is IR, -ffp-contract has no effect. Backend uses default
// default FP fuse option.

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -ffp-contract=fast-honor-pragmas \
// RUN:   -O3 -disable-llvm-passes -target-cpu gfx906 -o %t.ll -x hip %s
// RUN: cat %t.ll  | FileCheck -check-prefixes=COMMON,AMD-OPT-FAST-IR %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -S \
// RUN:   -O3 -target-cpu gfx906 -o - -x ir %t.ll \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-OPT-FASTSTD %s

// Explicit -ffp-contract=on -- fusing by front-end.
// In IR,
//    mult/add in the same statement - llvm.fmuladd instrinsic emitted
//    mult/add in different statement -  fmul/fadd instructions without
//                                       contract flag are emitted.
// In backend
//    nvptx/amdgcn - assumes standard fp fuse option, which only
//                   fuses mult/add insts with contract flag or
//                   llvm.fmuladd intrinsics.

// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -S \
// RUN:   -ffp-contract=on -disable-llvm-passes -o - %s \
// RUN:   | FileCheck -check-prefixes=COMMON,NV-ON %s
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -S \
// RUN:   -target-cpu gfx906 -disable-llvm-passes -o - -x hip %s \
// RUN:   -ffp-contract=on \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-ON %s
// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -S \
// RUN:   -O3 -o - %s \
// RUN:   -ffp-contract=on \
// RUN:   | FileCheck -check-prefixes=COMMON,NV-OPT-ON %s
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -S \
// RUN:   -O3 -target-cpu gfx906 -o - -x hip %s \
// RUN:   -ffp-contract=on \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-OPT-ON %s

// Check separate compile/backend steps corresponding to -save-temps.

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -ffp-contract=on \
// RUN:   -O3 -disable-llvm-passes -target-cpu gfx906 -o %t.ll -x hip %s
// RUN: cat %t.ll  | FileCheck -check-prefixes=COMMON,AMD-OPT-ON-IR %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -S \
// RUN:   -O3 -target-cpu gfx906 -o - -x ir %t.ll \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-OPT-ON %s

// Explicit -ffp-contract=off should disable instruction fusing.
// In IR, fmul/fadd instructions without contract flag are emitted.
// In backend
//    nvptx/amdgcn - assumes standard fp fuse option, which only
//                   fuses mult/add insts with contract flag or
//                   llvm.fmuladd intrinsics.

// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -S \
// RUN:   -ffp-contract=off -disable-llvm-passes -o - %s \
// RUN:   | FileCheck -check-prefixes=COMMON,NV-OFF %s
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -S \
// RUN:   -target-cpu gfx906 -disable-llvm-passes -o - -x hip %s \
// RUN:   -ffp-contract=off \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-OFF %s
// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -S \
// RUN:   -O3 -o - %s \
// RUN:   -ffp-contract=off \
// RUN:   | FileCheck -check-prefixes=COMMON,NV-OPT-OFF %s
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -S \
// RUN:   -O3 -target-cpu gfx906 -o - -x hip %s \
// RUN:   -ffp-contract=off \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-OPT-OFF %s

// Check separate compile/backend steps corresponding to -save-temps.

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -ffp-contract=off \
// RUN:   -O3 -disable-llvm-passes -target-cpu gfx906 -o %t.ll -x hip %s
// RUN: cat %t.ll  | FileCheck -check-prefixes=COMMON,AMD-OPT-OFF-IR %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -S \
// RUN:   -O3 -target-cpu gfx906 -o - -x ir %t.ll \
// RUN:   | FileCheck -check-prefixes=COMMON,AMD-OPT-OFF %s

#include "Inputs/cuda.h"

// Test multiply/add in the same statement, which can be emitted as FMA when
// fp-contract is on or fast.
__host__ __device__ float func(float a, float b, float c) { return a + b * c; }
// COMMON-LABEL: _Z4funcfff
// NV-ON:       fma.rn.f32
// NV-ON-NEXT:  st.param.f32
// AMD-ON:       v_fmac_f32_e64
// AMD-ON-NEXT:  s_setpc_b64

// NV-OFF:      mul.rn.f32
// NV-OFF-NEXT: add.rn.f32
// NV-OFF-NEXT: st.param.f32
// AMD-OFF:      v_mul_f32_e64
// AMD-OFF-NEXT: v_add_f32_e64
// AMD-OFF-NEXT: s_setpc_b64

// NV-OPT-FAST: fma.rn.f32
// NV-OPT-FAST-NEXT: st.param.f32
// NV-OPT-FASTSTD: fma.rn.f32
// NV-OPT-FASTSTD-NEXT: st.param.f32
// NV-OPT-ON: fma.rn.f32
// NV-OPT-ON-NEXT: st.param.f32
// NV-OPT-OFF: mul.rn.f32
// NV-OPT-OFF-NEXT: add.rn.f32
// NV-OPT-OFF-NEXT: st.param.f32

// AMD-OPT-FAST-IR: fmul contract float
// AMD-OPT-FAST-IR: fadd contract float
// AMD-OPT-ON-IR: @llvm.fmuladd.f32
// AMD-OPT-OFF-IR: fmul float
// AMD-OPT-OFF-IR: fadd float

// AMD-OPT-FAST: v_fmac_f32_e32
// AMD-OPT-FAST-NEXT: s_setpc_b64
// AMD-OPT-FASTSTD: v_fmac_f32_e32
// AMD-OPT-FASTSTD-NEXT: s_setpc_b64
// AMD-OPT-ON: v_fmac_f32_e32
// AMD-OPT-ON-NEXT: s_setpc_b64
// AMD-OPT-OFF: v_mul_f32_e32
// AMD-OPT-OFF-NEXT: v_add_f32_e32
// AMD-OPT-OFF-NEXT: s_setpc_b64

// Test multiply/add in the different statements, which can be emitted as
// FMA when fp-contract is fast but not on.
__host__ __device__ float func2(float a, float b, float c) {
  float t = b * c;
  return t + a;
}
// COMMON-LABEL: _Z5func2fff
// NV-OPT-FAST: fma.rn.f32
// NV-OPT-FAST-NEXT: st.param.f32
// NV-OPT-FASTSTD: fma.rn.f32
// NV-OPT-FASTSTD-NEXT: st.param.f32
// NV-OPT-ON: mul.rn.f32
// NV-OPT-ON: add.rn.f32
// NV-OPT-ON-NEXT: st.param.f32
// NV-OPT-OFF: mul.rn.f32
// NV-OPT-OFF: add.rn.f32
// NV-OPT-OFF-NEXT: st.param.f32

// AMD-OPT-FAST-IR: fmul contract float
// AMD-OPT-FAST-IR: fadd contract float
// AMD-OPT-ON-IR: fmul float
// AMD-OPT-ON-IR: fadd float
// AMD-OPT-OFF-IR: fmul float
// AMD-OPT-OFF-IR: fadd float

// AMD-OPT-FAST: v_fmac_f32_e32
// AMD-OPT-FAST-NEXT: s_setpc_b64
// AMD-OPT-FASTSTD: v_fmac_f32_e32
// AMD-OPT-FASTSTD-NEXT: s_setpc_b64
// AMD-OPT-ON: v_mul_f32_e32
// AMD-OPT-ON-NEXT: v_add_f32_e32
// AMD-OPT-ON-NEXT: s_setpc_b64
// AMD-OPT-OFF: v_mul_f32_e32
// AMD-OPT-OFF-NEXT: v_add_f32_e32
// AMD-OPT-OFF-NEXT: s_setpc_b64

// Test multiply/add in the different statements, which is forced
// to be compiled with fp contract on. fmul/fadd without contract
// flags are emitted in IR. In nvptx, they are emitted as FMA in
// fp-contract is fast but not on, as nvptx backend uses the same
// fp fuse option as front end, whereas fast fp fuse option in
// backend fuses fadd/fmul disregarding contract flag. In amdgcn
// they are not fused as amdgcn always use standard fp fusion
// option which respects contract flag.
  __host__ __device__ float func3(float a, float b, float c) {
#pragma clang fp contract(on)
  float t = b * c;
  return t + a;
}
// COMMON-LABEL: _Z5func3fff
// NV-OPT-FAST: fma.rn.f32
// NV-OPT-FAST-NEXT: st.param.f32
// NV-OPT-FASTSTD: mul.rn.f32
// NV-OPT-FASTSTD: add.rn.f32
// NV-OPT-FASTSTD-NEXT: st.param.f32
// NV-OPT-ON: mul.rn.f32
// NV-OPT-ON: add.rn.f32
// NV-OPT-ON-NEXT: st.param.f32
// NV-OPT-OFF: mul.rn.f32
// NV-OPT-OFF: add.rn.f32
// NV-OPT-OFF-NEXT: st.param.f32

// AMD-OPT-FAST-IR: fmul float
// AMD-OPT-FAST-IR: fadd float
// AMD-OPT-ON-IR: fmul float
// AMD-OPT-ON-IR: fadd float
// AMD-OPT-OFF-IR: fmul float
// AMD-OPT-OFF-IR: fadd float

// AMD-OPT-FAST: v_fmac_f32_e32
// AMD-OPT-FAST-NEXT: s_setpc_b64
// AMD-OPT-FASTSTD: v_mul_f32_e32
// AMD-OPT-FASTSTD-NEXT: v_add_f32_e32
// AMD-OPT-FASTSTD-NEXT: s_setpc_b64
// AMD-OPT-ON: v_mul_f32_e32
// AMD-OPT-ON-NEXT: v_add_f32_e32
// AMD-OPT-ON-NEXT: s_setpc_b64
// AMD-OPT-OFF: v_mul_f32_e32
// AMD-OPT-OFF-NEXT: v_add_f32_e32
// AMD-OPT-OFF-NEXT: s_setpc_b64
