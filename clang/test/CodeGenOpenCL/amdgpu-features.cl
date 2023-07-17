// REQUIRES: amdgpu-registered-target

// Check that appropriate features are defined for every supported AMDGPU
// "-target" and "-mcpu" options.

// RUN: %clang_cc1 -triple amdgcn -S -emit-llvm -o - %s | FileCheck --check-prefix=NOCPU %s
// RUN: %clang_cc1 -triple amdgcn -target-feature +wavefrontsize32 -S -emit-llvm -o - %s | FileCheck --check-prefix=NOCPU-WAVE32 %s
// RUN: %clang_cc1 -triple amdgcn -target-feature +wavefrontsize64 -S -emit-llvm -o - %s | FileCheck --check-prefix=NOCPU-WAVE64 %s

// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx600 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX600 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx601 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX601 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx602 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX602 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx700 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX700 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx701 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX701 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx702 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX702 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx703 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX703 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx704 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX704 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx705 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX705 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx801 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX801 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx802 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX802 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx803 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX803 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx805 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX805 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx810 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX810 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx900 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX900 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx902 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX902 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx904 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX904 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx906 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX906 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx908 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX908 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx909 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX909 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx90a -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX90A %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx90c -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX90C %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx940 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX940 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx941 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX941 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx942 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX942 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1010 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1010 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1011 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1011 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1012 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1012 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1013 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1013 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1030 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1030 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1031 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1031 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1032 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1032 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1033 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1033 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1034 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1034 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1035 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1035 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1036 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1036 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1100 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1100 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1101 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1101 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1102 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1102 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1103 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1103 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1150 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1150 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1151 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1151 %s

// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1103 -target-feature +wavefrontsize64 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1103-W64 %s

// NOCPU-NOT: "target-features"
// NOCPU-WAVE32: "target-features"="+wavefrontsize32"
// NOCPU-WAVE64: "target-features"="+wavefrontsize64"

// GFX600: "target-features"="+s-memtime-inst,+wavefrontsize64"
// GFX601: "target-features"="+s-memtime-inst,+wavefrontsize64"
// GFX602: "target-features"="+s-memtime-inst,+wavefrontsize64"
// GFX700: "target-features"="+ci-insts,+s-memtime-inst,+wavefrontsize64"
// GFX701: "target-features"="+ci-insts,+s-memtime-inst,+wavefrontsize64"
// GFX702: "target-features"="+ci-insts,+s-memtime-inst,+wavefrontsize64"
// GFX703: "target-features"="+ci-insts,+s-memtime-inst,+wavefrontsize64"
// GFX704: "target-features"="+ci-insts,+s-memtime-inst,+wavefrontsize64"
// GFX705: "target-features"="+ci-insts,+s-memtime-inst,+wavefrontsize64"
// GFX801: "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX802: "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX803: "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX805: "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX810: "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX900: "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX902: "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX904: "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX906: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX908: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX909: "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX90A: "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX90C: "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX940: "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX941: "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX942: "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX1010: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dpp,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1011: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1012: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1013: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dpp,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1030: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1031: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1032: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1033: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1034: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1035: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1036: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1100: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1101: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1102: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1103: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1150: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1151: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"

// GFX1103-W64: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot10-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize64"

kernel void test() {}
