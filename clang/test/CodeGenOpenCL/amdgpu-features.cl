// REQUIRES: amdgpu-registered-target

// Check that appropriate features are defined for every supported AMDGPU
// "-target" and "-mcpu" options.

// RUN: %clang_cc1 -triple amdgcn -emit-llvm -o - %s | FileCheck --check-prefix=NOCPU %s
// RUN: %clang_cc1 -triple amdgcn -target-feature +wavefrontsize32 -emit-llvm -o - %s | FileCheck --check-prefix=NOCPU-WAVE32 %s
// RUN: %clang_cc1 -triple amdgcn -target-feature +wavefrontsize64 -emit-llvm -o - %s | FileCheck --check-prefix=NOCPU-WAVE64 %s

// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx600 -emit-llvm -o - %s | FileCheck --check-prefix=GFX600 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx601 -emit-llvm -o - %s | FileCheck --check-prefix=GFX601 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx602 -emit-llvm -o - %s | FileCheck --check-prefix=GFX602 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx700 -emit-llvm -o - %s | FileCheck --check-prefix=GFX700 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx701 -emit-llvm -o - %s | FileCheck --check-prefix=GFX701 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx702 -emit-llvm -o - %s | FileCheck --check-prefix=GFX702 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx703 -emit-llvm -o - %s | FileCheck --check-prefix=GFX703 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx704 -emit-llvm -o - %s | FileCheck --check-prefix=GFX704 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx705 -emit-llvm -o - %s | FileCheck --check-prefix=GFX705 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx801 -emit-llvm -o - %s | FileCheck --check-prefix=GFX801 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx802 -emit-llvm -o - %s | FileCheck --check-prefix=GFX802 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx803 -emit-llvm -o - %s | FileCheck --check-prefix=GFX803 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx805 -emit-llvm -o - %s | FileCheck --check-prefix=GFX805 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx810 -emit-llvm -o - %s | FileCheck --check-prefix=GFX810 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx900 -emit-llvm -o - %s | FileCheck --check-prefix=GFX900 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx902 -emit-llvm -o - %s | FileCheck --check-prefix=GFX902 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx904 -emit-llvm -o - %s | FileCheck --check-prefix=GFX904 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx906 -emit-llvm -o - %s | FileCheck --check-prefix=GFX906 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx908 -emit-llvm -o - %s | FileCheck --check-prefix=GFX908 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx909 -emit-llvm -o - %s | FileCheck --check-prefix=GFX909 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx90a -emit-llvm -o - %s | FileCheck --check-prefix=GFX90A %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx90c -emit-llvm -o - %s | FileCheck --check-prefix=GFX90C %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx942 -emit-llvm -o - %s | FileCheck --check-prefix=GFX942 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx950 -emit-llvm -o - %s | FileCheck --check-prefix=GFX950 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1010 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1010 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1011 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1011 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1012 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1012 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1013 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1013 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1030 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1030 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1031 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1031 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1032 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1032 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1033 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1033 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1034 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1034 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1035 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1035 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1036 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1036 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1100 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1100 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1101 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1101 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1102 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1102 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1103 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1103 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1150 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1150 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1151 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1151 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1152 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1152 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1153 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1153 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1200 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1200 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1201 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1201 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1250 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1250 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1251 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1251 %s

// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1103 -target-feature +wavefrontsize64 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1103-W64 %s

// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx9-4-generic -emit-llvm -o - %s | FileCheck --check-prefix=GFX9_4_Generic %s

// NOCPU-NOT: "target-features"
// NOCPU-WAVE32: "target-features"="+wavefrontsize32"
// NOCPU-WAVE64: "target-features"="+wavefrontsize64"

// GFX600: "target-features"="+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+s-memtime-inst,+wavefrontsize64
// GFX601: "target-features"="+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+s-memtime-inst,+wavefrontsize64
// GFX602: "target-features"="+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+s-memtime-inst,+wavefrontsize64
// GFX700: "target-features"="+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+s-memtime-inst,+wavefrontsize64"
// GFX701: "target-features"="+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+s-memtime-inst,+wavefrontsize64"
// GFX702: "target-features"="+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+s-memtime-inst,+wavefrontsize64"
// GFX703: "target-features"="+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+s-memtime-inst,+wavefrontsize64"
// GFX704: "target-features"="+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+s-memtime-inst,+wavefrontsize64"
// GFX705: "target-features"="+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+s-memtime-inst,+wavefrontsize64"
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
// GFX90A: "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX90C: "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX942: "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f64,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-conversion-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64,+xf32-insts"
// GFX9_4_Generic: "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f64,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX950: "target-features"="+16-bit-insts,+ashr-pk-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f64,+atomic-global-pk-add-bf16-inst,+bf8-cvt-scale-insts,+bitop3-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot12-insts,+dot13-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+f16bf16-to-fp6bf6-cvt-scale-insts,+f32-to-f16bf16-cvt-sr-insts,+fp4-cvt-scale-insts,+fp6bf6-cvt-scale-insts,+fp8-conversion-insts,+fp8-cvt-scale-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+gfx950-insts,+mai-insts,+permlane16-swap,+permlane32-swap,+prng-inst,+s-memrealtime,+s-memtime-inst,+wavefrontsize64"
// GFX1010: "target-features"="+16-bit-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dpp,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1011: "target-features"="+16-bit-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1012: "target-features"="+16-bit-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1013: "target-features"="+16-bit-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dpp,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1030: "target-features"="+16-bit-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1031: "target-features"="+16-bit-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1032: "target-features"="+16-bit-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1033: "target-features"="+16-bit-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1034: "target-features"="+16-bit-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1035: "target-features"="+16-bit-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1036: "target-features"="+16-bit-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32"
// GFX1100: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1101: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1102: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1103: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1150: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1151: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1152: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1153: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1200: "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1201: "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1250: "target-features"="+16-bit-insts,+ashr-pk-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+atomic-global-pk-add-bf16-inst,+bf16-cvt-insts,+bf16-pk-insts,+bf16-trans-insts,+bitop3-insts,+ci-insts,+clusters,+dl-insts,+dot7-insts,+dot8-insts,+dpp,+fp8-conversion-insts,+fp8e5m3-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx1250-insts,+gfx8-insts,+gfx9-insts,+permlane16-swap,+prng-inst,+setprio-inc-wg-inst,+tanh-insts,+tensor-cvt-lut-insts,+transpose-load-f4f6-insts,+vmem-pref-insts,+wavefrontsize32"
// GFX1251: "target-features"="+16-bit-insts,+ashr-pk-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+atomic-global-pk-add-bf16-inst,+bf16-cvt-insts,+bf16-pk-insts,+bf16-trans-insts,+bitop3-insts,+ci-insts,+clusters,+dl-insts,+dot7-insts,+dot8-insts,+dpp,+fp8-conversion-insts,+fp8e5m3-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx1250-insts,+gfx8-insts,+gfx9-insts,+permlane16-swap,+prng-inst,+setprio-inc-wg-inst,+tanh-insts,+tensor-cvt-lut-insts,+transpose-load-f4f6-insts,+vmem-pref-insts,+wavefrontsize32"

// GFX1103-W64: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize64"

kernel void test() {}
