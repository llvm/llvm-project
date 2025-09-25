// REQUIRES: amdgpu-registered-target

// Check the readonly feature will can be written to the IR
// if there is no target specified.

// RUN: %clang_cc1 -triple amdgcn -emit-llvm -o - %s | FileCheck --check-prefix=NOCPU %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx942 -emit-llvm -o - %s | FileCheck --check-prefix=GFX942 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1100 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1100 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1200 -emit-llvm -o - %s | FileCheck --check-prefix=GFX1200 %s

__attribute__((target("gws,image-insts,vmem-to-lds-load-insts"))) void test() {}

// NOCPU: "target-features"="+gws,+image-insts,+vmem-to-lds-load-insts"
// GFX942: "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f64,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-conversion-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64,+xf32-insts"
// GFX1100: "target-features"="+16-bit-insts,+atomic-fadd-rtn-insts,+atomic-fmin-fmax-global-f32,+ci-insts,+dl-insts,+dot10-insts,+dot12-insts,+dot5-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32"
// GFX1200: "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32
