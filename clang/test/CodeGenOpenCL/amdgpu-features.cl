// REQUIRES: amdgpu-registered-target

// Check that appropriate features are defined for every supported AMDGPU
// "-target" and "-mcpu" options.

// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx600 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX600 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx601 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX601 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx700 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX700 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx801 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX801 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx904 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX904 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx906 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX906 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx908 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX908 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1010 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1010 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1011 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1011 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1012 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1012 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1030 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1030 %s

// GFX600-NOT: "target-features"
// GFX601-NOT: "target-features"
// GFX700: "target-features"="+ci-insts,+flat-address-space"
// GFX801: "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+s-memrealtime"
// GFX904: "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime"
// GFX906: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime"
// GFX908: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime"
// GFX1010: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dpp,+flat-address-space,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime"
// GFX1011: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dpp,+flat-address-space,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime"
// GFX1012: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dpp,+flat-address-space,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime"
// GFX1030: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dpp,+flat-address-space,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime"

kernel void test() {}
