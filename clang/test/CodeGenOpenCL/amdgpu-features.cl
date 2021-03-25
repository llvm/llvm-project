// REQUIRES: amdgpu-registered-target

// Check that appropriate features are defined for every supported AMDGPU
// "-target" and "-mcpu" options.

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
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1010 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1010 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1011 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1011 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1012 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1012 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1030 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1030 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1031 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1031 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1032 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1032 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1033 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX1033 %s

// GFX600: "target-features"="+s-memtime-inst"
// GFX601: "target-features"="+s-memtime-inst"
// GFX602: "target-features"="+s-memtime-inst"
// GFX700: "target-features"="+ci-insts,+flat-address-space,+s-memtime-inst"
// GFX701: "target-features"="+ci-insts,+flat-address-space,+s-memtime-inst"
// GFX702: "target-features"="+ci-insts,+flat-address-space,+s-memtime-inst"
// GFX703: "target-features"="+ci-insts,+flat-address-space,+s-memtime-inst"
// GFX704: "target-features"="+ci-insts,+flat-address-space,+s-memtime-inst"
// GFX705: "target-features"="+ci-insts,+flat-address-space,+s-memtime-inst"
// GFX801: "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+s-memrealtime,+s-memtime-inst"
// GFX802: "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+s-memrealtime,+s-memtime-inst"
// GFX803: "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+s-memrealtime,+s-memtime-inst"
// GFX805: "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+s-memrealtime,+s-memtime-inst"
// GFX810: "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+s-memrealtime,+s-memtime-inst"
// GFX900: "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst"
// GFX902: "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst"
// GFX904: "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst"
// GFX906: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst"
// GFX908: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst"
// GFX909: "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst"
// GFX90A: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst"
// GFX90C: "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst"
// GFX1010: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dpp,+flat-address-space,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst"
// GFX1011: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst"
// GFX1012: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst"
// GFX1030: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst"
// GFX1031: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst"
// GFX1032: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst"
// GFX1033: "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst"

kernel void test() {}
