// REQUIRES: amdgpu-registered-target

// Test that by default, AMDGPU functions only emit delta target-features
// (features that differ from the target CPU's defaults). This reduces IR bloat.

// Default behavior for gfx90a: test_default has no target-features,
// test_explicit_attr has only the delta (+gfx11-insts).
// RUN: %clang_cc1 -triple amdgpu9.0a -emit-llvm -o - %s | FileCheck --check-prefix=GFX90A %s

// With -target-feature, both functions get the delta feature.
// gfx1030 defaults to wavefrontsize32, so +wavefrontsize64 is a delta.
// RUN: %clang_cc1 -triple amdgpu10.30 -target-feature +wavefrontsize64 -emit-llvm -o - %s | FileCheck --check-prefix=CMDLINE %s

// GFX90A-LABEL: define {{.*}} @test_default()
// GFX90A-SAME: #[[ATTR_DEFAULT:[0-9]+]]
// GFX90A-LABEL: define {{.*}} @test_explicit_attr()
// GFX90A-SAME: #[[ATTR_EXPLICIT:[0-9]+]]
//
// The processor is encoded in the triple's subarch, so there is no
// target-cpu attribute. test_default has NO target-features.
// GFX90A: attributes #[[ATTR_DEFAULT]] = {
// GFX90A-NOT: "target-cpu"
// GFX90A-NOT: "target-features"
// GFX90A-SAME: }
//
// test_explicit_attr should have ONLY the delta target-features
// GFX90A: attributes #[[ATTR_EXPLICIT]] = {
// GFX90A-NOT: "target-cpu"
// GFX90A-SAME: "target-features"="+gfx11-insts"
// GFX90A-SAME: }

// With -target-feature +wavefrontsize64, test_default gets just that delta,
// test_explicit_attr gets both +gfx11-insts and +wavefrontsize64.
// CMDLINE-LABEL: define {{.*}} @test_default()
// CMDLINE-SAME: #[[ATTR_DEFAULT:[0-9]+]]
// CMDLINE-LABEL: define {{.*}} @test_explicit_attr()
// CMDLINE-SAME: #[[ATTR_EXPLICIT:[0-9]+]]
//
// CMDLINE: attributes #[[ATTR_DEFAULT]] = {
// CMDLINE-NOT: "target-cpu"
// CMDLINE-SAME: "target-features"="+wavefrontsize64"
// CMDLINE-SAME: }
//
// CMDLINE: attributes #[[ATTR_EXPLICIT]] = {
// CMDLINE-NOT: "target-cpu"
// CMDLINE-SAME: "target-features"="{{[^"]*}}+gfx11-insts{{[^"]*}}+wavefrontsize64{{[^"]*}}"
// CMDLINE-SAME: }

kernel void test_default() {}

__attribute__((target("gfx11-insts")))
void test_explicit_attr() {}
