// REQUIRES: amdgpu-registered-target

// Check that the default wavefront size is determined by the triple's subarch
// alone, with no -target-cpu. Clang only emits a wavefrontsize target-feature
// when it differs from the target default, so passing -target-feature
// +wavefrontsizeN shows up in the IR target-features attribute exactly when N is
// NOT the subarch default. gfx9 defaults to wave64; gfx10/gfx11/gfx12 default to
// wave32.

// gfx9 default is wave64: +wavefrontsize64 matches the default and is elided,
// while +wavefrontsize32 is rejected because gfx9 is wave64 only.
// RUN: %clang_cc1 -triple amdgpu9.00 -target-feature +wavefrontsize64 -emit-llvm -o - %s | FileCheck --check-prefix=NO-DELTA %s
// RUN: not %clang_cc1 -triple amdgpu9.00 -target-feature +wavefrontsize32 -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=ERR-WAVE32 %s
// The gfx9 major-family subarch behaves the same as its members.
// RUN: %clang_cc1 -triple amdgpu9 -target-feature +wavefrontsize64 -emit-llvm -o - %s | FileCheck --check-prefix=NO-DELTA %s

// gfx11 default is wave32: +wavefrontsize64 is a delta and is emitted, while
// +wavefrontsize32 matches the default and is elided.
// RUN: %clang_cc1 -triple amdgpu11.00 -target-feature +wavefrontsize64 -emit-llvm -o - %s | FileCheck --check-prefix=WAVE64-DELTA %s
// RUN: %clang_cc1 -triple amdgpu11.00 -target-feature +wavefrontsize32 -emit-llvm -o - %s | FileCheck --check-prefix=NO-DELTA %s
// RUN: %clang_cc1 -triple amdgpu11 -target-feature +wavefrontsize64 -emit-llvm -o - %s | FileCheck --check-prefix=WAVE64-DELTA %s

// gfx1250 is wave32 only: +wavefrontsize64 is rejected.
// RUN: not %clang_cc1 -triple amdgpu12.50 -target-feature +wavefrontsize64 -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=ERR-WAVE64 %s

// A amdgpu triple with no subarch and no -target-cpu has no default wave size,
// so both overrides are accepted and emitted verbatim.
// RUN: %clang_cc1 -triple amdgpu -target-feature +wavefrontsize64 -emit-llvm -o - %s | FileCheck --check-prefix=WAVE64-DELTA %s
// RUN: %clang_cc1 -triple amdgpu -target-feature +wavefrontsize32 -emit-llvm -o - %s | FileCheck --check-prefix=WAVE32-DELTA %s

kernel void foo() {}

// NO-DELTA-NOT: "target-features"
// WAVE64-DELTA: "target-features"="{{[^"]*}}+wavefrontsize64{{[^"]*}}"
// WAVE32-DELTA: "target-features"="{{[^"]*}}+wavefrontsize32{{[^"]*}}"
// ERR-WAVE32: error: option '+wavefrontsize32' cannot be specified on this target
// ERR-WAVE64: error: option '+wavefrontsize64' cannot be specified on this target
