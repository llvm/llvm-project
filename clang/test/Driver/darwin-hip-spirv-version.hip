// Regression test: HIP-SPIR-V offloading to a Darwin host whose triple carries
// a Darwin *kernel* version (e.g. darwin24.3.0) must not crash the driver.
//
// The HIP-SPIR-V offload path calls Darwin::ensureTargetInitialized() early.
// It records the deployment version via setTarget(); the later host job calls
// setTarget() again from AddDeploymentTarget(). setTarget()'s reinit guard only
// short-circuits when the two versions match. ensureTargetInitialized() must
// therefore record the converted macOS *product* version (15.x) rather than the
// raw Darwin *kernel* version (24.x); otherwise the second call mismatches and
// trips the "Target already initialized!" assertion.

// REQUIRES: spirv-registered-target
// UNSUPPORTED: system-windows, system-cygwin

// RUN: %clang -### -x hip --offload=spirv64 --target=arm64-apple-darwin24.3.0 \
// RUN:   -nogpulib -nogpuinc %s 2>&1 | FileCheck %s

// The Darwin host -cc1 job uses the converted macOS product version, and the
// driver does not assert/crash.
// CHECK: "-triple" "arm64-apple-macosx{{[0-9.]+}}"
// CHECK-NOT: "-triple" "arm64-apple-darwin
