// The HIP-SPIR-V offload path lazily initializes the Darwin host target from
// the triple (ensureTargetInitialized) before AddDeploymentTarget() computes
// the authoritative deployment target; a disagreement between the two used to
// trip setTarget()'s "Target already initialized!" assertion.

// REQUIRES: spirv-registered-target
// UNSUPPORTED: system-windows, system-cygwin

// A Darwin *kernel* version in the triple (darwin24.3.0) is recorded as the
// macOS *product* version (macosx15) and the driver does not crash.
// RUN: %clang -### -x hip --offload=spirv64 --target=arm64-apple-darwin24.3.0 \
// RUN:   -nogpulib -nogpuinc %s 2>&1 | FileCheck %s
// CHECK: "-triple" "arm64-apple-macosx15{{[0-9.]*}}"
// CHECK-NOT: "-triple" "arm64-apple-darwin

// A deployment target from the environment overrides the lazily-recorded
// triple version instead of asserting.
// RUN: env MACOSX_DEPLOYMENT_TARGET=14.0 %clang -### -x hip \
// RUN:   --offload=spirv64 --target=arm64-apple-darwin24.3.0 \
// RUN:   -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=DEPLOY %s
// DEPLOY: "-triple" "arm64-apple-macosx14.0.0"

// Same for -mmacosx-version-min.
// RUN: %clang -### -x hip --offload=spirv64 --target=arm64-apple-darwin24.3.0 \
// RUN:   -mmacosx-version-min=14.0 -nogpulib -nogpuinc %s 2>&1 \
// RUN:   | FileCheck --check-prefix=VERMIN %s
// VERMIN: "-triple" "arm64-apple-macosx14.0.0"

// An unversioned triple must not crash either; the version is inferred (and
// on a macOS host may differ from the lazily-recorded default).
// RUN: %clang -### -x hip --offload=spirv64 --target=arm64-apple-darwin \
// RUN:   -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=NOVER %s
// NOVER: "-triple" "arm64-apple-macosx{{[0-9.]+}}"
