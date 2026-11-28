// Test that -ffp-contract is set to the right value for HLSL when combined
// with -ffast-math, -fno-fast-math, -ffp-model, -funsafe-math-optimizations,
// and -fno-unsafe-math-optimizations. HLSL relies on the frontend default
// (fast) when no option changes it.

// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=DEFAULT %s
// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FPC-ON %s
// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library -ffast-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FPC-ON %s
// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library -ffp-contract=off -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FPC-OFF %s
// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library -ffp-contract=fast -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FPC-FAST %s
// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library -fno-fast-math -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FPC-FAST %s
// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library -fno-fast-math -ffp-contract=off -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FPC-OFF %s
// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library -ffp-model=strict -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FPC-OFF %s
// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library -ffp-contract=off \
// RUN:   -funsafe-math-optimizations -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FPC-OFF %s

// DEFAULT: "-cc1"
// DEFAULT-NOT: "-ffp-contract=
// FPC-ON: "-ffp-contract=on"
// FPC-OFF: "-ffp-contract=off"
// FPC-FAST: "-ffp-contract=fast"
