// HLSL enables fast math by default in the frontend. The driver must forward
// -fno-fast-math whenever the final state is not fast math, independent of
// the order in which the flags were given.

// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FAST %s
// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library \
// RUN:   -fno-signed-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FAST %s
// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library \
// RUN:   -fno-fast-math -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FAST %s
// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library \
// RUN:   -ffast-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-FAST %s
// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library \
// RUN:   -ffast-math -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-FAST %s
// RUN: %clang -### --target=dxil-pc-shadermodel6.3-library \
// RUN:   -fno-fast-math -ffast-math -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-FAST %s

// FAST: "-cc1"
// FAST-NOT: "-fno-fast-math"
// FAST: "-ffast-math"
// FAST-NOT: "-fno-fast-math"

// NO-FAST: "-cc1"
// NO-FAST-NOT: "-ffast-math"
// NO-FAST: "-fno-fast-math"
// NO-FAST-NOT: "-ffast-math"
