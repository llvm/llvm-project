// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -v --rocm-path=%S/Inputs/rocm 2>&1 \
// RUN:   | FileCheck -check-prefixes=FOUND %s

// RUN: %clang -v --rocm-path=%S/Inputs/rocm 2>&1 \
// RUN:   -target amdgcn-amd-amdhsa \
// RUN:   | FileCheck -check-prefixes=FOUND %s

// FOUND: Found HIP installation: {{.*Inputs.*rocm}}, version 3.6.20214-a2917cd

// When --rocm-path is set and .hipVersion is not found, use default version

// RUN: %clang -v --rocm-path=%S 2>&1 \
// RUN:   | FileCheck -check-prefixes=DEFAULT %s

// RUN: %clang -v --rocm-path=%S 2>&1 \
// RUN:   -target amdgcn-amd-amdhsa \
// RUN:   | FileCheck -check-prefixes=DEFAULT %s

// DEFAULT: Found HIP installation: {{.*Driver}}, version 3.5.

// RUN: %clang -v --rocm-path=%S --hip-version=3.7.0 2>&1 \
// RUN:   | FileCheck -check-prefixes=SPECIFIED %s

// RUN: %clang -v --rocm-path=%S --hip-version=3.7.0 2>&1 \
// RUN:   -target amdgcn-amd-amdhsa \
// RUN:   | FileCheck -check-prefixes=SPECIFIED %s

// SPECIFIED: Found HIP installation: {{.*Driver}}, version 3.7.0

// RUN: %clang -v --rocm-path=%S --hip-version=3.7 2>&1 \
// RUN:   | FileCheck -check-prefixes=SPECIFIED2 %s

// SPECIFIED2: Found HIP installation: {{.*Driver}}, version 3.7.0

// RUN: %clang -v --rocm-path=%S --hip-version=4.0.21025 2>&1 \
// RUN:   | FileCheck -check-prefixes=SPECIFIED3 %s

// SPECIFIED3: Found HIP installation: {{.*Driver}}, version 4.0.21025

// RUN: %clang -v --rocm-path=%S --hip-version=4 2>&1 \
// RUN:   | FileCheck -check-prefixes=SPECIFIED4 %s

// SPECIFIED4: Found HIP installation: {{.*Driver}}, version 4.0.0

// RUN: not %clang -v --rocm-path=%S --hip-version=x.y 2>&1 \
// RUN:   | FileCheck -check-prefixes=INVALID %s

// INVALID: error: invalid value 'x.y' in '--hip-version=x.y'
