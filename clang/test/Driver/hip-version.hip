// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// UNSUPPORTED: target={{.*}}-aix{{.*}}

// RUN: %clang -v --rocm-path=%S/Inputs/rocm 2>&1 \
// RUN:   | FileCheck -check-prefixes=FOUND %s

// RUN: env ROCM_PATH=%S/Inputs/rocm %clang -v 2>&1 \
// RUN:   | FileCheck -check-prefixes=FOUND %s

// RUN: %clang -v --rocm-path=%S/Inputs/rocm 2>&1 \
// RUN:   --target=amdgcn-amd-amdhsa \
// RUN:   | FileCheck -check-prefixes=FOUND %s

// RUN: %clang -v --hip-path=%S/Inputs/rocm 2>&1 \
// RUN:   | FileCheck -check-prefixes=FOUND %s

// RUN: env HIP_PATH=%S/Inputs/rocm %clang -v 2>&1 \
// RUN:   | FileCheck -check-prefixes=FOUND %s

// RUN: rm -rf %t/Inputs
// RUN: mkdir -p %t/Inputs
// RUN: cp -R %S/Inputs/rocm %t/Inputs
// RUN: mkdir -p %t/Inputs/rocm/share/hip
// RUN: mkdir -p %t/Inputs/rocm/hip
// RUN: mv %t/Inputs/rocm/bin/.hipVersion %t/Inputs/rocm/share/hip/version
// RUN: %clang -v --rocm-path=%t/Inputs/rocm 2>&1 \
// RUN:   | FileCheck -check-prefixes=FOUND %s
// RUN: %clang -v --hip-path=%t/Inputs/rocm 2>&1 \
// RUN:   | FileCheck -check-prefixes=FOUND %s
// RUN: %clang -v --hip-path=%t/Inputs/rocm/hip 2>&1 \
// RUN:   | FileCheck -check-prefixes=HIP-PATH %s

// FOUND: Found HIP installation: {{.*Inputs.*rocm}}, version 3.6.20214-a2917cd
// HIP-PATH: Found HIP installation: {{.*Inputs.*rocm.*hip}}, version 3.6.20214-a2917cd

// When --rocm-path is set and .hipVersion is not found, use default version

// RUN: %clang -v --rocm-path=%S 2>&1 \
// RUN:   | FileCheck -check-prefixes=DEFAULT %s

// When --hip-path is set and .hipVersion is not found, use default version

// RUN: %clang -v --hip-path=%S 2>&1 \
// RUN:   | FileCheck -check-prefixes=DEFAULT %s

// RUN: %clang -v --rocm-path=%S 2>&1 \
// RUN:   --target=amdgcn-amd-amdhsa \
// RUN:   | FileCheck -check-prefixes=DEFAULT %s

// DEFAULT: Found HIP installation: {{.*Driver}}, version 3.5.

// RUN: %clang -v --rocm-path=%S --hip-version=3.7.0 2>&1 \
// RUN:   | FileCheck -check-prefixes=SPECIFIED %s

// RUN: %clang -v --rocm-path=%S --hip-version=3.7.0 2>&1 \
// RUN:   --target=amdgcn-amd-amdhsa \
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

// Test that the HIP_RUNTIME_SANITIZER field of the HIP version file is
// parsed with -fsanitize= semantics. A missing field, or a "none" value,
// means the runtime has no sanitizer. A comma-separated list containing
// "address" means the runtime was built with the address sanitizer.

// RUN: rm -rf %t/Inputs
// RUN: mkdir -p %t/Inputs

// No HIP_RUNTIME_SANITIZER field at all.
// RUN: cp -R %S/Inputs/rocm %t/Inputs/rocm-no-field
// RUN: %clang -v --rocm-path=%t/Inputs/rocm-no-field 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOASAN %s

// HIP_RUNTIME_SANITIZER=none.
// RUN: cp -R %S/Inputs/rocm %t/Inputs/rocm-none
// RUN: printf 'HIP_RUNTIME_SANITIZER=none\n' >> %t/Inputs/rocm-none/bin/.hipVersion
// RUN: %clang -v --rocm-path=%t/Inputs/rocm-none 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOASAN %s

// HIP_RUNTIME_SANITIZER set to a sanitizer other than address.
// RUN: cp -R %S/Inputs/rocm %t/Inputs/rocm-other
// RUN: printf 'HIP_RUNTIME_SANITIZER=undefined\n' >> %t/Inputs/rocm-other/bin/.hipVersion
// RUN: %clang -v --rocm-path=%t/Inputs/rocm-other 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOASAN %s

// HIP_RUNTIME_SANITIZER=address.
// RUN: cp -R %S/Inputs/rocm %t/Inputs/rocm-address
// RUN: printf 'HIP_RUNTIME_SANITIZER=address\n' >> %t/Inputs/rocm-address/bin/.hipVersion
// RUN: %clang -v --rocm-path=%t/Inputs/rocm-address 2>&1 \
// RUN:   | FileCheck -check-prefixes=ASAN %s

// HIP_RUNTIME_SANITIZER as a comma-separated list including address.
// RUN: cp -R %S/Inputs/rocm %t/Inputs/rocm-multi
// RUN: printf 'HIP_RUNTIME_SANITIZER=undefined,address\n' >> %t/Inputs/rocm-multi/bin/.hipVersion
// RUN: %clang -v --rocm-path=%t/Inputs/rocm-multi 2>&1 \
// RUN:   | FileCheck -check-prefixes=ASAN %s

// ASAN: Found HIP installation: {{.*}}
// ASAN-NEXT: HIP runtime is built with the address sanitizer enabled

// NOASAN: Found HIP installation: {{.*}}
// NOASAN-NOT: HIP runtime is built with the address sanitizer enabled
