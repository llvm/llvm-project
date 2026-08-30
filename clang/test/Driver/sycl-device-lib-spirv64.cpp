// Tests that the SYCL toolchain links libclang_rt.builtins.bc for spirv64
// via -mlink-builtin-bitcode, that --no-offloadlib suppresses it, and that
// -nolibsycl does not suppress it (builtins are target infrastructure, not
// SYCL runtime).

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   -resource-dir=%S/Inputs/spirv64-sycl %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-BUILTIN-BC
// CHECK-BUILTIN-BC: "-triple" "spirv64-unknown-unknown"
// CHECK-BUILTIN-BC: "-mlink-builtin-bitcode" "{{.*}}lib{{[/\\]+}}spirv64-unknown-unknown{{[/\\]+}}libclang_rt.builtins.bc"

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl --no-offloadlib \
// RUN:   -resource-dir=%S/Inputs/spirv64-sycl %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-OFFLOADLIB
// CHECK-NO-OFFLOADLIB: "-cc1" "-triple" "spirv64-unknown-unknown"
// CHECK-NO-OFFLOADLIB-NOT: "-mlink-builtin-bitcode"

// -nolibsycl suppresses the SYCL runtime library but NOT compiler-rt builtins.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -nolibsycl \
// RUN:   -resource-dir=%S/Inputs/spirv64-sycl %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOLIBSYCL
// CHECK-NOLIBSYCL: "-triple" "spirv64-unknown-unknown"
// CHECK-NOLIBSYCL: "-mlink-builtin-bitcode" "{{.*}}lib{{[/\\]+}}spirv64-unknown-unknown{{[/\\]+}}libclang_rt.builtins.bc"

// Test fallback to LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF layout:
// lib/<os>/libclang_rt.builtins-spirv64.bc
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   -resource-dir=%S/Inputs/spirv64-sycl-legacy %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-LEGACY-BC
// CHECK-LEGACY-BC: "-triple" "spirv64-unknown-unknown"
// CHECK-LEGACY-BC: "-mlink-builtin-bitcode" "{{.*}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.builtins-spirv64.bc"

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   -resource-dir=%T/nonexistent %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MISSING
// CHECK-MISSING: error: no compiler-rt builtins bitcode library '{{.*}}libclang_rt.builtins{{.*}}.bc' found in the clang resource directory
