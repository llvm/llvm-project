// Tests that the SYCL toolchain links libclang_rt.builtins.bc for spirv64
// via -mlink-builtin-bitcode, and that --no-offloadlib suppresses it.

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

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   -resource-dir=%T/nonexistent %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MISSING
// CHECK-MISSING: error: no compiler-rt builtins bitcode library '{{.*}}libclang_rt.builtins.bc' found in the clang resource directory
