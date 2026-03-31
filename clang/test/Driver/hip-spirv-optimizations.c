// Verify SPIR-V compilation enables optimizations and defaults to -O3
// RUN: %clang --target=x86_64-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -### -x hip %s 2>&1 | FileCheck %s

// CHECK-NOT: -disable-llvm-passes
// CHECK-NOT: -disable-llvm-optzns
// CHECK: "-O3"

// Verify user-specified optimization level is respected
// RUN: %clang --target=x86_64-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -O2 -### -x hip %s 2>&1 \
// RUN:         | FileCheck %s --check-prefix=CHECK-O2
// CHECK-O2: "-O2"
// CHECK-O2-NOT: "-O3"

// Verify -O0 is respected
// RUN: %clang --target=x86_64-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -O0 -### -x hip %s 2>&1 \
// RUN:         | FileCheck %s --check-prefix=CHECK-O0
// CHECK-O0: "-O0"
// CHECK-O0-NOT: "-O3"

// Verify user can still disable all optimizations
// RUN: %clang --target=x86_64-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -disable-llvm-passes -### -x hip %s 2>&1 \
// RUN:         | FileCheck %s --check-prefix=CHECK-DISABLED
// CHECK-DISABLED: -disable-llvm-passes
