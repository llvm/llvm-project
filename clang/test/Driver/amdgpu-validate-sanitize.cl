// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx900:xnack+ \
// RUN:   -fsanitize=address \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck %s

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx1250    \
// RUN:   -fsanitize=address \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck %s

// FIXME: This should error, but is silently ignored
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx900:xnack- \
// RUN:   -fsanitize=address \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=ERR %s

// CHECK: "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-mlink-bitcode-file" "{{.*}}asanrtl.bc"
// CHECK-SAME: "-fsanitize=address"

// FIXME: Should not be forwarding argument
// ERR-NOT: asanrtl.bc
// ERR: "-fsanitize=address"
