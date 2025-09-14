// UNSUPPORTED: system-windows

// RUN: not %clang -### -x hip --target=x86_64-linux-gnu --offload=foo \
// RUN:   --hip-path=%S/Inputs/hipspv -nogpuinc -nogpulib %s \
// RUN: 2>&1 | FileCheck --check-prefix=INVALID-TARGET %s

// INVALID-TARGET: error: invalid or unsupported offload target: '{{.*}}'
