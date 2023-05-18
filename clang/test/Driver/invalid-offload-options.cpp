// REQUIRES: x86-registered-target
// UNSUPPORTED: system-windows

// RUN: not %clang -### -x hip --target=x86_64-linux-gnu --offload= \
// RUN:   --hip-path=%S/Inputs/hipspv -nogpuinc -nogpulib %s \
// RUN: 2>&1 | FileCheck --check-prefix=INVALID-TARGET %s
// RUN: not %clang -### -x hip --target=x86_64-linux-gnu --offload=foo \
// RUN:   --hip-path=%S/Inputs/hipspv -nogpuinc -nogpulib %s \
// RUN: 2>&1 | FileCheck --check-prefix=INVALID-TARGET %s

// INVALID-TARGET: error: invalid or unsupported offload target: '{{.*}}'

// In the future we should be able to specify multiple targets for HIP
// compilation but currently it is not supported.
//
// RUN: not %clang -### -x hip --target=x86_64-linux-gnu --offload=foo,bar \
// RUN:   --hip-path=%S/Inputs/hipspv -nogpuinc -nogpulib %s \
// RUN: 2>&1 | FileCheck --check-prefix=TOO-MANY-TARGETS %s
// RUN: not %clang -### -x hip --target=x86_64-linux-gnu \
// RUN:   --offload=foo --offload=bar \
// RUN:   --hip-path=%S/Inputs/hipspv -nogpuinc -nogpulib %s \
// RUN: 2>&1 | FileCheck --check-prefix=TOO-MANY-TARGETS %s

// TOO-MANY-TARGETS: error: only one offload target is supported

// RUN: not %clang -### -x hip --target=x86_64-linux-gnu -nogpuinc -nogpulib \
// RUN:   --offload=amdgcn-amd-amdhsa --offload-arch=gfx900 %s \
// RUN: 2>&1 | FileCheck --check-prefix=OFFLOAD-ARCH-MIX %s

// OFFLOAD-ARCH-MIX: error: option '--offload-arch' cannot be specified with '--offload'
