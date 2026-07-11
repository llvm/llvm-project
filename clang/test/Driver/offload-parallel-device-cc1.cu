// REQUIRES: x86-registered-target, amdgpu-registered-target
// REQUIRES: nvptx-registered-target, lld

// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu \
// RUN:   -nostdinc -nogpuinc -nohipwrapperinc -nogpulib \
// RUN:   --offload-arch=gfx900 --offload-arch=gfx906 --offload-jobs=2 \
// RUN:   -O3 -c %s -o %t.o

// RUN: cd %t && %clang -x cuda --target=x86_64-unknown-linux-gnu \
// RUN:   -nocudainc -nocudalib \
// RUN:   --cuda-gpu-arch=sm_70 --cuda-gpu-arch=sm_80 --offload-jobs=2 \
// RUN:   --cuda-device-only -S -Werror %s

// RUN: not %clang -x cuda --target=x86_64-unknown-linux-gnu \
// RUN:   -nocudainc -nocudalib \
// RUN:   --cuda-gpu-arch=sm_70 --cuda-gpu-arch=sm_80 --offload-jobs=0x4 \
// RUN:   --cuda-device-only -S %s 2>&1 | FileCheck -check-prefix=INVJOBS %s
// INVJOBS: clang: error: invalid integral value '0x4' in '--offload-jobs=0x4'

// Empty source file. RUN lines are execution smoke tests for the driver
// path that runs independent offload device cc1 jobs through --offload-jobs.
