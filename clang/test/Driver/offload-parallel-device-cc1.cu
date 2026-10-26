// NOTE: Do not autogenerate. These checks cover job execution and selective
// driver diagnostics.
// REQUIRES: x86-registered-target, amdgpu-registered-target
// REQUIRES: nvptx-registered-target, lld

// RUN: rm -rf %t && mkdir -p %t
// RUN: not %clang -x c++ --offload-jobs=1 -Werror -fsyntax-only %s 2>&1 | \
// RUN:   FileCheck -check-prefix=NOOFFLOAD %s
// NOOFFLOAD: argument unused during compilation: '--offload-jobs=1'
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx900 --offload-jobs=1 -Werror -E %s -o %t/hip.i
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu \
// RUN:   -nostdinc -nogpuinc -nohipwrapperinc -nogpulib \
// RUN:   --offload-arch=gfx900 --offload-arch=gfx906 --offload-jobs=2 \
// RUN:   -O3 -c %s -o %t.o

// RUN: cd %t && %clang -x cuda --target=x86_64-unknown-linux-gnu \
// RUN:   -nocudainc -nocudalib \
// RUN:   --cuda-gpu-arch=sm_70 --cuda-gpu-arch=sm_80 --offload-jobs=2 \
// RUN:   --cuda-device-only -S -Werror %s

// Device-only compilation to an object parallelizes the per-target compiles
// even though each is followed by its own device link.
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu \
// RUN:   -nostdinc -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx900 --offload-arch=gfx906 --offload-jobs=2 \
// RUN:   --offload-device-only -O3 -c -Werror %s -o %t/hip-dev.o

// Saved intermediates introduce unfinished preprocessing jobs between device
// compiles. A later compile must wait for its own preprocessing job.
// RUN: mkdir -p %t/save-temps
// RUN: cd %t/save-temps && %clang -x hip --target=x86_64-unknown-linux-gnu \
// RUN:   -nostdinc -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx900 --offload-arch=gfx906 --offload-jobs=2 \
// RUN:   --offload-device-only -O3 -c -save-temps -Werror %s -o hip-dev.o
// RUN: mkdir -p %t/save-temps-obj
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu \
// RUN:   -nostdinc -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx900 --offload-arch=gfx906 --offload-jobs=2 \
// RUN:   --offload-device-only -O3 -c -save-temps=obj -Werror %s \
// RUN:   -o %t/save-temps-obj/hip-dev.o

// RUN: mkdir -p %t/cuda-save-temps
// RUN: cd %t/cuda-save-temps && %clang -x cuda --target=x86_64-unknown-linux-gnu \
// RUN:   -nocudainc -nocudalib \
// RUN:   --cuda-gpu-arch=sm_70 --cuda-gpu-arch=sm_80 --offload-jobs=2 \
// RUN:   --cuda-device-only -S -save-temps -Werror %s

// Syntax-only device compiles have no output artifacts.
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu \
// RUN:   -nostdinc -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx900 --offload-arch=gfx906 --offload-jobs=2 \
// RUN:   --offload-device-only -fsyntax-only -Werror %s
// RUN: %clang -x cuda --target=x86_64-unknown-linux-gnu \
// RUN:   -nocudainc -nocudalib \
// RUN:   --cuda-gpu-arch=sm_70 --cuda-gpu-arch=sm_80 --offload-jobs=2 \
// RUN:   --cuda-device-only -fsyntax-only -Werror %s

// RUN: not %clang -x cuda --target=x86_64-unknown-linux-gnu \
// RUN:   -nocudainc -nocudalib \
// RUN:   --cuda-gpu-arch=sm_70 --cuda-gpu-arch=sm_80 --offload-jobs=0x4 \
// RUN:   --cuda-device-only -S %s 2>&1 | FileCheck -check-prefix=INVJOBS %s
// INVJOBS: clang: error: invalid integral value '0x4' in '--offload-jobs=0x4'

// Empty source file. RUN lines test --offload-jobs with and without
// independent offload device cc1 jobs.
