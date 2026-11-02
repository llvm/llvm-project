// NOTE: Do not autogenerate. Successful compilation checks PCH consumption
// by each device target, including syntax-only jobs without output artifacts.
// REQUIRES: x86-registered-target, amdgpu-registered-target, lld
// RUN: split-file %s %t

// Build a PCH for each device target with the same frontend options as its user.
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu \
// RUN:   -nostdinc -nogpuinc -nogpulib --offload-device-only \
// RUN:   --offload-arch=gfx900 -O3 -Werror -fsyntax-only -Xclang -emit-pch \
// RUN:   %t/header.h > %t/gfx900.h.pch
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu \
// RUN:   -nostdinc -nogpuinc -nogpulib --offload-device-only \
// RUN:   --offload-arch=gfx906 -O3 -Werror -fsyntax-only -Xclang -emit-pch \
// RUN:   %t/header.h > %t/gfx906.h.pch

// Each device compile reads its own PCH while running in parallel.
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu \
// RUN:   -nostdinc -nogpuinc -nogpulib --offload-device-only \
// RUN:   --offload-arch=gfx900 --offload-arch=gfx906 --offload-jobs=2 \
// RUN:   -Xarch_gfx900 -include%t/gfx900.h -Xarch_gfx906 -include%t/gfx906.h \
// RUN:   -O3 -Werror -c %t/input.cu -o %t/hip-dev.o
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu \
// RUN:   -nostdinc -nogpuinc -nogpulib --offload-device-only \
// RUN:   --offload-arch=gfx900 --offload-arch=gfx906 --offload-jobs=2 \
// RUN:   -Xarch_gfx900 -include%t/gfx900.h -Xarch_gfx906 -include%t/gfx906.h \
// RUN:   -O3 -Werror -fsyntax-only %t/input.cu

//--- header.h
__attribute__((device)) inline int get_value() { return 42; }

//--- input.cu
__attribute__((global)) void kernel(int *p) { *p = get_value(); }
