// REQUIRES: amdgpu-registered-target
// REQUIRES: x86-registered-target

// RUN: %clang -x hip %s -fopenmp=libomp \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=gfx906 \
// RUN:   -nogpulib -nogpuinc \
// RUN:   -nostdinc -nostdlib -nodefaultlibs \
// RUN:   -I %S/../Headers/Inputs/include \
// RUN:   -fsyntax-only

#include <omp.h>

const char ompver_str[] = {'I', 'N', 'F', 'O', ':', 'O', 'p', 'e', 'n', 'M', 'P', '-',
                           'd', 'a', 't', 'e', '[',
                           ('0' + ((_OPENMP / 100000) % 10)),
                           ('0' + ((_OPENMP / 10000) % 10)),
                           ('0' + ((_OPENMP / 1000) % 10)),
                           ('0' + ((_OPENMP / 100) % 10)),
                           ('0' + ((_OPENMP / 10) % 10)),
                           ('0' + ((_OPENMP / 1) % 10)),
                           ']', '\0'};

int main(void) {
  return 0;
}
