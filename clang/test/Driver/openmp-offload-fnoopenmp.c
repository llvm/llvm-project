// REQUIRES: amdgpu-registered-target, x86-registered-target

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --offload-arch=gfx906 \
// RUN:   %s 2>&1 | FileCheck -check-prefix=OFFLOAD %s
// OFFLOAD: warning: argument unused during compilation: '--offload-arch=gfx906'

// RUN: %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   --offload-arch=gfx906 \
// RUN:   -fno-openmp \
// RUN:   %s 2>&1 | FileCheck -check-prefix=OFFLOAD1 %s
// OFFLOAD1: warning: argument unused during compilation: '--offload-arch=gfx906'

// RUN: %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 \
// RUN:   -fno-openmp \
// RUN:   %s 2>&1 | FileCheck -check-prefix=LEGACY %s
// LEGACY: warning: argument unused during compilation: '-Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906'

// RUN: %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   --offload-arch=gfx906 \
// RUN:   --offload-arch=gfx908 \
// RUN:   -fno-openmp \
// RUN:   %s 2>&1 | FileCheck -check-prefix=MOFFLOAD %s
// MOFFLOAD: warning: argument unused during compilation: '--offload-arch=gfx906'
// MOFFLOAD-NEXT: warning: argument unused during compilation: '--offload-arch=gfx908'

// RUN: %clang -### -target x86_64-linux-gnu -fopenmp\
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 \
// RUN:   -fno-openmp \
// RUN:   %s 2>&1 | FileCheck -check-prefix=MLEGACY %s
// MLEGACY: warning: argument unused during compilation: '-Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906'
// MLEGACY: warning: argument unused during compilation: '-Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908'
