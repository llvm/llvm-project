// RUN: rm -rf %t && mkdir -p %t && cd %t
// RUN: mkdir d e f && cp %s d/a.cpp

/// Test HIP offloading: -ftime-trace should generate traces for both host and device.
// RUN: %clang -### -ftime-trace -ftime-trace-granularity=0 -x hip d/a.cpp --offload-arch=gfx906 --offload-arch=gfx90a \
// RUN:   -nogpulib -nogpuinc -c -o e/a.o --target=x86_64-linux-gnu 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HIP
// HIP: -cc1{{.*}} "-triple" "amdgcn-amd-amdhsa"{{.*}} "-ftime-trace=e{{/|\\\\}}a-hip-amdgcn-amd-amdhsa-gfx906.json"
// HIP: -cc1{{.*}} "-triple" "amdgcn-amd-amdhsa"{{.*}} "-ftime-trace=e{{/|\\\\}}a-hip-amdgcn-amd-amdhsa-gfx90a.json"
// HIP: -cc1{{.*}} "-triple" "x86_64{{.*}}"{{.*}} "-ftime-trace=e{{/|\\\\}}a.json"

/// Test HIP offloading with new driver: same output as above.
// RUN: %clang -### -ftime-trace -ftime-trace-granularity=0 -x hip d/a.cpp --offload-arch=gfx906 --offload-arch=gfx90a \
// RUN:   -nogpulib -nogpuinc -c -o e/a.o --target=x86_64-linux-gnu --offload-new-driver 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HIP

/// Test HIP offloading with -ftime-trace=<dir>: traces go to specified directory.
// RUN: %clang -### -ftime-trace=f -ftime-trace-granularity=0 -x hip d/a.cpp --offload-arch=gfx906 \
// RUN:   -nogpulib -nogpuinc -c -o e/a.o --target=x86_64-linux-gnu 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HIP-DIR
// HIP-DIR: -cc1{{.*}} "-triple" "amdgcn-amd-amdhsa"{{.*}} "-ftime-trace=f{{/|\\\\}}a-hip-amdgcn-amd-amdhsa-gfx906.json"
// HIP-DIR: -cc1{{.*}} "-triple" "x86_64{{.*}}"{{.*}} "-ftime-trace=f{{/|\\\\}}a.json"

/// Test HIP offloading with --save-temps: both host and device get unique trace files.
// RUN: %clang -### -ftime-trace -ftime-trace-granularity=0 -x hip d/a.cpp --offload-arch=gfx906 \
// RUN:   -nogpulib -nogpuinc -c -o e/a.o --target=x86_64-linux-gnu --save-temps 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HIP-SAVE-TEMPS
// HIP-SAVE-TEMPS: -cc1{{.*}} "-triple" "amdgcn-amd-amdhsa"{{.*}} "-ftime-trace=e{{/|\\\\}}a-hip-amdgcn-amd-amdhsa-gfx906.json"
// HIP-SAVE-TEMPS: -cc1{{.*}} "-triple" "x86_64{{.*}}"{{.*}} "-ftime-trace=e{{/|\\\\}}a-host-x86_64-unknown-linux-gnu.json"

/// Test CUDA offloading: -ftime-trace should generate traces for both host and device.
// RUN: %clang -### -ftime-trace -ftime-trace-granularity=0 -x cuda d/a.cpp --offload-arch=sm_70 --offload-arch=sm_80 \
// RUN:   -c -o e/a.o --target=x86_64-linux-gnu --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CUDA
// CUDA: -cc1{{.*}} "-triple" "nvptx64-nvidia-cuda"{{.*}} "-ftime-trace=e{{/|\\\\}}a-cuda-nvptx64-nvidia-cuda-sm_70.json"
// CUDA: -cc1{{.*}} "-triple" "nvptx64-nvidia-cuda"{{.*}} "-ftime-trace=e{{/|\\\\}}a-cuda-nvptx64-nvidia-cuda-sm_80.json"
// CUDA: -cc1{{.*}} "-triple" "x86_64{{.*}}"{{.*}} "-ftime-trace=e{{/|\\\\}}a.json"
