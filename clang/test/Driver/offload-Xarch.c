// RUN: %clang -x cuda %s -Xarch_nvptx64 -O3 -S -nogpulib -nogpuinc -### 2>&1 | FileCheck -check-prefix=O3ONCE %s
// RUN: %clang -x hip %s -Xarch_amdgcn -O3 -S -nogpulib -nogpuinc -### 2>&1 | FileCheck -check-prefix=O3ONCE %s
// RUN: %clang -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -nogpulib -nogpuinc \
// RUN:   -Xarch_amdgcn -march=gfx90a -Xarch_amdgcn -O3 -S -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=O3ONCE %s
// RUN: %clang -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -nogpulib -nogpuinc \
// RUN:   -Xarch_nvptx64 -march=sm_52 -Xarch_nvptx64 -O3 -S -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=O3ONCE %s
// O3ONCE: "-O3"
// O3ONCE-NOT: "-O3"

// RUN: %clang -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -nogpulib \
// RUN:   --target=x86_64-unknown-linux-gnu -Xarch_nvptx64 --offload-arch=sm_52,sm_60 -nogpuinc \
// RUN:   -Xarch_amdgcn --offload-arch=gfx90a,gfx1030 -ccc-print-bindings -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=OPENMP %s
//
// OPENMP: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HOST_BC:.+]]"
// OPENMP: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[GFX1030_BC:.+]]"
// OPENMP: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[GFX90A_BC:.+]]"
// OPENMP: # "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[SM52_PTX:.+]]"
// OPENMP: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[SM52_PTX]]"], output: "[[SM52_CUBIN:.+]]"
// OPENMP: # "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[SM60_PTX:.+]]"
// OPENMP: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[SM60_PTX]]"], output: "[[SM60_CUBIN:.+]]"
// OPENMP: # "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[GFX1030_BC]]", "[[GFX90A_BC]]", "[[SM52_CUBIN]]", "[[SM60_CUBIN]]"], output: "[[BINARY:.+]]"
// OPENMP: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[BINARY]]"], output: "[[HOST_OBJ:.+]]"
// OPENMP: # "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN: %clang -x cuda %s --offload-arch=sm_52,sm_60 -Xarch_sm_52 -O3 -Xarch_sm_60 -O0 \
// RUN:   -S -nogpulib -nogpuinc -### 2>&1 | FileCheck -check-prefix=CUDA %s
// CUDA: "-cc1" "-triple" "nvptx64-nvidia-cuda" {{.*}}"-target-cpu" "sm_52" {{.*}}"-O3"
// CUDA: "-cc1" "-triple" "nvptx64-nvidia-cuda" {{.*}}"-target-cpu" "sm_60" {{.*}}"-O0"
