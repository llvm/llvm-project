///
/// Perform several driver tests for OpenMP offloading
///

// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:          -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 \
// RUN:          --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-nvptx-test.bc %s 2>&1 \
// RUN:   | FileCheck %s
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:          --offload-arch=sm_52 \
// RUN:          --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-nvptx-test.bc %s 2>&1 \
// RUN:   | FileCheck %s

// verify the tools invocations
// CHECK: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-emit-llvm-bc"{{.*}}"-x" "c"
// CHECK: "-cc1" "-triple" "nvptx64-nvidia-cuda" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}}"-target-cpu" "sm_52"
// CHECK: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-emit-obj"
// CHECK: clang-linker-wrapper{{.*}}"--"{{.*}} "-o" "a.out"

// RUN:   %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PHASES %s
// CHECK-PHASES: 0: input, "[[INPUT:.+]]", c, (host-openmp)
// CHECK-PHASES: 1: preprocessor, {0}, cpp-output, (host-openmp)
// CHECK-PHASES: 2: compiler, {1}, ir, (host-openmp)
// CHECK-PHASES: 3: input, "[[INPUT]]", c, (device-openmp)
// CHECK-PHASES: 4: preprocessor, {3}, cpp-output, (device-openmp)
// CHECK-PHASES: 5: compiler, {4}, ir, (device-openmp)
// CHECK-PHASES: 6: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (nvptx64-nvidia-cuda)" {5}, ir
// CHECK-PHASES: 7: backend, {6}, assembler, (device-openmp)
// CHECK-PHASES: 8: assembler, {7}, object, (device-openmp)
// CHECK-PHASES: 9: offload, "device-openmp (nvptx64-nvidia-cuda)" {8}, object
// CHECK-PHASES: 10: clang-offload-packager, {9}, image
// CHECK-PHASES: 11: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, " (x86_64-unknown-linux-gnu)" {10}, ir
// CHECK-PHASES: 12: backend, {11}, assembler, (host-openmp)
// CHECK-PHASES: 13: assembler, {12}, object, (host-openmp)
// CHECK-PHASES: 14: clang-linker-wrapper, {13}, image, (host-openmp)

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HOST_BC:.+]]"
// CHECK-BINDINGS: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_BC:.+]]"
// CHECK-BINDINGS: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[DEVICE_BC]]"], output: "[[DEVICE_OBJ:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[DEVICE_OBJ]]"], output: "[[BINARY:.+.out]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[BINARY]]"], output: "[[HOST_OBJ:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 -nogpulib -save-temps %s 2>&1 | FileCheck %s --check-prefix=CHECK-TEMP-BINDINGS
// CHECK-TEMP-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[DEVICE_OBJ:.+]]"], output: "[[BINARY:.+.out]]"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_52 --offload-arch=sm_70 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-ARCH-BINDINGS
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_52,sm_70 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-ARCH-BINDINGS
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_52,sm_70,sm_35,sm_80 --no-offload-arch=sm_35,sm_80 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-ARCH-BINDINGS
// CHECK-ARCH-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.*]]"], output: "[[HOST_BC:.*]]"
// CHECK-ARCH-BINDINGS: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_BC_SM_52:.*]]"
// CHECK-ARCH-BINDINGS: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[DEVICE_BC_SM_52]]"], output: "[[DEVICE_OBJ_SM_52:.*]]"
// CHECK-ARCH-BINDINGS: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_BC_SM_70:.*]]"
// CHECK-ARCH-BINDINGS: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[DEVICE_BC_SM_70]]"], output: "[[DEVICE_OBJ_SM_70:.*]]"
// CHECK-ARCH-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[DEVICE_OBJ_SM_52]]", "[[DEVICE_OBJ_SM_70]]"], output: "[[BINARY:.*]]"
// CHECK-ARCH-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[BINARY]]"], output: "[[HOST_OBJ:.*]]"
// CHECK-ARCH-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm_70 \
// RUN:     -fopenmp-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa --offload-arch=gfx908  \
// RUN:     -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-NVIDIA-AMDGPU

// CHECK-NVIDIA-AMDGPU: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HOST_BC:.+]]"
// CHECK-NVIDIA-AMDGPU: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[NVIDIA_PTX:.+]]"
// CHECK-NVIDIA-AMDGPU: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[NVIDIA_PTX]]"], output: "[[NVIDIA_CUBIN:.+]]"
// CHECK-NVIDIA-AMDGPU: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[AMD_BC:.+]]"
// CHECK-NVIDIA-AMDGPU: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[NVIDIA_CUBIN]]", "[[AMD_BC]]"], output: "[[BINARY:.*]]"
// CHECK-NVIDIA-AMDGPU: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[BINARY]]"], output: "[[HOST_OBJ:.+]]"
// CHECK-NVIDIA-AMDGPU: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN:   %clang -x ir -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp --offload-arch=sm_52 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-IR

// CHECK-IR: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT_IR:.+]]"], output: "[[OBJECT:.+]]"
// CHECK-IR: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[OBJECT]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -emit-llvm -S -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-EMIT-LLVM-IR
// CHECK-EMIT-LLVM-IR: "-cc1"{{.*}}"-triple" "nvptx64-nvidia-cuda"{{.*}}"-emit-llvm"

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvida-cuda -march=sm_70 \
// RUN:          --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-new-nvptx-test.bc \
// RUN:          -nogpulib %s -o openmp-offload-gpu 2>&1 \
// RUN:   | FileCheck -check-prefix=DRIVER_EMBEDDING %s

// DRIVER_EMBEDDING: -fembed-offload-object={{.*}}.out

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:     --offload-host-only -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-HOST-ONLY
// CHECK-HOST-ONLY: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.*]]"], output: "[[OUTPUT:.*]]"
// CHECK-HOST-ONLY: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[OUTPUT]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:     --offload-device-only -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEVICE-ONLY
// CHECK-DEVICE-ONLY: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.*]]"], output: "[[HOST_BC:.*]]"
// CHECK-DEVICE-ONLY: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_ASM:.*]]"
// CHECK-DEVICE-ONLY: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[DEVICE_ASM]]"], output: "{{.*}}-openmp-nvptx64-nvidia-cuda.o"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:     --offload-device-only -E -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEVICE-ONLY-PP
// CHECK-DEVICE-ONLY-PP: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT:.*]]"], output: "-"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=sm_52 \
// RUN:     -foffload-lto %s 2>&1 | FileCheck --check-prefix=CHECK-LTO-LIBRARY %s

// CHECK-LTO-LIBRARY: {{.*}}-lomptarget{{.*}}-lomptarget.devicertl

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=sm_52 -nogpulib \
// RUN:     -foffload-lto %s 2>&1 | FileCheck --check-prefix=CHECK-NO-LIBRARY %s

// CHECK-NO-LIBRARY-NOT: {{.*}}-lomptarget{{.*}}-lomptarget.devicertl

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=sm_52 -nogpulib \
// RUN:     -Xoffload-linker a -Xoffload-linker-nvptx64-nvidia-cuda b -Xoffload-linker-nvptx64 c \
// RUN:     %s 2>&1 | FileCheck --check-prefix=CHECK-XLINKER %s

// CHECK-XLINKER: -device-linker=a{{.*}}-device-linker=nvptx64-nvidia-cuda=b{{.*}}-device-linker=nvptx64-nvidia-cuda=c{{.*}}--

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=sm_52 -nogpulib \
// RUN:     -foffload-lto %s 2>&1 | FileCheck --check-prefix=CHECK-LTO-FEATURES %s

// CHECK-LTO-FEATURES: clang-offload-packager{{.*}}--image={{.*}}feature=+ptx{{[0-9]+}}

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=sm_52 -nogpulib \
// RUN:     -Xopenmp-target=nvptx64-nvidia-cuda --cuda-feature=+ptx64 -foffload-lto %s 2>&1 \
// RUN:    | FileCheck --check-prefix=CHECK-SET-FEATURES %s

// CHECK-SET-FEATURES: clang-offload-packager{{.*}}--image={{.*}}feature=+ptx64
