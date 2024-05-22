// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang -### -target x86_64-linux-gnu -nocudalib -ccc-print-bindings -fgpu-rdc \
// RUN:        --offload-new-driver --offload-arch=sm_35 --offload-arch=sm_70 %s 2>&1 \
// RUN: | FileCheck -check-prefix BINDINGS %s

//      BINDINGS: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[PTX_SM_35:.+]]"
// BINDINGS-NEXT: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[PTX_SM_35]]"], output: "[[CUBIN_SM_35:.+]]"
// BINDINGS-NEXT: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]"], output: "[[PTX_SM_70:.+]]"
// BINDINGS-NEXT: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[PTX_SM_70:.+]]"], output: "[[CUBIN_SM_70:.+]]"
// BINDINGS-NEXT: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[CUBIN_SM_35]]", "[[CUBIN_SM_70]]"], output: "[[BINARY:.+]]"
// BINDINGS-NEXT: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT]]", "[[BINARY]]"], output: "[[HOST_OBJ:.+]]"
// BINDINGS-NEXT: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN: %clang -### -target x86_64-linux-gnu -nocudalib -ccc-print-bindings -fgpu-rdc \
// RUN:        --offload-new-driver --offload-arch=sm_35 --offload-arch=sm_70 %s 2>&1 \
// RUN: | FileCheck -check-prefix BINDINGS-HOST %s

// BINDINGS-HOST: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[OUTPUT:.+]]"
// BINDINGS-HOST: # "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[OUTPUT]]"], output: "a.out"

// RUN: %clang -### -target x86_64-linux-gnu -nocudalib -ccc-print-bindings -fgpu-rdc \
// RUN:        --offload-new-driver --offload-arch=sm_35 --offload-arch=sm_70 %s 2>&1 \
// RUN: | FileCheck -check-prefix BINDINGS-DEVICE %s

// BINDINGS-DEVICE: # "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[PTX:.+]]"
// BINDINGS-DEVICE: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[PTX]]"], output: "[[CUBIN:.+]]"

// RUN: %clang -### -target x86_64-linux-gnu -nocudalib --cuda-feature=+ptx61 --offload-arch=sm_70 --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda %s 2>&1 | FileCheck -check-prefix MANUAL-FEATURE %s
// MANUAL-FEATURE: -cc1{{.*}}-target-feature{{.*}}+ptx61

// RUN: %clang -### -target x86_64-linux-gnu -nocudalib -ccc-print-bindings --offload-link %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE-LINK %s

// DEVICE-LINK: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[INPUT:.+]]"], output: "a.out"

// RUN: %clang -### -target x86_64-linux-gnu -nocudalib --offload-new-driver \
// RUN:   --offload-arch=sm_35 --offload-arch=sm_70 --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefix GPU-BINARY %s

// GPU-BINARY: fatbinary{{.*}}"--create" "{{.*}}.fatbin"
// GPU-BINARY: -cc1{{.*}}-fcuda-include-gpubinary" "{{.*}}.fatbin"
