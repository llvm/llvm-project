// RUN: %clang -### -target x86_64-linux-gnu -foffload-via-llvm -ccc-print-bindings \
// RUN:        --offload-arch=sm_35 --offload-arch=sm_70 %s 2>&1 \
// RUN: | FileCheck -check-prefix BINDINGS %s

// BINDINGS: "nvptx64-nvidia-cuda-llvm" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[PTX_SM_35:.+]]"
// BINDINGS-NEXT: "nvptx64-nvidia-cuda-llvm" - "NVPTX::Assembler", inputs: ["[[PTX_SM_35]]"], output: "[[CUBIN_SM_35:.+]]"
// BINDINGS-NEXT: "nvptx64-nvidia-cuda-llvm" - "clang", inputs: ["[[INPUT]]"], output: "[[PTX_SM_70:.+]]"
// BINDINGS-NEXT: "nvptx64-nvidia-cuda-llvm" - "NVPTX::Assembler", inputs: ["[[PTX_SM_70:.+]]"], output: "[[CUBIN_SM_70:.+]]"
// BINDINGS-NEXT: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[CUBIN_SM_35]]", "[[CUBIN_SM_70]]"], output: "[[BINARY:.+]]"
// BINDINGS-NEXT: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT]]", "[[BINARY]]"], output: "[[HOST_OBJ:.+]]"
// BINDINGS-NEXT: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN: %clang -### -target x86_64-linux-gnu -foffload-via-llvm -ccc-print-bindings \
// RUN:        --offload-arch=sm_35 --offload-arch=sm_70 %s 2>&1 \
// RUN: | FileCheck -check-prefix BINDINGS-DEVICE %s

// BINDINGS-DEVICE: # "nvptx64-nvidia-cuda-llvm" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[PTX:.+]]"
// BINDINGS-DEVICE: # "nvptx64-nvidia-cuda-llvm" - "NVPTX::Assembler", inputs: ["[[PTX]]"], output: "[[CUBIN:.+]]"

// RUN: %clang -### -target x86_64-linux-gnu -ccc-print-bindings --offload-link -foffload-via-llvm %s 2>&1 | FileCheck -check-prefix DEVICE-LINK %s

// DEVICE-LINK: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[INPUT:.+]]"], output: "a.out"

// RUN: %clang -### -target x86_64-linux-gnu -foffload-via-llvm \
// RUN:        -fgpu-default-stream=per-thread --offload-arch=sm_35 %s 2>&1 \
// RUN: | FileCheck -check-prefix PER-THREAD-DEFAULT-STREAM %s

// PER-THREAD-DEFAULT-STREAM: LLVMOffloadKernelPerThreadDefaultStream.o
// PER-THREAD-DEFAULT-STREAM: "-lLLVMOffloadKernel"

// RUN: %clang -### -target x86_64-linux-gnu -foffload-via-llvm \
// RUN:        -fgpu-default-stream=legacy --offload-arch=sm_35 %s 2>&1 \
// RUN: | FileCheck -check-prefix LEGACY-DEFAULT-STREAM %s

// LEGACY-DEFAULT-STREAM-NOT: LLVMOffloadKernelPerThreadDefaultStream.o
// LEGACY-DEFAULT-STREAM: "-lLLVMOffloadKernel"
