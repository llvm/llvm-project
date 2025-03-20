// RUN: %clang -### -target x86_64-linux-gnu -foffload-via-llvm -ccc-print-bindings \
// RUN:        --offload-arch=sm_35 --offload-arch=sm_70 %s 2>&1 \
// RUN: | FileCheck -check-prefix BINDINGS %s

//      BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HOST_BC:.+]]"
// BINDINGS-NEXT: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[PTX_SM_35:.+]]"
// BINDINGS-NEXT: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[PTX_SM_35]]"], output: "[[CUBIN_SM_35:.+]]"
// BINDINGS-NEXT: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[PTX_SM_70:.+]]"
// BINDINGS-NEXT: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[PTX_SM_70:.+]]"], output: "[[CUBIN_SM_70:.+]]"
// BINDINGS-NEXT: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[CUBIN_SM_35]]", "[[CUBIN_SM_70]]"], output: "[[BINARY:.+]]"
// BINDINGS-NEXT: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[BINARY]]"], output: "[[HOST_OBJ:.+]]"
// BINDINGS-NEXT: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN: %clang -### -target x86_64-linux-gnu -foffload-via-llvm -ccc-print-bindings \
// RUN:        --offload-arch=sm_35 --offload-arch=sm_70 %s 2>&1 \
// RUN: | FileCheck -check-prefix BINDINGS-DEVICE %s

// BINDINGS-DEVICE: # "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[PTX:.+]]"
// BINDINGS-DEVICE: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[PTX]]"], output: "[[CUBIN:.+]]"

// RUN: %clang -### -target x86_64-linux-gnu -ccc-print-bindings --offload-link -foffload-via-llvm %s 2>&1 | FileCheck -check-prefix DEVICE-LINK %s

// DEVICE-LINK: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[INPUT:.+]]"], output: "a.out"
