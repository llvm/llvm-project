// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -x hip %s -save-temps \
// RUN:         -use-spirv-backend -ccc-print-bindings \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHECK-SPIRV-BASE,CHECK-SPIRV

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -x hip %s -save-temps \
// RUN:         -use-spirv-backend -fgpu-rdc -ccc-print-bindings \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHECK-SPIRV-BASE,CHECK-SPIRV-RDC

// CHECK-SPIRV-BASE: # "spirv64-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HIPI:.+\.hipi]]"
// CHECK-SPIRV-BASE: # "spirv64-amd-amdhsa" - "clang", inputs: ["[[HIPI]]"], output: "[[SPV_BC:.+\.bc]]"
// CHECK-SPIRV: # "spirv64-amd-amdhsa" - "Offload::Packager", inputs: ["[[SPV_BC]]"], output: "[[HIP_OUT:.+\.out]]"
// CHECK-SPIRV: # "spirv64-amd-amdhsa" - "Offload::Linker", inputs: ["[[HIP_OUT]]"], output: "[[HIPFB:.+\.hipfb]]"
// CHECK-SPIRV-RDC: # "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[SPV_BC]]"], output: "[[HIP_OUT:.+\.out]]"
// CHECK-SPIRV-BASE: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT]]"], output: "[[HIPI:.+\.hipi]]"
// CHECK-SPIRV: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HIPI]]", "[[HIPFB]]"], output: "[[x86_BC:.+\.bc]]"
// CHECK-SPIRV-RDC: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HIPI]]", "[[HIP_OUT]]"], output: "[[x86_BC:.+\.bc]]"
// CHECK-SPIRV-BASE: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[x86_BC]]"], output: "[[x86_S:.+\.s]]"
// CHECK-SPIRV-BASE: # "x86_64-unknown-linux-gnu" - "clang::as", inputs: ["[[x86_S]]"], output: "[[x86_O:.+\.o]]"
// CHECK-SPIRV-BASE: # "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[x86_O]]"], output: "{{.+\.out}}"

// CHECK-SPIRV # "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[x86_O]]"], output: "[[x86_O:.+\.o]]"
// CHECK-SPIRV # "x86_64-unknown-linux-gnu" - "GNU::Linker", inputs: ["[[x86_O]]"], output: "{{.+\.out}}"

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -x hip %s -save-temps \
// RUN:         -use-spirv-backend --offload-device-only -ccc-print-bindings \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-OFFLOAD-DEVICE-ONLY

// CHECK-SPIRV-OFFLOAD-DEVICE-ONLY: # "spirv64-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HIPI:.+\.hipi]]"
// CHECK-SPIRV-OFFLOAD-DEVICE-ONLY: # "spirv64-amd-amdhsa" - "clang", inputs: ["[[HIPI]]"], output: "[[SPV_BC:.+\.bc]]"
// CHECK-SPIRV-OFFLOAD-DEVICE-ONLY: # "spirv64-amd-amdhsa" - "clang", inputs: ["[[SPV_BC]]"], output: "[[SPV_OUT:.+\.out]]"
// CHECK-SPIRV-OFFLOAD-DEVICE-ONLY: # "spirv64-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[SPV_OUT]]"], output: "{{.+\.hipfb}}"

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -x hip %s -save-temps \
// RUN:         -use-spirv-backend --offload-device-only -fgpu-rdc -ccc-print-bindings \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-OFFLOAD-DEVICE-ONLY-RDC

// CHECK-SPIRV-OFFLOAD-DEVICE-ONLY-RDC: # "spirv64-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HIPI:.+\.hipi]]"
// CHECK-SPIRV-OFFLOAD-DEVICE-ONLY-RDC: # "spirv64-amd-amdhsa" - "clang", inputs: ["[[HIPI]]"], output: "[[SPV_BC:.+\.bc]]"
// CHECK-SPIRV-OFFLOAD-DEVICE-ONLY-RDC: # "spirv64-amd-amdhsa" - "clang", inputs: ["[[SPV_BC]]"], output: "{{.+}}"

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -x hip %s -save-temps \
// RUN:         -use-spirv-backend --offload-device-only -S -fgpu-rdc -ccc-print-bindings \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-OFFLOAD-DEVICE-ONLY-RDC

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -x hip %s -save-temps \
// RUN:         -use-spirv-backend --offload-device-only -S -ccc-print-bindings \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-TEXTUAL-OFFLOAD-DEVICE-ONLY

// CHECK-SPIRV-TEXTUAL-OFFLOAD-DEVICE-ONLY: # "spirv64-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HIPI:.+\.hipi]]"
// CHECK-SPIRV-TEXTUAL-OFFLOAD-DEVICE-ONLY: # "spirv64-amd-amdhsa" - "clang", inputs: ["[[HIPI]]"], output: "[[SPV_BC:.+\.bc]]"
// CHECK-SPIRV-TEXTUAL-OFFLOAD-DEVICE-ONLY: # "spirv64-amd-amdhsa" - "clang", inputs: ["[[SPV_BC]]"], output: "{{.+\.s}}"
