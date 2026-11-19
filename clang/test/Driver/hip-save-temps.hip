// Check to ensure we have a .tmp for unoptimized bitcode with -fno-gpu-rdc.
// RUN: %clang -### -ccc-print-bindings --target=x86_64-linux-gnu -nogpulib \
// RUN:   -save-temps -fno-gpu-rdc -nogpuinc --offload-arch=gfx906 -c %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=NORDC-BINDINGS %s
//      NORDC-BINDINGS: "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HIPI:.+]]"
// NORDC-BINDINGS-NEXT: "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[HIPI]]"], output: "[[TMP_BC:.+]]"
// NORDC-BINDINGS-NEXT: "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[TMP_BC]]"], output: "[[BC:.+]]"
// NORDC-BINDINGS-NEXT: "amdgpu-amd-amdhsa" - "Offload::Packager", inputs: ["[[BC]]"], output: "[[OUTPUT:.+]]"
// NORDC-BINDINGS-NEXT: "amdgpu-amd-amdhsa" - "Offload::Linker", inputs: ["[[OUTPUT]]"], output: "[[FATBIN:.+]]"

// Check to ensure we have a .tmp for unoptimized bitcode with -fgpu-rdc.
// RUN: %clang -### -ccc-print-bindings --target=x86_64-linux-gnu -nogpulib \
// RUN:   -save-temps -fgpu-rdc -nogpuinc --offload-arch=gfx906 -c %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=RDC-BINDINGS %s
//      RDC-BINDINGS: # "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HIPI:.+]]"
// RDC-BINDINGS-NEXT: # "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[HIPI]]"], output: "[[TMP_BC:.+]]"
// RDC-BINDINGS-NEXT: # "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[TMP_BC]]"], output: "[[BC:.+]]"
// RDC-BINDINGS-NEXT: # "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[BC]]"], output: "[[OUTPUT:.+]]"

// RUN: %clang -### -ccc-print-bindings --target=x86_64-linux-gnu -nogpulib \
// RUN:   -save-temps -fno-gpu-rdc -nogpuinc --offload-arch=gfx906 --offload-arch=gfx1030 -c %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=NORDC-BINDINGS-MULTIARCH %s
//
//      NORDC-BINDINGS-MULTIARCH: "amdgpu10.30-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HIPI_0:.+]]"
// NORDC-BINDINGS-MULTIARCH-NEXT: "amdgpu10.30-amd-amdhsa" - "clang", inputs: ["[[HIPI_0]]"], output: "[[TMP_BC_0:.+]]"
// NORDC-BINDINGS-MULTIARCH-NEXT: "amdgpu10.30-amd-amdhsa" - "clang", inputs: ["[[TMP_BC_0]]"], output: "[[BC_0:.+]]"
// NORDC-BINDINGS-MULTIARCH-NEXT: "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HIPI_1:.+]]"
// NORDC-BINDINGS-MULTIARCH-NEXT: "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[HIPI_1]]"], output: "[[TMP_BC_1:.+]]"
// NORDC-BINDINGS-MULTIARCH-NEXT: "amdgpu9.06-amd-amdhsa" - "clang", inputs: ["[[TMP_BC_1]]"], output: "[[BC_1:.+]]"
// NORDC-BINDINGS-MULTIARCH-NEXT: "amdgpu-amd-amdhsa" - "Offload::Packager", inputs: ["[[BC_0]]", "[[BC_1]]"], output: "[[OUTPUT:.+]]"
// NORDC-BINDINGS-MULTIARCH-NEXT: "amdgpu-amd-amdhsa" - "Offload::Linker", inputs: ["[[OUTPUT]]"], output: "[[FATBIN:.+]]"
