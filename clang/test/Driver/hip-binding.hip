// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -ccc-print-bindings -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -c 2>&1 | FileCheck -check-prefix=NRDCS %s
// RUN: %clang -ccc-print-bindings -target x86_64-linux-gnu --offload-new-driver \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -c 2>&1 | FileCheck -check-prefix=NRDCS %s
// NRDCS: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[IN:.*hip-binding.hip]]"], output: "[[OBJ1:.*o]]"
// NRDCS: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[OBJ1]]"], output: "[[IMG1:.*]]"
// NRDCS: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[IN:.*hip-binding.hip]]"], output: "[[OBJ2:.*o]]"
// NRDCS: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[OBJ2]]"], output: "[[IMG2:.*]]"
// NRDCS: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[IMG1]]", "[[IMG2]]"], output: "[[FATBIN:.*]]"
// NRDCS: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[IN]]", "[[FATBIN]]"], output: "{{.*}}"

// RUN: %clang -ccc-print-bindings -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -c -fgpu-rdc 2>&1 | FileCheck -check-prefix=RDCS %s
// RDCS: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[IN:.*hip-binding.hip]]"], output: "[[BC1:.*bc]]"
// RDCS: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[IN:.*hip-binding.hip]]"], output: "[[BC2:.*bc]]"
// RDCS: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[IN]]"], output: "[[HOSTOBJ:.*o]]"
// RDCS: # "x86_64-unknown-linux-gnu" - "offload bundler", inputs: ["[[BC1]]", "[[BC2]]", "[[HOSTOBJ]]"], output: "{{.*}}"

// RUN: %clang -ccc-print-bindings -target x86_64-linux-gnu --offload-new-driver \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -c -fgpu-rdc 2>&1 | FileCheck -check-prefix=RDCS-NEW %s
// RDCS-NEW: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HIP803:.+]]"
// RDCS-NEW: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]"], output: "[[HIP900:.+]]"
// RDCS-NEW: # "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[HIP803]]", "[[HIP900]]"], output: "[[HIPFB:.+]]"
// RDCS-NEW: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT]]", "[[HIPFB]]"], output: "{{.*}}"

// RUN: touch %t.o
// RUN: %clang --hip-link -ccc-print-bindings -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 -fgpu-rdc %t.o\
// RUN: 2>&1 | FileCheck %s

// CHECK: # "x86_64-unknown-linux-gnu" - "offload bundler", inputs: ["[[IN:.*o]]"], outputs: ["[[HOSTOBJ:.*o]]", "{{.*o}}", "{{.*o}}"]
// CHECK: # "amdgcn-amd-amdhsa" - "offload bundler", inputs: ["[[IN]]"], outputs: ["{{.*o}}", "[[DOBJ1:.*o]]", "[[DOBJ2:.*o]]"]
// CHECK: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[DOBJ1]]"], output: "[[IMG1:.*out]]"
// CHECK-NOT: offload bundler
// CHECK: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[DOBJ2]]"], output: "[[IMG2:.*out]]"
// CHECK-NOT: offload bundler
// CHECK: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[IMG1]]", "[[IMG2]]"], output: "[[FATBINOBJ:.*o]]"
// CHECK-NOT: offload bundler
// CHECK: # "x86_64-unknown-linux-gnu" - "GNU::Linker", inputs: ["[[HOSTOBJ]]", "[[FATBINOBJ]]"], output: "a.out"

// RUN: %clang --hip-link -ccc-print-bindings -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %t.o\
// RUN: 2>&1 | FileCheck -check-prefix=NORDC %s

// NORDC-NOT: offload bundler
// NORDC: # "x86_64-unknown-linux-gnu" - "GNU::Linker", inputs: ["{{.*o}}"], output: "a.out"

//
// Check to make sure we can generate multiple outputs for device-only
// compilation and fail with '-o'.
//
// RUN: %clang -### -target x86_64-linux-gnu --offload-new-driver -ccc-print-bindings \
// RUN:        --offload-arch=gfx90a --offload-arch=gfx908 --offload-device-only -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=MULTI-D-ONLY %s
//      MULTI-D-ONLY: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[GFX908:.+]]"
// MULTI-D-ONLY-NEXT: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[GFX908]]"], output: "[[GFX908_OUT:.+]]"
// MULTI-D-ONLY-NEXT: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]"], output: "[[GFX90a:.+]]"
// MULTI-D-ONLY-NEXT: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[GFX90a]]"], output: "[[GFX90a_OUT:.+]]"
//
// RUN: %clang -### -target x86_64-linux-gnu --offload-new-driver -ccc-print-bindings \
// RUN:        --offload-arch=gfx90a --offload-arch=gfx908 --offload-device-only -c -o %t %s 2>&1 \
// RUN: | FileCheck -check-prefix=MULTI-D-ONLY-O %s
// MULTI-D-ONLY-O: error: cannot specify -o when generating multiple output files
