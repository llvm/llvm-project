// RUN: %clang -ccc-print-bindings --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   --no-offload-new-driver -c 2>&1 | FileCheck -check-prefix=NRDCS %s
// RUN: %clang -ccc-print-bindings --target=x86_64-linux-gnu --offload-new-driver \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   --no-offload-new-driver -c 2>&1 | FileCheck -check-prefix=NRDCS %s
// NRDCS: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[IN:.*hip-binding.hip]]"], output: "[[OBJ1:.*o]]"
// NRDCS: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[OBJ1]]"], output: "[[IMG1:.*]]"
// NRDCS: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[IN:.*hip-binding.hip]]"], output: "[[OBJ2:.*o]]"
// NRDCS: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[OBJ2]]"], output: "[[IMG2:.*]]"
// NRDCS: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[IMG1]]", "[[IMG2]]"], output: "[[FATBIN:.*]]"
// NRDCS: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[IN]]", "[[FATBIN]]"], output: "{{.*}}"

// RUN: %clang -ccc-print-bindings --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   --no-offload-new-driver -c -fgpu-rdc 2>&1 | FileCheck -check-prefix=RDCS %s
// RDCS: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[IN:.*hip-binding.hip]]"], output: "[[BC1:.*bc]]"
// RDCS: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[IN:.*hip-binding.hip]]"], output: "[[BC2:.*bc]]"
// RDCS: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[IN]]"], output: "[[HOSTOBJ:.*o]]"
// RDCS: # "x86_64-unknown-linux-gnu" - "offload bundler", inputs: ["[[BC1]]", "[[BC2]]", "[[HOSTOBJ]]"], output: "{{.*}}"

// RUN: %clang -ccc-print-bindings --target=x86_64-linux-gnu --offload-new-driver \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN:   -c -fgpu-rdc 2>&1 | FileCheck -check-prefix=RDCS-NEW %s
// RDCS-NEW: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HIP803:.+]]"
// RDCS-NEW: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]"], output: "[[HIP900:.+]]"
// RDCS-NEW: # "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[HIP803]]", "[[HIP900]]"], output: "[[HIPFB:.+]]"
// RDCS-NEW: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT]]", "[[HIPFB]]"], output: "{{.*}}"

// RUN: touch %t.o
// RUN: %clang --hip-link -ccc-print-bindings --target=x86_64-linux-gnu \
// RUN:   --no-offload-new-driver --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 -fgpu-rdc %t.o\
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

// RUN: %clang --hip-link -ccc-print-bindings --target=x86_64-linux-gnu \
// RUN:   --no-offload-new-driver --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %t.o\
// RUN: 2>&1 | FileCheck -check-prefix=NORDC %s

// NORDC-NOT: offload bundler
// NORDC: # "x86_64-unknown-linux-gnu" - "GNU::Linker", inputs: ["{{.*o}}"], output: "a.out"

//
// Check to make sure we can generate multiple outputs for device-only
// compilation and fail with '-o'.
//
// RUN: %clang -### --target=x86_64-linux-gnu --offload-new-driver -ccc-print-bindings -nogpulib -nogpuinc \
// RUN:        --offload-arch=gfx90a --offload-arch=gfx908 --offload-device-only -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=MULTI-D-ONLY %s
//      MULTI-D-ONLY: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[GFX908:.+]]"
// MULTI-D-ONLY-NEXT: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[GFX908]]"], output: "[[GFX908_OUT:.+]]"
// MULTI-D-ONLY-NEXT: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]"], output: "[[GFX90a:.+]]"
// MULTI-D-ONLY-NEXT: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[GFX90a]]"], output: "[[GFX90a_OUT:.+]]"
//
// RUN: not %clang -### --target=x86_64-linux-gnu --offload-new-driver -ccc-print-bindings -nogpulib -nogpuinc \
// RUN:        --no-gpu-bundle-output --offload-arch=gfx90a --offload-arch=gfx908 --offload-device-only -c -o %t %s 2>&1 \
// RUN: | FileCheck -check-prefix=MULTI-D-ONLY-NO-BUNDLE-O %s
// MULTI-D-ONLY-NO-BUNDLE-O: error: cannot specify -o when generating multiple output files

// RUN: %clang -### --target=x86_64-linux-gnu --offload-new-driver -ccc-print-bindings -nogpulib -nogpuinc \
// RUN:        --gpu-bundle-output --offload-arch=gfx90a --offload-arch=gfx908 --offload-device-only -c -o a.out %s 2>&1 \
// RUN: | FileCheck -check-prefix=MULTI-D-ONLY-O %s
//      MULTI-D-ONLY-O: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[GFX908_OBJ:.+]]"
// MULTI-D-ONLY-O-NEXT: "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[GFX908_OBJ]]"], output: "[[GFX908:.+]]"
// MULTI-D-ONLY-O-NEXT: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]"], output: "[[GFX90A_OBJ:.+]]"
// MULTI-D-ONLY-O-NEXT: "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[GFX90A_OBJ]]"], output: "[[GFX90A:.+]]"
// MULTI-D-ONLY-O-NEXT: "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[GFX908]]", "[[GFX90A]]"], output: "a.out"

//
// Check to ensure that we can use '-fsyntax-only' for HIP output with the new
// driver.
// 
// RUN: %clang -### --target=x86_64-unknown-linux-gnu --offload-new-driver -nogpulib -nogpuinc \
// RUN:        -fsyntax-only --offload-arch=gfx90a --offload-arch=gfx908 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=SYNTAX-ONLY %s
// SYNTAX-ONLY: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-fsyntax-only"
// SYNTAX-ONLY: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-fsyntax-only"
// SYNTAX-ONLY: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-fsyntax-only"

//
// Check to ensure that we can use '-foffload' when not operating in RDC-mode.
//
// RUN: %clang -### --target=x86_64-linux-gnu -fno-gpu-rdc --offload-new-driver -ccc-print-bindings \
// RUN:        -nogpulib -nogpuinc -foffload-lto --offload-arch=gfx90a --offload-arch=gfx908 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=LTO-NO-RDC %s
//      LTO-NO-RDC: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[LTO_908:.+]]"
// LTO-NO-RDC-NEXT: # "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]"], output: "[[LTO_90A:.+]]"
// LTO-NO-RDC-NEXT: # "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[LTO_908]]", "[[LTO_90A]]"], output: "[[PKG:.+]]"
// LTO-NO-RDC-NEXT: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT]]", "[[PKG]]"], output: "[[OBJ:.+]]"
// LTO-NO-RDC-NEXT: # "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[OBJ]]"], output: "hip-binding.o"
