// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: touch %t.o
// RUN: %clang --hip-link -ccc-print-bindings -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 -fgpu-rdc %t.o\
// RUN: 2>&1 | FileCheck %s

// CHECK: # "amdgcn-amd-amdhsa" - "offload bundler", inputs: ["[[IN:.*o]]"], outputs: ["[[OBJ1:.*o]]", "[[OBJ2:.*o]]", "[[OBJ3:.*o]]"] 
// CHECK: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[OBJ2]]"], output: "[[IMG2:.*out]]"
// CHECK-NOT: offload bundler
// CHECK: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[OBJ3]]"], output: "[[IMG3:.*out]]"
// CHECK-NOT: offload bundler
// CHECK: # "x86_64-unknown-linux-gnu" - "GNU::Linker", inputs: ["[[OBJ1]]", "[[IMG2]]", "[[IMG3]]"], output: "a.out"

// RUN: %clang --hip-link -ccc-print-bindings -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %t.o\
// RUN: 2>&1 | FileCheck -check-prefix=NORDC %s

// NORDC-NOT: offload bundler
// NORDC: # "x86_64-unknown-linux-gnu" - "GNU::Linker", inputs: ["{{.*o}}"], output: "a.out"
