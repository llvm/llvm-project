// RUN: touch %t.o
// RUN: %clang --hip-link -ccc-print-bindings --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %t.o %S/Inputs/in.so \
// RUN:   -fgpu-rdc 2>&1 | FileCheck %s

// CHECK: # "x86_64-unknown-linux-gnu" - "offload bundler", inputs: ["[[IN:.*o]]"], outputs: ["[[HOSTOBJ:.*o]]", "{{.*o}}", "{{.*o}}"]
// CHECK: # "amdgcn-amd-amdhsa" - "offload bundler", inputs: ["[[IN]]"], outputs: ["{{.*o}}", "[[DOBJ1:.*o]]", "[[DOBJ2:.*o]]"]
// CHECK: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[DOBJ1]]"], output: "[[IMG1:.*out]]"
// CHECK-NOT: offload bundler
// CHECK: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[DOBJ2]]"], output: "[[IMG2:.*out]]"
// CHECK-NOT: offload bundler
// CHECK: # "amdgcn-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[IMG1]]", "[[IMG2]]"], output: "[[FATBINOBJ:.*o]]"
// CHECK-NOT: offload bundler
// CHECK: # "x86_64-unknown-linux-gnu" - "GNU::Linker", inputs: ["[[HOSTOBJ]]", "{{.*}}/Inputs/in.so", "[[FATBINOBJ]]"], output: "a.out"

