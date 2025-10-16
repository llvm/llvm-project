// RUN: %clang -x hip %s --cuda-device-only --offload-arch=amdgcnspirv -use-experimental-spirv-backend -nogpuinc -nogpulib -ccc-print-bindings 2>&1 | FileCheck %s 

// CHECK: # "spirv64-amd-amdhsa" - "clang", inputs: ["{{.*}}.c"], output: "[[SPV_FILE:.*.spv]]"
// CHECK: # "spirv64-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[SPV_FILE]]"], output: "{{.*.hipfb}}"
