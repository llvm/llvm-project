// Verify that SPIR-V compilation does not crash during the llvm-link step
// due to extra args that are not meant to be forwarded there.
//
// RUN: %clang -### --target=spirv64-amd-amdhsa -use-spirv-backend \
// RUN:   -Xlinker -opt-bisect-limit=-1 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-LINKER-OPT
//
// RUN: %clang -### --target=spirv64-amd-amdhsa -use-spirv-backend \
// RUN:   -Xlinker -mllvm -Xlinker -opt-bisect-limit=-1 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-LINKER-MLLVM
//
// CHECK-LINKER-OPT: "{{.*}}llvm-link"
// CHECK-LINKER-OPT-SAME: "-o" "{{.*}}.bc" "{{.*}}.bc"{{$}}
//
// CHECK-LINKER-MLLVM: "{{.*}}llvm-link"
// CHECK-LINKER-MLLVM-SAME: "-o" "{{.*}}.bc" "{{.*}}.bc"{{$}}
