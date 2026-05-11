// Verify that SPIR-V compilation does not crash during the llvm-link step
// due to extra args that are not meant to be forwarded there.
//
// RUN: %clang -### --target=spirv64-amd-amdhsa -use-spirv-backend \
// RUN:   -Xlinker -opt-bisect-limit=-1 %s 2>&1 \
// RUN:   | FileCheck %s
//
// RUN: %clang -### --target=spirv64-amd-amdhsa -use-spirv-backend \
// RUN:   -Xlinker -mllvm -Xlinker -opt-bisect-limit=-1 %s 2>&1 \
// RUN:   | FileCheck %s
//
// CHECK: "{{.*}}llvm-link"
// CHECK-NOT: opt-bisect-limit
// CHECK-NOT: -mllvm
// CHECK-SAME: "-o" "{{.*}}.bc" "{{.*}}.bc"{{$}}
