// RUN: %clang -### -emit-llvm --cuda-device-only \
// RUN:   -nocudalib -nocudainc --offload=spirv32-unknown-unknown -c %s 2>&1 | FileCheck %s

// Make sure there's no sm_* suffix on the output name
// CHECK: "-cc1" "-triple" "spirv32-unknown-unknown" {{.*}} "-fcuda-is-device" {{.*}} "-o" "cuda-device-triple-cuda-spirv32-unknown-unknown.bc"
