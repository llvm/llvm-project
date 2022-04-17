
// RUN: %clang -### -emit-llvm --cuda-device-only \
// RUN:   -nocudalib -nocudainc --offload=spirv32-unknown-unknown -c %s 2>&1 | FileCheck %s

// CHECK: "-cc1" "-triple" "spirv32-unknown-unknown" {{.*}} "-fcuda-is-device" {{.*}}
