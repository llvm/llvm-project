// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// UNSUPPORTED: system-windows

// RUN: %clang -### -target x86_64-linux-gnu --offload=spirv64 \
// RUN:   --hip-path=%S/Inputs/hipspv -nohipwrapperinc %s \
// RUN: 2>&1 | FileCheck %s

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "spirv64"
// CHECK-SAME: "-aux-triple" "{{.*}}" "-emit-llvm-bc"
// CHECK-SAME: "-fcuda-is-device"
// CHECK-SAME: "-fcuda-allow-variadic-functions"
// CHECK-SAME: "-mlink-builtin-bitcode" {{".*/hipspv/lib/hip-device-lib/hipspv-spirv64.bc"}}
// CHECK-SAME: "-isystem" {{".*/hipspv/include"}}
// CHECK-SAME: "-fhip-new-launch-api"
// CHECK-SAME: "-o" [[DEV_BC:".*bc"]]
// CHECK-SAME: "-x" "hip"

// CHECK: {{".*llvm-link"}} [[DEV_BC]] "-o" [[LINK_BC:".*bc"]]

// CHECK: {{".*opt"}} [[LINK_BC]] "-load-pass-plugin"
// CHECK-SAME: {{".*/hipspv/lib/libLLVMHipSpvPasses.so"}}
// CHECK-SAME: "-passes=hip-post-link-passes" "-o" [[LOWER_BC:".*bc"]]

// CHECK: {{".*llvm-spirv"}} "--spirv-max-version=1.1" "--spirv-ext=+all"
// CHECK-SAME: [[LOWER_BC]] "-o" "[[SPIRV_OUT:.*out]]"

// CHECK: {{".*clang-offload-bundler"}} "-type=o" "-bundle-align=4096"
// CHECK-SAME: "-targets=host-x86_64-unknown-linux,hip-spirv64----generic"
// CHECK-SAME: "-input={{.*}}" "-input=[[SPIRV_OUT]]" "-output=[[BUNDLE:.*hipfb]]"

// CHECK: [[CLANG]] "-cc1" "-triple" {{".*"}} "-aux-triple" "spirv64"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: "-fcuda-include-gpubinary" "[[BUNDLE]]"
// CHECK-SAME: "-o" [[OBJ_HOST:".*o"]] "-x" "hip"

// CHECK: {{".*ld.*"}} {{.*}}[[OBJ_HOST]]
