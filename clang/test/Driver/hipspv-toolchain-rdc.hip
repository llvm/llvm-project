// REQUIRES: x86-registered-target
// UNSUPPORTED: system-windows

// RUN: %clang -### -x hip -target x86_64-linux-gnu --offload=spirv64 \
// RUN:   --no-offload-new-driver -fgpu-rdc --hip-path=%S/Inputs/hipspv -nohipwrapperinc \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s

// Emit objects for host side path
// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-aux-triple" "spirv64"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" [[A_OBJ_HOST:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC:".*a.cu"]]

// CHECK: [[CLANG]] "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-aux-triple" "spirv64"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" [[B_OBJ_HOST:".*o"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC:".*b.hip"]]

// Emit code (LLVM BC) for device side path.
// CHECK: [[CLANG]] "-cc1" "-triple" "spirv64"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: "-fcuda-is-device" "-fcuda-allow-variadic-functions"
// CHECK-SAME: "-fvisibility=hidden" "-fapply-global-visibility-to-externs"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" [[A_BC1:".*bc"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[A_SRC]]

// CHECK: [[CLANG]] "-cc1" "-triple" "spirv64"
// CHECK-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-SAME: "-emit-llvm-bc"
// CHECK-SAME: "-fcuda-is-device" "-fcuda-allow-variadic-functions"
// CHECK-SAME: "-fvisibility=hidden" "-fapply-global-visibility-to-externs"
// CHECK-SAME: "-fgpu-rdc"
// CHECK-SAME: {{.*}} "-o" [[B_BC1:".*bc"]] "-x" "hip"
// CHECK-SAME: {{.*}} [[B_SRC]]

// Link device code, lower it with HIPSPV passes and emit SPIR-V binary.
// CHECK: {{".*llvm-link.*"}} [[A_BC1]] [[B_BC1]] "-o" [[AB_LINK:".*bc"]]
// CHECK: {{".*opt.*"}} [[AB_LINK]] "-load-pass-plugin"
// CHECK-SAME: "{{.*}}/Inputs/hipspv/lib/libLLVMHipSpvPasses.so"
// CHECK-SAME: "-o" [[AB_LOWER:".*bc"]]
// CHECK: {{".*llvm-spirv"}} "--spirv-max-version=1.1" "--spirv-ext=+all"
// CHECK-SAME: [[AB_LOWER]] "-o" "[[AB_SPIRV:.*out]]"

// Construct fat binary object.
// CHECK: [[BUNDLER:".*clang-offload-bundler"]] "-type=o" "-bundle-align=4096"
// CHECK-SAME: "-targets={{.*}},hip-spirv64----generic"
// CHECK-SAME: "-input=/dev/null" "-input=[[AB_SPIRV]]"
// CHECK-SAME: "-output=[[AB_FATBIN:.*hipfb]]"
// CHECK: {{".*llvm-mc.*"}} "-o" [[OBJBUNDLE:".*o"]] "{{.*}}.mcin"
// CHECK-SAME: "--filetype=obj"

// Output the executable
// CHECK: {{".*ld.*"}} {{.*}}"-o" "a.out" {{.*}} [[A_OBJ_HOST]] [[B_OBJ_HOST]]
// CHECK-SAME: [[OBJBUNDLE]]
