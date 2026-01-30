// UNSUPPORTED: system-windows, system-cygwin

// RUN: %clang -### -target x86_64-linux-gnu --offload=spirv64 \
// RUN:   --no-offload-new-driver --hip-path=%S/Inputs/hipspv -nohipwrapperinc %s \
// RUN: 2>&1 | FileCheck --check-prefixes=CHECK,OLD \
// RUN:   -DTRIPLE=spirv64 %s

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --offload=spirv64-unknown-chipstar \
// RUN:   --offload-new-driver --hip-path=%S/Inputs/hipspv -nohipwrapperinc %s \
// RUN: 2>&1 | FileCheck --check-prefixes=CHECK,NEW \
// RUN:   -DTRIPLE=spirv64-unknown-chipstar -DHIP_PATH=%S/Inputs/hipspv %s

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "[[TRIPLE]]"
// CHECK-SAME: "-aux-triple" "{{.*}}" "-emit-llvm-bc"
// CHECK-SAME: "-fcuda-is-device"
// CHECK-SAME: "-fcuda-allow-variadic-functions"
// CHECK-SAME: "-mlink-builtin-bitcode" {{".*/hipspv/lib/hip-device-lib/hipspv-}}[[TRIPLE]].bc"
// CHECK-SAME: "-isystem" {{".*/hipspv/include"}}
// CHECK-SAME: "-fhip-new-launch-api"
// CHECK-SAME: "-o" "[[OBJ_DEV:.*(o|bc)]]"
// CHECK-SAME: "-x" "hip"

// OLD: {{".*llvm-link"}} "-o" [[LINK_BC:".*bc"]] "[[OBJ_DEV]]"

// OLD: {{".*opt"}} [[LINK_BC]] "-load-pass-plugin"
// OLD-SAME: {{".*/hipspv/lib/libLLVMHipSpvPasses.so"}}
// OLD-SAME: "-passes=hip-post-link-passes" "-o" [[LOWER_BC:".*bc"]]

// OLD: {{".*llvm-spirv"}} "--spirv-max-version=1.1" "--spirv-ext=+all"
// OLD-SAME: [[LOWER_BC]] "-o" "[[SPIRV_OUT:.*out]]"

// OLD: {{".*clang-offload-bundler"}} "-type=o" "-bundle-align=4096"
// OLD-SAME: "-targets=host-x86_64-unknown-linux-gnu,hip-spirv64----generic"
// OLD-SAME: "-input={{.*}}" "-input=[[SPIRV_OUT]]" "-output=[[BUNDLE:.*hipfb]]"

// NEW: {{".*llvm-offload-binary"}} "-o" "[[PACKAGE:.*.out]]"
// NEW-SAME: "--image=file=[[OBJ_DEV]],triple=[[TRIPLE]],arch=generic,kind=hip"

// NEW: {{".*clang-linker-wrapper"}} "--device-compiler=[[TRIPLE]]=--hip-path=[[HIP_PATH]]"
// NEW-SAME: "--emit-fatbin-only" "-o" "[[BUNDLE:.*hipfb]]"

// CHECK: [[CLANG]] "-cc1" "-triple" {{".*"}} "-aux-triple" "[[TRIPLE]]"
// CHECK-SAME: "-emit-obj"
// CHECK-SAME: "-fcuda-include-gpubinary" "[[BUNDLE]]"
// CHECK-SAME: "-o" [[OBJ_HOST:".*o"]] "-x" "hip"

// OLD: {{".*ld.*"}} {{.*}}[[OBJ_HOST]]

// NEW: {{".*clang-linker-wrapper"}}
// NEW-SAME: "--linker-path={{.*ld.*}}" "-o" "a.out"
// NEW-SAME: [[OBJ_HOST]]

//------------------------------------------------------------------------------
// Check the clang command, invoked by the linker wrapper, selects the HIPSPV
// toolchain for the new offload driver.

// RUN: %clang -target x86_64-linux-gnu --offload-new-driver -nogpuinc \
// RUN:   -nogpulib --offload=spirv64-unknown-chipstar \
// RUN:   --hip-path=%S/Inputs/hipspv -Xoffload-compiler \
// RUN:   '-###' -c %s -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHIPSTAR

//      CHIPSTAR: {{".*llvm-link"}}
// CHIPSTAR-SAME: "-o" [[LINK_BC:".*bc"]] "{{[^ ]*.o}}"

//      CHIPSTAR: {{".*opt"}} [[LINK_BC]] "-load-pass-plugin"
// CHIPSTAR-SAME: {{".*/hipspv/lib/libLLVMHipSpvPasses.so"}}
// CHIPSTAR-SAME: "-passes=hip-post-link-passes" "-o" [[LOWER_BC:".*bc"]]

//      CHIPSTAR: {{".*llvm-spirv"}} "--spirv-max-version=1.2"
// CHIPSTAR-SAME: "--spirv-ext=-all,+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups"
// CHIPSTAR-SAME: [[LOWER_BC]] "-o" "[[SPIRV_OUT:.*img]]"

// RUN: %clang -target x86_64-linux-gnu --offload-new-driver -nogpuinc \
// RUN:   -nogpulib --offload=spirv64v1.3-unknown-chipstar \
// RUN:   --hip-path=%S/Inputs/hipspv -Xoffload-compiler \
// RUN:   '-###' -c %s -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHIPSTAR_SUBARCH

//      CHIPSTAR_SUBARCH: {{".*llvm-link"}}
// CHIPSTAR_SUBARCH-SAME: "-o" [[LINK_BC:".*bc"]] "{{[^ ]*.o}}"

//      CHIPSTAR_SUBARCH: {{".*opt"}} [[LINK_BC]] "-load-pass-plugin"
// CHIPSTAR_SUBARCH-SAME: {{".*/hipspv/lib/libLLVMHipSpvPasses.so"}}
// CHIPSTAR_SUBARCH-SAME: "-passes=hip-post-link-passes" "-o" [[LOWER_BC:".*bc"]]

//      CHIPSTAR_SUBARCH: {{".*llvm-spirv"}}
// CHIPSTAR_SUBARCH-SAME: "--spirv-ext=-all,+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups"
// CHIPSTAR_SUBARCH-SAME: [[LOWER_BC]] "-o" "[[SPIRV_OUT:.*img]]"

//-----------------------------------------------------------------------------
// Check llvm-spirv-<LLVM_VERSION_MAJOR> is used if it is found in PATH.
// RUN: mkdir -p %t/versioned
// RUN: touch %t/versioned/llvm-spirv-%llvm-version-major \
// RUN:   && chmod +x %t/versioned/llvm-spirv-%llvm-version-major
// RUN: env "PATH=%t/versioned" %clang -### -target x86_64-linux-gnu \
// RUN:   --offload=spirv64 --hip-path=%S/Inputs/hipspv -nohipwrapperinc \
// RUN:   --no-offload-new-driver %s 2>&1 \
// RUN:   | FileCheck -DVERSION=%llvm-version-major \
// RUN:   --check-prefix=VERSIONED %s

// RUN: env "PATH=%t/versioned" %clang -target x86_64-linux-gnu \
// RUN:   --offload-new-driver -nogpuinc \
// RUN:   -nogpulib --offload=spirv64-unknown-chipstar \
// RUN:   --hip-path=%S/Inputs/hipspv -Xoffload-compiler '-###' -c %s \
// RUN:   -o /dev/null 2>&1 \
// RUN: | FileCheck -DVERSION=%llvm-version-major --check-prefix=VERSIONED %s

// VERSIONED: {{.*}}llvm-spirv-[[VERSION]]
