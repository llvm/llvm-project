// REQUIRES: spirv-registered-target
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

// RUN: %clang -cc1 %s -triple spirv64-unknown-chipstar -emit-llvm-bc -o %t.dev.bc
// RUN: llvm-offload-binary -o %t.dev.out \
// RUN:   --image=file=%t.dev.bc,kind=hip,triple=spirv64-unknown-chipstar,arch=generic

// RUN: clang-linker-wrapper --dry-run \
// RUN:   --device-compiler=spirv64-unknown-chipstar=--hip-path="%S/Inputs/hipspv" \
// RUN:   --host-triple=spirv64-unknown-chipstar \
// RUN:   --linker-path=clang-offload-bundler \
// RUN:   --emit-fatbin-only -o /dev/null %t.dev.out \
// RUN: 2>&1 | FileCheck %s --check-prefix=WRAPPER -DHIP_PATH=%S/Inputs/hipspv

// WRAPPER: clang{{.*}}" --no-default-config -o {{[^ ]*.img}}
// WRAPPER-SAME: --target=spirv64-unknown-chipstar
// WRAPPER-SAME: -mllvm -spirv-ext=+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups
// WRAPPER-SAME: -x ir

// Linker wrapper with -no-use-spirv-backend uses llvm-spirv translator.
// RUN: clang-linker-wrapper --dry-run \
// RUN:   --device-compiler=spirv64-unknown-chipstar=--hip-path="%S/Inputs/hipspv" \
// RUN:   --device-compiler=spirv64-unknown-chipstar=-no-use-spirv-backend \
// RUN:   --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=ld --emit-fatbin-only -o /dev/null %t.dev.out \
// RUN: 2>&1 | FileCheck %s --check-prefix=WRAPPER-TRANSLATOR

// WRAPPER-TRANSLATOR: --spirv-max-version=1.2
// WRAPPER-TRANSLATOR-SAME: --spirv-ext=-all,+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups

// RUN: touch %t.dummy.o
// RUN: %clang -### --no-default-config -o %t.dummy.img \
// RUN:   --target=spirv64-unknown-chipstar %t.dummy.o \
// RUN:   --hip-path="%S/Inputs/hipspv" \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHIPSTAR -DHIP_PATH=%S/Inputs/hipspv

//      CHIPSTAR: {{".*llvm-link"}}
// CHIPSTAR-SAME: "-o" [[LINK_BC:".*bc"]] "{{[^ ]*.o}}"

//      CHIPSTAR: {{".*opt"}} [[LINK_BC]] "-load-pass-plugin"
// CHIPSTAR-SAME: "[[HIP_PATH]]/lib/libLLVMHipSpvPasses.so"
// CHIPSTAR-SAME: "-passes=hip-post-link-passes" "-o" [[LOWER_BC:".*bc"]]

// Default chipStar: in-tree SPIR-V backend (native).
//      CHIPSTAR: {{".*clang.*"}} "--no-default-config" "-c"
// CHIPSTAR-SAME: "--target=spirv64-unknown-chipstar"
// CHIPSTAR-SAME: "-mllvm" "-spirv-ext=+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups,+SPV_EXT_relaxed_printf_string_address_space"
// CHIPSTAR-SAME: [[LOWER_BC]] "-o" "[[SPIRV_OUT:.*img]]"

// chipStar with -no-use-spirv-backend: llvm-spirv translator path.
// RUN: %clang -### --no-default-config -o %t.dummy.img \
// RUN:   --target=spirv64-unknown-chipstar %t.dummy.o \
// RUN:   --hip-path="%S/Inputs/hipspv" -no-use-spirv-backend \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHIPSTAR-TRANSLATOR -DHIP_PATH=%S/Inputs/hipspv

//      CHIPSTAR-TRANSLATOR: {{".*llvm-link"}}
// CHIPSTAR-TRANSLATOR-SAME: "-o" [[LINK_BC:".*bc"]] "{{[^ ]*.o}}"

//      CHIPSTAR-TRANSLATOR: {{".*opt"}} [[LINK_BC]] "-load-pass-plugin"
// CHIPSTAR-TRANSLATOR-SAME: "[[HIP_PATH]]/lib/libLLVMHipSpvPasses.so"
// CHIPSTAR-TRANSLATOR-SAME: "-passes=hip-post-link-passes" "-o" [[LOWER_BC:".*bc"]]

//      CHIPSTAR-TRANSLATOR: {{".*llvm-spirv"}} "--spirv-max-version=1.2"
// CHIPSTAR-TRANSLATOR-SAME: "--spirv-ext=-all,+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups"
// CHIPSTAR-TRANSLATOR-SAME: [[LOWER_BC]] "-o" "[[SPIRV_OUT:.*img]]"

// RUN: %clang -### --no-default-config -o %t.dummy.img \
// RUN:   --target=spirv64v1.3-unknown-chipstar \
// RUN:   %t.dummy.o --hip-path="%S/Inputs/hipspv" \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHIPSTAR-SUBARCH -DHIP_PATH=%S/Inputs/hipspv

//      CHIPSTAR-SUBARCH: {{".*llvm-link"}}
// CHIPSTAR-SUBARCH-SAME: "-o" [[LINK_BC:".*bc"]] "{{[^ ]*.o}}"

//      CHIPSTAR-SUBARCH: {{".*opt"}} [[LINK_BC]] "-load-pass-plugin"
// CHIPSTAR-SUBARCH-SAME: "[[HIP_PATH]]/lib/libLLVMHipSpvPasses.so"
// CHIPSTAR-SUBARCH-SAME: "-passes=hip-post-link-passes" "-o" [[LOWER_BC:".*bc"]]

// Subarch chipStar also defaults to native backend.
//      CHIPSTAR-SUBARCH: {{".*clang.*"}} "--no-default-config" "-c"
// CHIPSTAR-SUBARCH-SAME: "--target=spirv64v1.3-unknown-chipstar"
// CHIPSTAR-SUBARCH-SAME: "-mllvm" "-spirv-ext=+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups,+SPV_EXT_relaxed_printf_string_address_space"
// CHIPSTAR-SUBARCH-SAME: [[LOWER_BC]] "-o" "[[SPIRV_OUT:.*img]]"

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

// RUN: env "PATH=%t/versioned" %clang -### --no-default-config \
// RUN:  -o %t.dummy.img --target=spirv64-unknown-chipstar %t.dummy.o \
// RUN:  --hip-path="%S/Inputs/hipspv" -no-use-spirv-backend -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefix=VERSIONED-CHIPSTAR

// Accept either versioned (llvm-spirv-[[VERSION]] from PATH) or clang-adjacent path.
// VERSIONED: {{.*}}llvm-spirv

// ChipStar with -no-use-spirv-backend may use clang-adjacent llvm-spirv (no version in name).
// VERSIONED-CHIPSTAR: --spirv-max-version=1.2
// VERSIONED-CHIPSTAR-SAME: --spirv-ext=-all,+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups
