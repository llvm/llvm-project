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
// NEW-SAME: "--no-lto"
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

// Test the in-tree SPIR-V backend path (no llvm-spirv available).
// Run from a directory that doesn't contain llvm-spirv and use
// --no-canonical-prefixes so getExecutableDir() looks there instead of
// the build bin dir. Empty PATH ensures PATH lookup also fails.
// RUN: mkdir -p %t/no-spirv %t/empty
// RUN: ln -sf clang-linker-wrapper %t/no-spirv/clang-linker-wrapper
// RUN: env "PATH=%t/empty" %t/no-spirv/clang-linker-wrapper \
// RUN:   --no-canonical-prefixes --dry-run \
// RUN:   --device-compiler=spirv64-unknown-chipstar=--hip-path="%S/Inputs/hipspv" \
// RUN:   --host-triple=spirv64-unknown-chipstar \
// RUN:   --linker-path=clang-offload-bundler \
// RUN:   --emit-fatbin-only -o /dev/null %t.dev.out \
// RUN: 2>&1 | FileCheck %s --check-prefix=WRAPPER -DHIP_PATH=%S/Inputs/hipspv

// The linker wrapper runs opt (HipSpvPasses) then uses the in-tree SPIR-V
// backend when llvm-spirv is not available.
// WRAPPER: "{{.*}}opt" {{.*}}-load-pass-plugin
// WRAPPER-SAME: {{.*}}libLLVMHipSpvPasses.so
// WRAPPER-SAME: -passes=hip-post-link-passes

// WRAPPER: "{{.*}}clang{{.*}}" --no-default-config -o
// WRAPPER-SAME: --target=spirv64-unknown-chipstar
// WRAPPER-SAME: -mllvm -spirv-ext=+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups,+SPV_EXT_relaxed_printf_string_address_space,+SPV_KHR_bit_instructions,+SPV_EXT_shader_atomic_float_add
// WRAPPER-SAME: -c -x ir

// Test the llvm-spirv translator path (llvm-spirv available in executable dir).
// Place a fake llvm-spirv next to the clang-linker-wrapper symlink.
// RUN: mkdir -p %t/with-spirv
// RUN: ln -sf clang-linker-wrapper %t/with-spirv/clang-linker-wrapper
// RUN: touch %t/with-spirv/llvm-spirv && chmod +x %t/with-spirv/llvm-spirv
// RUN: env "PATH=%t/empty" %t/with-spirv/clang-linker-wrapper \
// RUN:   --no-canonical-prefixes --dry-run \
// RUN:   --device-compiler=spirv64-unknown-chipstar=--hip-path="%S/Inputs/hipspv" \
// RUN:   --host-triple=spirv64-unknown-chipstar \
// RUN:   --linker-path=clang-offload-bundler \
// RUN:   --emit-fatbin-only -o /dev/null %t.dev.out \
// RUN: 2>&1 | FileCheck %s --check-prefix=WRAPPER-TR

// WRAPPER-TR: "{{.*}}opt" {{.*}}-load-pass-plugin
// WRAPPER-TR-SAME: {{.*}}libLLVMHipSpvPasses.so
// WRAPPER-TR-SAME: -passes=hip-post-link-passes

// WRAPPER-TR: "{{.*}}llvm-spirv" {{.*}}--spirv-max-version=1.2
// WRAPPER-TR-SAME: --spirv-ext=-all,+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups

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

//      CHIPSTAR: {{".*clang.*"}} "--no-default-config" "-c"
// CHIPSTAR-SAME: "--target=spirv64-unknown-chipstar"
// CHIPSTAR-SAME: "-mllvm" "-spirv-ext=+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups,+SPV_EXT_relaxed_printf_string_address_space,+SPV_KHR_bit_instructions,+SPV_EXT_shader_atomic_float_add"
// CHIPSTAR-SAME: [[LOWER_BC]] "-o" "[[SPIRV_OUT:.*img]]"

// RUN: %clang -### --no-default-config -o %t.dummy.img \
// RUN:   --target=spirv64v1.3-unknown-chipstar \
// RUN:   %t.dummy.o --hip-path="%S/Inputs/hipspv" \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHIPSTAR-SUBARCH -DHIP_PATH=%S/Inputs/hipspv

//      CHIPSTAR-SUBARCH: {{".*llvm-link"}}
// CHIPSTAR-SUBARCH-SAME: "-o" [[LINK_BC:".*bc"]] "{{[^ ]*.o}}"

//      CHIPSTAR-SUBARCH: {{".*opt"}} [[LINK_BC]] "-load-pass-plugin"
// CHIPSTAR-SUBARCH-SAME: "[[HIP_PATH]]/lib/libLLVMHipSpvPasses.so"
// CHIPSTAR-SUBARCH-SAME: "-passes=hip-post-link-passes" "-o" [[LOWER_BC:".*bc"]]

//      CHIPSTAR-SUBARCH: {{".*clang.*"}} "--no-default-config" "-c"
// CHIPSTAR-SUBARCH-SAME: "--target=spirv64v1.3-unknown-chipstar"
// CHIPSTAR-SUBARCH-SAME: "-mllvm" "-spirv-ext=+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups,+SPV_EXT_relaxed_printf_string_address_space,+SPV_KHR_bit_instructions,+SPV_EXT_shader_atomic_float_add"
// CHIPSTAR-SUBARCH-SAME: [[LOWER_BC]] "-o" "[[SPIRV_OUT:.*img]]"

// Check unknown linker options are ignored - such as ones that are targeted at
// spirv-link. HIPSPV toolchain does linking via llvm-link.
// RUN: %clang -### --no-default-config -o %t.dummy.img \
// RUN:   --target=spirv64-unknown-chipstar %t.dummy.o \
// RUN:   --hip-path="%S/Inputs/hipspv" -Xlinker foobar \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHIPSTAR -DHIP_PATH=%S/Inputs/hipspv

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

// VERSIONED: {{.*}}llvm-spirv-[[VERSION]]
