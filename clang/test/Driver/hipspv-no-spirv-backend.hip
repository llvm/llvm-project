// Coverage for HIPSPV behavior that must not depend on whether the in-tree
// SPIR-V backend was built. Deliberately not guarded by the
// spirv-registered-target feature, so builders configured without the SPIR-V
// target (e.g. X86-only bots) exercise it too.

// UNSUPPORTED: system-windows

// The device compilation job layout is a property of the driver, not of the
// registered targets: an RDC device compile emits bitcode and defers the
// SPIR-V emission to clang-linker-wrapper.

// RUN: %clang -### -x hip --target=x86_64-linux-gnu --offload=spirv64 \
// RUN:   -fgpu-rdc --hip-path=%S/Inputs/hipspv \
// RUN:   -nohipwrapperinc %S/Inputs/hip_multiple_inputs/a.cu \
// RUN: 2>&1 | FileCheck --check-prefix=RDC-JOB %s

// RDC-JOB: "-cc1" "-triple" "spirv64"
// RDC-JOB-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// RDC-JOB-SAME: "-emit-llvm-bc"
// RDC-JOB-SAME: "-fcuda-is-device"
// RDC-JOB: {{".*clang-linker-wrapper.*"}}

// -fno-integrated-objemitter selects the external llvm-spirv translator, and
// it is accepted in either configuration.

// RUN: touch %t.o
// RUN: %clang -### --no-default-config -o %t.img \
// RUN:   --target=spirv64-unknown-chipstar %t.o --hip-path=%S/Inputs/hipspv \
// RUN:   -fno-integrated-objemitter \
// RUN: 2>&1 | FileCheck --check-prefix=XTOR -DHIP_PATH=%S/Inputs/hipspv %s

//      XTOR: {{".*llvm-link"}}
// XTOR-SAME: "-o" [[LINK_BC:".*bc"]] "{{[^ ]*.o}}"

//      XTOR: {{".*opt"}} [[LINK_BC]] "-load-pass-plugin"
// XTOR-SAME: "[[HIP_PATH]]/lib/libLLVMHipSpvPasses.so"
// XTOR-SAME: "-passes=hip-post-link-passes" "-o" [[LOWER_BC:".*bc"]]

//      XTOR: {{".*llvm-spirv.*"}} "--spirv-max-version=1.2"
// XTOR-SAME: [[LOWER_BC]] "-o" "{{.*img}}"
// XTOR-NOT: "-cc1"
