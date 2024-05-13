// REQUIRES: x86-registered-target
// UNSUPPORTED: system-windows

// RUN: %clang -### -target x86_64-linux-gnu --offload=spirv64 \
// RUN:  --hip-path=%S/Inputs/hipspv -nogpuinc %s \
// RUN: 2>&1 | FileCheck --check-prefixes=FROM-HIP-PATH %s

// RUN: %clang -### -target x86_64-linux-gnu --offload=spirv64 \
// RUN: -nogpuinc -nogpulib --hipspv-pass-plugin=%S/Inputs/pass-plugin.so %s \
// RUN: 2>&1 | FileCheck --check-prefixes=FROM-OPTION %s

// RUN: not %clang -### --target=x86_64-linux-gnu --offload=spirv64 \
// RUN: -nogpuinc -nogpulib --hipspv-pass-plugin=foo.so %s \
// RUN: 2>&1 | FileCheck --check-prefixes=FROM-OPTION-INVALID %s

// RUN: %clang -### -target x86_64-linux-gnu --offload=spirv64 \
// RUN: -nogpuinc -nogpulib %s \
// RUN: 2>&1 | FileCheck --check-prefixes=NO-PLUGIN %s

// FROM-HIP-PATH: {{".*opt"}} {{".*.bc"}} "-load-pass-plugin"
// FROM-HIP-PATH-SAME: {{".*/Inputs/hipspv/lib/libLLVMHipSpvPasses.so"}}
// FROM-OPTION: {{".*opt"}} {{".*.bc"}} "-load-pass-plugin"
// FROM-OPTION-SAME: {{".*/Inputs/pass-plugin.so"}}
// FROM-OPTION-INVALID: error: no such file or directory: 'foo.so'
// NO-PLUGIN-NOT: {{".*opt"}} {{".*.bc"}} "-load-pass-plugin"
// NO-PLUGIN-NOT: {{".*/Inputs/hipspv/lib/libLLVMHipSpvPasses.so"}}
