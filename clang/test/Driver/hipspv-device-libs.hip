// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// UNSUPPORTED: system-windows

// RUN: %clang -### -target x86_64-linux-gnu --offload=spirv64 \
// RUN:  --hip-path=%S/Inputs/hipspv %s \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,HIP-PATH %s

// Test --hip-device-lib-path
// RUN: %clang -### -target x86_64-linux-gnu --offload=spirv64 \
// RUN:  --hip-path=%S/Inputs/hipspv \
// RUN:   --hip-device-lib-path=%S/Inputs/hipspv-dev-lib %s \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,HIP-DEV-LIB-PATH %s

// Test --hip-device-lib w/ --hip-device-lib-path and HIP_DEVICE_LIB_PATH.
// RUN: env HIP_DEVICE_LIB_PATH=%S/Inputs/hipspv-dev-lib/a \
// RUN:   %clang -### -target x86_64-linux-gnu --offload=spirv64 \
// RUN:    --hip-path=%S/Inputs/hipspv \
// RUN:    --hip-device-lib-path=%S/Inputs/hipspv-dev-lib/b \
// RUN:    --hip-device-lib=a.bc --hip-device-lib=b.bc %s \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,HIP-DEV-LIB %s

// ALL: {{"[^"]*clang[^"]*"}}
// HIP-PATH: "-mlink-builtin-bitcode" {{".*/hipspv/lib/hip-device-lib/hipspv-spirv64.bc"}}
// HIP-DEV-LIB-PATH-NOT: "-mlink-builtin-bitcode" {{".*/hipspv/lib/hip-device-lib/hipspv-spirv64.bc"}}
// HIP-DEV-LIB-PATH: "-mlink-builtin-bitcode" {{".*/hipspv-dev-lib/hipspv-spirv64.bc"}}
// HIP-DEV-LIB: "-mlink-builtin-bitcode" {{".*/hipspv-dev-lib/a/a.bc"}}
// HIP-DEV-LIB-SAME: "-mlink-builtin-bitcode" {{".*/hipspv-dev-lib/b/b.bc"}}
