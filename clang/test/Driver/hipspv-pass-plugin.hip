// UNSUPPORTED: system-windows

// RUN: %clang -### -target x86_64-linux-gnu --offload=spirv64 \
// RUN: --no-offload-new-driver --hip-path=%S/Inputs/hipspv -nogpuinc %s \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,FROM-HIP-PATH %s

// RUN: %clang -### -target x86_64-linux-gnu --offload=spirv64 \
// RUN: --no-offload-new-driver -nogpuinc -nogpulib --hipspv-pass-plugin=%S/Inputs/pass-plugin.so %s \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,FROM-OPTION %s

// RUN: not %clang -### --target=x86_64-linux-gnu --offload=spirv64 \
// RUN: --no-offload-new-driver -nogpuinc -nogpulib --hipspv-pass-plugin=foo.so %s \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,FROM-OPTION-INVALID %s

// RUN: %clang -### -target x86_64-linux-gnu --offload=spirv64 \
// RUN: --no-offload-new-driver -nogpuinc -nogpulib %s \
// RUN: 2>&1 | FileCheck --check-prefixes=ALL,NO-PLUGIN %s

// RUN: %clang --offload-new-driver -nogpuinc -nogpulib \
// RUN:   --offload=spirv64-unknown-chipstar \
// RUN:   --hip-path=%S/Inputs/hipspv -Xoffload-compiler \
// RUN:   '-###' -c %s -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefixes=ALL,FROM-HIP-PATH

// RUN: %clang --offload-new-driver -nogpuinc -nogpulib \
// RUN:   --offload=spirv64-unknown-chipstar \
// RUN:   -Xoffload-compiler-spirv64-unknown-chipstar \
// RUN:   --hipspv-pass-plugin=%S/Inputs/pass-plugin.so -Xoffload-compiler \
// RUN:   '-###' -c %s -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefixes=ALL,FROM-OPTION

// RUN: not %clang --offload-new-driver -nogpuinc -nogpulib \
// RUN:   --offload=spirv64-unknown-chipstar \
// RUN:   -Xoffload-compiler-spirv64-unknown-chipstar \
// RUN:   --hipspv-pass-plugin=foo.so -Xoffload-compiler \
// RUN:   '-###' -c %s -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefixes=ALL,FROM-OPTION-INVALID

// RUN: %clang --offload-new-driver -nogpuinc -nogpulib \
// RUN:   --offload=spirv64-unknown-chipstar -Xoffload-compiler \
// RUN:   '-###' -c %s -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefixes=ALL,NO-PLUGIN

// FROM-HIP-PATH: {{".*opt"}} {{".*.bc"}} "-load-pass-plugin"
// FROM-HIP-PATH-SAME: {{".*/Inputs/hipspv/lib/libLLVMHipSpvPasses.so"}}
// FROM-OPTION: {{".*opt"}} {{".*.bc"}} "-load-pass-plugin"
// FROM-OPTION-SAME: {{".*/Inputs/pass-plugin.so"}}
// FROM-OPTION-INVALID: error: no such file or directory: 'foo.so'
// NO-PLUGIN-NOT: {{".*opt"}} {{".*.bc"}} "-load-pass-plugin"
// NO-PLUGIN-NOT: {{".*/Inputs/hipspv/lib/libLLVMHipSpvPasses.so"}}
// ALL: {{".*llvm-spirv[^ ]*"}}
