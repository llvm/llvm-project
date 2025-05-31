// RUN: %clang -target arm-none-eabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-EABI
// ARM-EABI: "{{[^"]*}}libclang_rt.builtins.a"

// RUN: %clang -target arm-linux-gnueabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-GNUEABI
// ARM-GNUEABI: "{{.*[/\\]}}libclang_rt.builtins.a"

// RUN: %clang -target arm-linux-gnueabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -mfloat-abi=hard -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-GNUEABI-ABI
// ARM-GNUEABI-ABI: "{{.*[/\\]}}libclang_rt.builtins.a"

// RUN: %clang -target arm-linux-gnueabihf \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-GNUEABIHF
// ARM-GNUEABIHF: "{{.*[/\\]}}libclang_rt.builtins.a"

// RUN: %clang -target arm-linux-gnueabihf \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -mfloat-abi=soft -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-GNUEABIHF-ABI
// ARM-GNUEABIHF-ABI: "{{.*[/\\]}}libclang_rt.builtins.a"

// RUN: %clang -target arm-windows-itanium \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-WINDOWS
// ARM-WINDOWS: "{{.*[/\\]}}clang_rt.builtins.lib"

// RUN: %clang -target arm-linux-androideabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-ANDROID
// ARM-ANDROID: "{{.*[/\\]}}libclang_rt.builtins.a"

// RUN: not %clang --target=arm-linux-androideabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -mfloat-abi=hard -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-ANDROIDHF
// ARM-ANDROIDHF: "{{.*[/\\]}}libclang_rt.builtins.a"

