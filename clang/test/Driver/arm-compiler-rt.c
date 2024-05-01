// DEFINE: %{arch} = arm
// DEFINE: %{suffix} = -DSUFFIX=%if !per_target_runtime_dir %{-%{arch}%}

// RUN: %clang -target arm-none-eabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-EABI %{suffix}
// ARM-EABI: "{{[^"]*}}libclang_rt.builtins[[SUFFIX]].a"

// RUN: %clang -target arm-linux-gnueabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-GNUEABI %{suffix}
// ARM-GNUEABI: "{{.*[/\\]}}libclang_rt.builtins[[SUFFIX]].a"

// REDEFINE: %{arch} = armhf
// RUN: %clang -target arm-linux-gnueabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -mfloat-abi=hard -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-GNUEABI-ABI %{suffix}
// ARM-GNUEABI-ABI: "{{.*[/\\]}}libclang_rt.builtins[[SUFFIX]].a"

// RUN: %clang -target arm-linux-gnueabihf \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-GNUEABIHF %{suffix}
// ARM-GNUEABIHF: "{{.*[/\\]}}libclang_rt.builtins[[SUFFIX]].a"

// REDEFINE: %{arch} = arm
// RUN: %clang -target arm-linux-gnueabihf \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -mfloat-abi=soft -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-GNUEABIHF-ABI %{suffix}
// ARM-GNUEABIHF-ABI: "{{.*[/\\]}}libclang_rt.builtins[[SUFFIX]].a"

// RUN: %clang -target arm-windows-itanium \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-WINDOWS %{suffix}
// ARM-WINDOWS: "{{.*[/\\]}}clang_rt.builtins[[SUFFIX]].lib"

// REDEFINE: %{arch} = arm-android
// RUN: %clang -target arm-linux-androideabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-ANDROID %{suffix}
// ARM-ANDROID: "{{.*[/\\]}}libclang_rt.builtins[[SUFFIX]].a"

// REDEFINE: %{arch} = armhf-android
// RUN: not %clang --target=arm-linux-androideabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -rtlib=compiler-rt -mfloat-abi=hard -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix ARM-ANDROIDHF %{suffix}
// ARM-ANDROIDHF: "{{.*[/\\]}}libclang_rt.builtins[[SUFFIX]].a"

