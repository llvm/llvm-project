// Test that -print-libgcc-file-name correctly respects -rtlib=compiler-rt.
// DEFINE: %{arch} = x86_64
// DEFINE: %{suffix} = -DSUFFIX=%if !per_target_runtime_dir %{-%{arch}%}

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=x86_64-pc-linux \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-X8664 %s %{suffix}
// CHECK-CLANGRT-X8664: libclang_rt.builtins[[SUFFIX]].a

// REDEFINE: %{arch} = i386
// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=i386-pc-linux \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-I386 %s %{suffix}
// CHECK-CLANGRT-I386: libclang_rt.builtins[[SUFFIX]].a

// Check whether alternate arch values map to the correct library.
//
// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=i686-pc-linux \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-I386 %s %{suffix}

// REDEFINE: %{arch} = arm
// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm-linux-gnueabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-ARM %s %{suffix}
// CHECK-CLANGRT-ARM: libclang_rt.builtins[[SUFFIX]].a

// REDEFINE: %{arch} = arm-android
// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm-linux-androideabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-ARM-ANDROID %s %{suffix}
// CHECK-CLANGRT-ARM-ANDROID: libclang_rt.builtins[[SUFFIX]].a

// REDEFINE: %{arch} = armhf
// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm-linux-gnueabihf \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-ARMHF %s %{suffix}
// CHECK-CLANGRT-ARMHF: libclang_rt.builtins[[SUFFIX]].a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm-linux-gnueabi -mfloat-abi=hard \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-ARM-ABI %s %{suffix}
// CHECK-CLANGRT-ARM-ABI: libclang_rt.builtins[[SUFFIX]].a

// REDEFINE: %{arch} = armv7m
// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=armv7m-none-eabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-ARM-BAREMETAL %s %{suffix}
// CHECK-CLANGRT-ARM-BAREMETAL: libclang_rt.builtins[[SUFFIX]].a

// Note this one will never have an arch suffix.
// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=armv7m-vendor-none-eabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-ARM-BAREMETAL-PER-TARGET %s
// CHECK-CLANGRT-ARM-BAREMETAL-PER-TARGET: libclang_rt.builtins.a
