// Test that -print-libgcc-file-name correctly respects -rtlib=compiler-rt.

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=x86_64-pc-linux \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-X8664 %s
// CHECK-CLANGRT-X8664: libclang_rt.builtins-x86_64.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=i386-pc-linux \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-I386 %s
// CHECK-CLANGRT-I386: libclang_rt.builtins-i386.a

// Check whether alternate arch values map to the correct library.
//
// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=i686-pc-linux \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-I386 %s

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm-linux-gnueabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-ARM %s
// CHECK-CLANGRT-ARM: libclang_rt.builtins-arm.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm-linux-androideabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-ARM-ANDROID %s
// CHECK-CLANGRT-ARM-ANDROID: libclang_rt.builtins-arm-android.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm-linux-gnueabihf \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-ARMHF %s
// CHECK-CLANGRT-ARMHF: libclang_rt.builtins-armhf.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm-linux-gnueabi -mfloat-abi=hard \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-ARM-ABI %s
// CHECK-CLANGRT-ARM-ABI: libclang_rt.builtins-armhf.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=armv7m-none-eabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_arch_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-ARM-BAREMETAL %s
// CHECK-CLANGRT-ARM-BAREMETAL: libclang_rt.builtins-armv7m.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=armv7m-vendor-none-eabi \
// RUN:     --sysroot=%S/Inputs/resource_dir_with_arch_subdir \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-ARM-BAREMETAL-PER-TARGET %s
// CHECK-CLANGRT-ARM-BAREMETAL-PER-TARGET: libclang_rt.builtins.a
