// Test that gcc-toolchain option works correctly with a aarch64-linux-gnu
// triple.
//
// RUN: %clang %s -### -v --target=aarch64-linux-gnu \
// RUN:   --gcc-toolchain=%S/Inputs/basic_android_ndk_tree/ 2>&1 \
// RUN: | FileCheck %s
//
// CHECK: Selected GCC installation: {{.*}}/Inputs/basic_android_ndk_tree/lib/gcc/aarch64-linux-android/4.9
