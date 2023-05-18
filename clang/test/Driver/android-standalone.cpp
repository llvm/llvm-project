// Test header and library paths when Clang is used with Android standalone
// toolchain.
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi -stdlib=libstdc++ \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck  %s
// CHECK: "-cc1"
// CHECK: "-internal-isystem" "{{.*}}/arm-linux-androideabi/include/c++/4.4.3"
// CHECK: "-internal-isystem" "{{.*}}/arm-linux-androideabi/include/c++/4.4.3/arm-linux-androideabi"
// CHECK: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.4.3"
// CHECK: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.4.3/../../../../arm-linux-androideabi/lib"
// CHECK: "-L{{.*}}/sysroot/usr/lib"
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=aarch64-linux-android -stdlib=libstdc++ \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-AARCH64 %s
// CHECK-AARCH64: "-cc1"
// CHECK-AARCH64: "-internal-isystem" "{{.*}}/aarch64-linux-android/include/c++/4.8"
// CHECK-AARCH64: "-internal-isystem" "{{.*}}/aarch64-linux-android/include/c++/4.8/aarch64-linux-android"
// CHECK-AARCH64: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-AARCH64: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-AARCH64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-AARCH64: "-L{{.*}}/lib/gcc/aarch64-linux-android/4.8"
// CHECK-AARCH64: "-L{{.*}}/lib/gcc/aarch64-linux-android/4.8/../../../../aarch64-linux-android/lib"
// CHECK-AARCH64: "-L{{.*}}/sysroot/usr/lib"
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm64-linux-android -stdlib=libstdc++ \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ARM64 %s
// CHECK-ARM64: "-cc1"
// CHECK-ARM64: "-internal-isystem" "{{.*}}/aarch64-linux-android/include/c++/4.8"
// CHECK-ARM64: "-internal-isystem" "{{.*}}/aarch64-linux-android/include/c++/4.8/aarch64-linux-android"
// CHECK-ARM64: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-ARM64: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-ARM64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ARM64: "-L{{.*}}/lib/gcc/aarch64-linux-android/4.8"
// CHECK-ARM64: "-L{{.*}}/lib/gcc/aarch64-linux-android/4.8/../../../../aarch64-linux-android/lib"
// CHECK-ARM64: "-L{{.*}}/sysroot/usr/lib"
