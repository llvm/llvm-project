// Test header and library paths when Clang is used with Android standalone
// toolchain.

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi21 \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  %s
// CHECK: "-cc1"
// CHECK: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK: "-internal-isystem" "{{.*}}/include/c++/v1"
// CHECK: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK: "-internal-isystem" "{{.*}}/sysroot/usr/local/include"
// CHECK: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include/arm-linux-androideabi"
// CHECK: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a/thumb"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/thumb"
// CHECK: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a/thumb"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/thumb"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a/thumb"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/thumb"
// CHECK: "-L{{.*}}/sysroot/usr/lib/arm-linux-androideabi/21"
// CHECK: "-L{{.*}}/sysroot/usr/lib/arm-linux-androideabi"
// CHECK: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/arm-linux-androideabi/lib"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a/thumb"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/thumb"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi14 \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-14 %s
// CHECK-14: "-L{{.*}}/sysroot/usr/lib/arm-linux-androideabi/14"
// CHECK-14: "-L{{.*}}/sysroot/usr/lib/arm-linux-androideabi"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi21 -stdlib=libstdc++ \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-STDCXX %s
// CHECK-STDCXX: "-cc1"
// CHECK-STDCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-STDCXX: "-internal-isystem" "{{.*}}/include/c++/4.9"
// CHECK-STDCXX-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7-a/thumb"
// CHECK-STDCXX-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7-a"
// CHECK-STDCXX-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/thumb"
// CHECK-STDCXX: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi"
// CHECK-STDCXX-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7-a/thumb"
// CHECK-STDCXX-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7-a"
// CHECK-STDCXX-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/thumb"
// CHECK-STDCXX: "-internal-isystem" "{{.*}}/include/c++/4.9/backward"
// CHECK-STDCXX: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-STDCXX-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a/thumb"
// CHECK-STDCXX-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a"
// CHECK-STDCXX-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/thumb"
// CHECK-STDCXX: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9"
// CHECK-STDCXX-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a/thumb"
// CHECK-STDCXX-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a"
// CHECK-STDCXX-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/thumb"
// CHECK-STDCXX-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a/thumb"
// CHECK-STDCXX-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a"
// CHECK-STDCXX-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/thumb"
// CHECK-STDCXX: "-L{{.*}}/sysroot/usr/lib/arm-linux-androideabi/21"
// CHECK-STDCXX: "-L{{.*}}/sysroot/usr/lib/arm-linux-androideabi"
// CHECK-STDCXX: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/arm-linux-androideabi/lib"
// CHECK-STDCXX-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a/thumb"
// CHECK-STDCXX-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a"
// CHECK-STDCXX-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/thumb"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=armv7a-none-linux-androideabi21 \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  --check-prefix=CHECK-ARMV7 %s
// CHECK-ARMV7: "-cc1"
// CHECK-ARMV7: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ARMV7: "-internal-isystem" "{{.*}}/include/c++/v1"
// CHECK-ARMV7: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK-ARMV7: "-internal-isystem" "{{.*}}/sysroot/usr/local/include"
// CHECK-ARMV7: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include/arm-linux-androideabi"
// CHECK-ARMV7: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-ARMV7: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-ARMV7: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a/thumb"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/thumb"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9"
// CHECK-ARMV7: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a/thumb"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/thumb"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/thumb"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a/thumb"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib"
// CHECK-ARMV7: "-L{{.*}}/sysroot/usr/lib/arm-linux-androideabi/21"
// CHECK-ARMV7: "-L{{.*}}/sysroot/usr/lib/arm-linux-androideabi"
// CHECK-ARMV7: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/thumb"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a/thumb"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib"

// Other flags that can trigger armv7 mode.
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi21 \
// RUN:     -march=armv7 \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  --check-prefix=CHECK-ARMV7 %s
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi21 \
// RUN:     -march=armv7a \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  --check-prefix=CHECK-ARMV7 %s
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi21 \
// RUN:     -march=armv7-a \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  --check-prefix=CHECK-ARMV7 %s

// ARM thumb mode.
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi21 \
// RUN:     -mthumb \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  --check-prefix=CHECK-THUMB %s
// CHECK-THUMB: "-cc1"
// CHECK-THUMB: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-THUMB: "-internal-isystem" "{{.*}}/include/c++/v1"
// CHECK-THUMB: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK-THUMB: "-internal-isystem" "{{.*}}/sysroot/usr/local/include"
// CHECK-THUMB: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include/arm-linux-androideabi"
// CHECK-THUMB: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-THUMB: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-THUMB: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a/thumb"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9"
// CHECK-THUMB: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/thumb"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a/thumb"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7/thumb"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib"
// CHECK-THUMB: "-L{{.*}}/sysroot/usr/lib/arm-linux-androideabi/21"
// CHECK-THUMB: "-L{{.*}}/sysroot/usr/lib/arm-linux-androideabi"
// CHECK-THUMB: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/thumb"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7/thumb"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib"
// CHECK-THUMB: "-L{{.*}}/sysroot/usr/lib"

// ARM V7 thumb mode.
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi21 \
// RUN:     -march=armv7-a -mthumb \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  --check-prefix=CHECK-ARMV7THUMB %s
// CHECK-ARMV7THUMB: "-cc1"
// CHECK-ARMV7THUMB: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ARMV7THUMB: "-internal-isystem" "{{.*}}/include/c++/v1"
// CHECK-ARMV7THUMB: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK-ARMV7THUMB: "-internal-isystem" "{{.*}}/sysroot/usr/local/include"
// CHECK-ARMV7THUMB: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include/arm-linux-androideabi"
// CHECK-ARMV7THUMB: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-ARMV7THUMB: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-ARMV7THUMB: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/thumb"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9"
// CHECK-ARMV7THUMB: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a/thumb"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/thumb"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/armv7-a"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/thumb"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib"
// CHECK-ARMV7THUMB: "-L{{.*}}/sysroot/usr/lib/arm-linux-androideabi/21"
// CHECK-ARMV7THUMB: "-L{{.*}}/sysroot/usr/lib/arm-linux-androideabi"
// CHECK-ARMV7THUMB: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a/thumb"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/thumb"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm-linux-androideabi21 \
// RUN:     -march=armv7-a -mthumb \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:     -print-multi-lib \
// RUN:   | FileCheck  --check-prefix=CHECK-ARM-MULTILIBS %s

// CHECK-ARM-MULTILIBS:      thumb;@mthumb
// CHECK-ARM-MULTILIBS-NEXT: armv7-a;@march=armv7-a
// CHECK-ARM-MULTILIBS-NEXT: armv7-a/thumb;@march=armv7-a@mthumb
// CHECK-ARM-MULTILIBS-NEXT: .;


// RUN: %clang -### %s 2>&1 \
// RUN:     --target=armv7a-none-linux-androideabi21 \
// RUN:     -mthumb \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  --check-prefix=CHECK-ARMV7THUMB %s

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=aarch64-linux-android21 \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-AARCH64 %s
// CHECK-AARCH64: "-cc1"
// CHECK-AARCH64: "-internal-isystem" "{{.*}}/include/c++/v1"
// CHECK-AARCH64: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include/aarch64-linux-android"
// CHECK-AARCH64: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-AARCH64: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-AARCH64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-AARCH64: "-L{{.*}}/lib/gcc/aarch64-linux-android/4.9"
// CHECK-AARCH64: "-L{{.*}}/sysroot/usr/lib/aarch64-linux-android/21"
// CHECK-AARCH64: "-L{{.*}}/sysroot/usr/lib/aarch64-linux-android"
// CHECK-AARCH64: "-L{{.*}}/lib/gcc/aarch64-linux-android/4.9/../../../../aarch64-linux-android/lib"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=arm64-linux-android21 \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ARM64 %s
// CHECK-ARM64: "-cc1"
// CHECK-ARM64: "-internal-isystem" "{{.*}}/include/c++/v1"
// CHECK-ARM64: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include/aarch64-linux-android"
// CHECK-ARM64: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-ARM64: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-ARM64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ARM64: "-L{{.*}}/lib/gcc/aarch64-linux-android/4.9"
// CHECK-ARM64: "-L{{.*}}/sysroot/usr/lib/aarch64-linux-android/21"
// CHECK-ARM64: "-L{{.*}}/sysroot/usr/lib/aarch64-linux-android"
// CHECK-ARM64: "-L{{.*}}/lib/gcc/aarch64-linux-android/4.9/../../../../aarch64-linux-android/lib"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=mipsel-linux-android21 \
// RUN:     -mips32 \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS %s
// CHECK-MIPS: "-cc1"
// CHECK-MIPS: "-internal-isystem" "{{.*}}/include/c++/v1"
// CHECK-MIPS: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-MIPS: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-MIPS: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-MIPS: "-L{{.*}}/lib/gcc/mipsel-linux-android/4.9"
// CHECK-MIPS: "-L{{.*}}/sysroot/usr/lib/mipsel-linux-android/21"
// CHECK-MIPS: "-L{{.*}}/sysroot/usr/lib/mipsel-linux-android"
// CHECK-MIPS: "-L{{.*}}/lib/gcc/mipsel-linux-android/4.9/../../../../mipsel-linux-android/lib"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=i686-linux-android21 \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-I686 %s
// CHECK-I686: "-cc1"
// CHECK-I686: "-internal-isystem" "{{.*}}/include/c++/v1"
// CHECK-I686: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include/i686-linux-android"
// CHECK-I686: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-I686: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-I686: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-I686: "-L{{.*}}/lib/gcc/i686-linux-android/4.9"
// CHECK-I686: "-L{{.*}}/sysroot/usr/lib/i686-linux-android/21"
// CHECK-I686: "-L{{.*}}/sysroot/usr/lib/i686-linux-android"
// CHECK-I686: "-L{{.*}}/lib/gcc/i686-linux-android/4.9/../../../../i686-linux-android/lib"

// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-linux-android21 \
// RUN:     --gcc-toolchain=%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-X86_64 %s
// CHECK-X86_64: "-cc1"
// CHECK-X86_64: "-internal-isystem" "{{.*}}/include/c++/v1"
// CHECK-X86_64: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include/x86_64-linux-android"
// CHECK-X86_64: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-X86_64: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-X86_64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-X86_64: "-L{{.*}}/lib/gcc/x86_64-linux-android/4.9"
// CHECK-X86_64: "-L{{.*}}/sysroot/usr/lib/x86_64-linux-android/21"
// CHECK-X86_64: "-L{{.*}}/sysroot/usr/lib/x86_64-linux-android"
// CHECK-X86_64: "-L{{.*}}/lib/gcc/x86_64-linux-android/4.9/../../../../x86_64-linux-android/lib"

// We need two sets of tests to verify that we both don't find non-Android
// toolchains installations and that we *do* find Android toolchains. We can't
// do both at the same time in this environment because we need to pass
// --sysroot to find the toolchains which would override searching in /usr. In a
// production environment --sysroot is not used and the toolchains are instead
// found relative to the clang binary, so both would be considered.

// RUN: %clang -v --target=i686-linux-android \
// RUN:     2>&1 | FileCheck --check-prefix=CHECK-I686-GCC-NOSYS %s

// CHECK-I686-GCC-NOSYS-NOT: Found candidate GCC installation: /usr{{.*}}

// RUN: %clang -v --target=i686-linux-android \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree \
// RUN:     2>&1 | FileCheck --check-prefix=CHECK-I686-GCC %s

// CHECK-I686-GCC-NOT: Found candidate GCC installation: /usr{{.*}}
// CHECK-I686-GCC: Found candidate GCC installation: {{.*}}i686-linux-android{{[/\\]}}4.9
// CHECK-I686-GCC-NEXT: Found candidate GCC installation: {{.*}}x86_64-linux-android{{[/\\]}}4.9
// CHECK-I686-GCC-NEXT: Selected GCC installation: {{.*}}i686-linux-android{{[/\\]}}4.9

// RUN: %clang -v --target=x86_64-linux-android \
// RUN:     2>&1 | FileCheck --check-prefix=CHECK-X86_64-GCC-NOSYS %s

// CHECK-X86_64-GCC-NOSYS-NOT: Found candidate GCC installation: /usr{{.*}}

// RUN: %clang -v --target=x86_64-linux-android \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree \
// RUN:     2>&1 | FileCheck --check-prefix=CHECK-X86_64-GCC %s

// CHECK-X86_64-GCC-NOT: Found candidate GCC installation: /usr{{.*}}
// CHECK-X86_64-GCC: Found candidate GCC installation: {{.*}}i686-linux-android{{[/\\]}}4.9
// CHECK-X86_64-GCC-NEXT: Found candidate GCC installation: {{.*}}x86_64-linux-android{{[/\\]}}4.9
// CHECK-X86_64-GCC-NEXT: Selected GCC installation: {{.*}}x86_64-linux-android{{[/\\]}}4.9
