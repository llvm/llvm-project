// Test header and library paths when Clang is used with Android standalone
// toolchain.
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -stdlib=libstdc++ \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  %s
// CHECK: {{.*}}clang{{.*}}" "-cc1"
// CHECK: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK: "-internal-isystem" "{{.*}}/include/c++/4.9"
// CHECK-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7-a/thumb"
// CHECK-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7-a"
// CHECK-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/thumb"
// CHECK: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi"
// CHECK-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7-a/thumb"
// CHECK-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7-a"
// CHECK-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/thumb"
// CHECK: "-internal-isystem" "{{.*}}/include/c++/4.9/backward"
// CHECK: "-internal-isystem" "{{.*}}/sysroot/usr/local/include"
// CHECK: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
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
// CHECK: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/arm-linux-androideabi/lib"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a/thumb"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a"
// CHECK-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/thumb"
// CHECK: "-L{{.*}}/sysroot/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv7a-none-linux-androideabi -stdlib=libstdc++ \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  --check-prefix=CHECK-ARMV7 %s
// CHECK-ARMV7: {{.*}}clang{{.*}}" "-cc1"
// CHECK-ARMV7: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ARMV7: "-internal-isystem" "{{.*}}/include/c++/4.9"
// CHECK-ARMV7-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7-a/thumb"
// CHECK-ARMV7-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/thumb"
// CHECK-ARMV7-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi"
// CHECK-ARMV7: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7-a"
// CHECK-ARMV7-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7-a/thumb"
// CHECK-ARMV7-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/thumb"
// CHECK-ARMV7-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi"
// CHECK-ARMV7: "-internal-isystem" "{{.*}}/include/c++/4.9/backward"
// CHECK-ARMV7: "-internal-isystem" "{{.*}}/sysroot/usr/local/include"
// CHECK-ARMV7: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
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
// CHECK-ARMV7: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/thumb"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a/thumb"
// CHECK-ARMV7-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib"
// CHECK-ARMV7: "-L{{.*}}/sysroot/usr/lib"
//
// Other flags that can trigger armv7 mode.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -stdlib=libstdc++ \
// RUN:     -march=armv7 \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  --check-prefix=CHECK-ARMV7 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -stdlib=libstdc++ \
// RUN:     -march=armv7a \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  --check-prefix=CHECK-ARMV7 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -stdlib=libstdc++ \
// RUN:     -march=armv7-a \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  --check-prefix=CHECK-ARMV7 %s
//
// ARM thumb mode.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -stdlib=libstdc++ \
// RUN:     -mthumb \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  --check-prefix=CHECK-THUMB %s
// CHECK-THUMB: {{.*}}clang{{.*}}" "-cc1"
// CHECK-THUMB: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-THUMB: "-internal-isystem" "{{.*}}/include/c++/4.9"
// CHECK-THUMB-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7/thumb"
// CHECK-THUMB-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7"
// CHECK-THUMB-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi"
// CHECK-THUMB: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/thumb"
// CHECK-THUMB-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7/thumb"
// CHECK-THUMB-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7"
// CHECK-THUMB-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi"
// CHECK-THUMB: "-internal-isystem" "{{.*}}/include/c++/4.9/backward"
// CHECK-THUMB: "-internal-isystem" "{{.*}}/sysroot/usr/local/include"
// CHECK-THUMB: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
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
// CHECK-THUMB: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/thumb"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7/thumb"
// CHECK-THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib"
// CHECK-THUMB: "-L{{.*}}/sysroot/usr/lib"
//
// ARM V7 thumb mode.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -stdlib=libstdc++ \
// RUN:     -march=armv7-a -mthumb \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  --check-prefix=CHECK-ARMV7THUMB %s
// CHECK-ARMV7THUMB: {{.*}}clang{{.*}}" "-cc1"
// CHECK-ARMV7THUMB: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ARMV7THUMB: "-internal-isystem" "{{.*}}/include/c++/4.9"
// CHECK-ARMV7THUMB-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7-a"
// CHECK-ARMV7THUMB-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/thumb"
// CHECK-ARMV7THUMB-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi"
// CHECK-ARMV7THUMB: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7-a/thumb"
// CHECK-ARMV7THUMB-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/armv7-a"
// CHECK-ARMV7THUMB-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi/thumb"
// CHECK-ARMV7THUMB-NOT: "-internal-isystem" "{{.*}}/include/c++/4.9/arm-linux-androideabi"
// CHECK-ARMV7THUMB: "-internal-isystem" "{{.*}}/include/c++/4.9/backward"
// CHECK-ARMV7THUMB: "-internal-isystem" "{{.*}}/sysroot/usr/local/include"
// CHECK-ARMV7THUMB: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
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
// CHECK-ARMV7THUMB: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a/thumb"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/thumb"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib/armv7-a"
// CHECK-ARMV7THUMB-NOT: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.9/../{{[^ ]*}}/lib"
// CHECK-ARMV7THUMB: "-L{{.*}}/sysroot/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv7a-none-linux-androideabi -stdlib=libstdc++ \
// RUN:     -mthumb \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck  --check-prefix=CHECK-ARMV7THUMB %s
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target aarch64-linux-android -stdlib=libstdc++ \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-AARCH64 %s
// CHECK-AARCH64: {{.*}}clang{{.*}}" "-cc1"
// CHECK-AARCH64: "-internal-isystem" "{{.*}}/include/c++/4.9"
// CHECK-AARCH64: "-internal-isystem" "{{.*}}/include/c++/4.9/aarch64-linux-android"
// CHECK-AARCH64: "-internal-isystem" "{{.*}}/include/c++/4.9/backward"
// CHECK-AARCH64: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-AARCH64: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-AARCH64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-AARCH64: "-L{{.*}}/lib/gcc/aarch64-linux-android/4.9"
// CHECK-AARCH64: "-L{{.*}}/lib/gcc/aarch64-linux-android/4.9/../../../../aarch64-linux-android/lib"
// CHECK-AARCH64: "-L{{.*}}/sysroot/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm64-linux-android -stdlib=libstdc++ \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ARM64 %s
// CHECK-ARM64: {{.*}}clang{{.*}}" "-cc1"
// CHECK-ARM64: "-internal-isystem" "{{.*}}/include/c++/4.9"
// CHECK-ARM64: "-internal-isystem" "{{.*}}/include/c++/4.9/aarch64-linux-android"
// CHECK-ARM64: "-internal-isystem" "{{.*}}/include/c++/4.9/backward"
// CHECK-ARM64: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-ARM64: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-ARM64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ARM64: "-L{{.*}}/lib/gcc/aarch64-linux-android/4.9"
// CHECK-ARM64: "-L{{.*}}/lib/gcc/aarch64-linux-android/4.9/../../../../aarch64-linux-android/lib"
// CHECK-ARM64: "-L{{.*}}/sysroot/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-android \
// RUN:     -mips32 -stdlib=libstdc++ \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS %s
// CHECK-MIPS: {{.*}}clang{{.*}}" "-cc1"
// CHECK-MIPS: "-internal-isystem" "{{.*}}/include/c++/4.9"
// CHECK-MIPS: "-internal-isystem" "{{.*}}/include/c++/4.9/mipsel-linux-android"
// CHECK-MIPS: "-internal-isystem" "{{.*}}/include/c++/4.9/backward"
// CHECK-MIPS: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-MIPS: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-MIPS: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-MIPS: "-L{{.*}}/lib/gcc/mipsel-linux-android/4.9"
// CHECK-MIPS: "-L{{.*}}/lib/gcc/mipsel-linux-android/4.9/../../../../mipsel-linux-android/lib"
// CHECK-MIPS: "-L{{.*}}/sysroot/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-android \
// RUN:     -march=mips32 -mips32r2 -stdlib=libstdc++ \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-MIPSR2 %s
// CHECK-MIPSR2: {{.*}}clang{{.*}}" "-cc1"
// CHECK-MIPSR2: "-internal-isystem" "{{.*}}/include/c++/4.9"
// CHECK-MIPSR2: "-internal-isystem" "{{.*}}/include/c++/4.9/mipsel-linux-android/mips-r2"
// CHECK-MIPSR2: "-internal-isystem" "{{.*}}/include/c++/4.9/backward"
// CHECK-MIPSR2: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-MIPSR2: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-MIPSR2: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-MIPSR2: "-L{{.*}}/lib/gcc/mipsel-linux-android/4.9/mips-r2"
// CHECK-MIPSR2: "-L{{.*}}/lib/gcc/mipsel-linux-android/4.9/../../../../mipsel-linux-android/lib/../libr2"
// CHECK-MIPSR2: "-L{{.*}}/sysroot/usr/lib/../libr2"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-android \
// RUN:     -mips32r6 -stdlib=libstdc++ \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-MIPSR6 %s
// CHECK-MIPSR6: {{.*}}clang{{.*}}" "-cc1"
// CHECK-MIPSR6: "-internal-isystem" "{{.*}}/include/c++/4.9"
// CHECK-MIPSR6: "-internal-isystem" "{{.*}}/include/c++/4.9/mipsel-linux-android/mips-r6"
// CHECK-MIPSR6: "-internal-isystem" "{{.*}}/include/c++/4.9/backward"
// CHECK-MIPSR6: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-MIPSR6: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-MIPSR6: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-MIPSR6: "-L{{.*}}/lib/gcc/mipsel-linux-android/4.9/mips-r6"
// CHECK-MIPSR6: "-L{{.*}}/lib/gcc/mipsel-linux-android/4.9/../../../../mipsel-linux-android/lib/../libr6"
// CHECK-MIPSR6: "-L{{.*}}/sysroot/usr/lib/../libr6"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mips64el-linux-android \
// RUN:     -march=mips32 -mips32r2 -stdlib=libstdc++ \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64-R2 %s
// CHECK-MIPS64-R2: {{.*}}clang{{.*}}" "-cc1"
// CHECK-MIPS64-R2: "-internal-isystem" "{{.*}}/include/c++/4.9"
// CHECK-MIPS64-R2: "-internal-isystem" "{{.*}}/include/mips64el-linux-android/c++/4.9/mips-r2"
// CHECK-MIPS64-R2: "-internal-isystem" "{{.*}}/include/c++/4.9/backward"
// CHECK-MIPS64-R2: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-MIPS64-R2: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-MIPS64-R2: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-MIPS64-R2: "-L{{.*}}/lib/gcc/mips64el-linux-android/4.9/32/mips-r2"
// CHECK-MIPS64-R2: "-L{{.*}}/lib/gcc/mips64el-linux-android/4.9/../../../../mips64el-linux-android/lib/../libr2"
// CHECK-MIPS64-R2: "-L{{.*}}/sysroot/usr/lib/../libr2"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target mips64el-linux-android \
// RUN:     -march=mips32 -mips32r6 -stdlib=libstdc++ \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64-R6 %s
// CHECK-MIPS64-R6: {{.*}}clang{{.*}}" "-cc1"
// CHECK-MIPS64-R6: "-internal-isystem" "{{.*}}/include/c++/4.9"
// CHECK-MIPS64-R6: "-internal-isystem" "{{.*}}/include/mips64el-linux-android/c++/4.9/mips-r6"
// CHECK-MIPS64-R6: "-internal-isystem" "{{.*}}/include/c++/4.9/backward"
// CHECK-MIPS64-R6: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-MIPS64-R6: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-MIPS64-R6: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-MIPS64-R6: "-L{{.*}}/lib/gcc/mips64el-linux-android/4.9/32/mips-r6"
// CHECK-MIPS64-R6: "-L{{.*}}/lib/gcc/mips64el-linux-android/4.9/../../../../mips64el-linux-android/lib/../libr6"
// CHECK-MIPS64-R6: "-L{{.*}}/sysroot/usr/lib/../libr6"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target i686-linux-android \
// RUN:     -stdlib=libstdc++ \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-I686 %s
// CHECK-I686: {{.*}}clang{{.*}}" "-cc1"
// CHECK-I686: "-internal-isystem" "{{.*}}/include/c++/4.9"
// CHECK-I686: "-internal-isystem" "{{.*}}/include/c++/4.9/i686-linux-android"
// CHECK-I686: "-internal-isystem" "{{.*}}/include/c++/4.9/backward"
// CHECK-I686: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK-I686: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK-I686: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-I686: "-L{{.*}}/lib/gcc/i686-linux-android/4.9"
// CHECK-I686: "-L{{.*}}/lib/gcc/i686-linux-android/4.9/../../../../i686-linux-android/lib"
// CHECK-I686: "-L{{.*}}/sysroot/usr/lib"
