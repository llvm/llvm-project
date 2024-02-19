// RUN: %clangxx %s -### -no-canonical-prefixes --target=arm-liteos -march=armv7-a \
// RUN:     -ccc-install-dir %S/Inputs/ohos_native_tree/llvm/bin \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     --sysroot=%S/Inputs/ohos_native_tree/sysroot -fuse-ld=ld 2>&1 | FileCheck %s
// CHECK: {{.*}}clang{{.*}}" "-cc1"
// CHECK: "-triple" "armv7-unknown-liteos-ohos"
// CHECK-NOT: "-fuse-init-array"
// CHECK: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK: "-internal-externc-isystem" "[[SYSROOT]]{{/|\\\\}}include"
// CHECK: {{.*}}ld.lld{{.*}}" "--sysroot=[[SYSROOT]]"
// CHECK: "-pie"
// CHECK: "-dynamic-linker" "/lib/ld-musl-arm.so.1"
// CHECK: Scrt1.o
// CHECK: crti.o
// CHECK: clang_rt.crtbegin.o
// CHECK: "-L{{.*[/\\]}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}"
// CHECK-NOT: "--push-state"
// CHECK-NOT: "--as-needed"
// CHECK: "-lc++"
// CHECK: "-lm"
// CHECK-NOT: "--pop-state"
// CHECK: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.builtins.a"
// CHECK: "-lc"
// CHECK: clang_rt.crtend.o
// CHECK: crtn.o

// RUN: not %clangxx %s -### --target=arm-unknown-liteos -stdlib=libstdc++ 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-STDLIB
// CHECK-STDLIB: error: invalid library name in argument '-stdlib=libstdc++'

// RUN: %clangxx %s -### --target=arm-unknown-liteos -static-libstdc++ \
// RUN:     -fuse-ld=ld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-STATIC
// CHECK-STATIC-NOT: "--push-state"
// CHECK-STATIC-NOT: "--as-needed"
// CHECK-STATIC: "-Bstatic"
// CHECK-STATIC: "-lc++"
// CHECK-STATIC: "-Bdynamic"
// CHECK-STATIC: "-lm"
// CHECK-STATIC-NOT: "--pop-state"
// CHECK-STATIC: "-lc"

// RUN: %clangxx %s -### --target=arm-unknown-liteos -static \
// RUN:     -fuse-ld=ld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-STATIC1
// CHECK-STATIC1-NOT: "-fuse-init-array"
// CHECK-STATIC1: "-static"
// CHECK-STATIC1: "-lc++"
// CHECK-STATIC1: "-lc++abi"
// CHECK-STATIC1: "-lunwind"
// CHECK-STATIC1: "-lm"
// CHECK-STATIC1: "-lc"

// RUN: %clangxx %s -### --target=arm-unknown-liteos -march=armv7-a -mfloat-abi=soft -static -fPIE -fPIC -fpic -pie \
// RUN:     --sysroot=%S/Inputs/ohos_native_tree/sysroot -fuse-ld=ld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-STATIC2
// CHECK-STATIC2: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-STATIC2: {{.*}}ld.lld{{.*}}" "--sysroot=[[SYSROOT]]"
// CHECK-STATIC2: "-static"
// CHECK-STATIC2: "-lc++"
// CHECK-STATIC2: "-lc++abi"
// CHECK-STATIC2: "-lunwind"
// CHECK-STATIC2: "-lm"
// CHECK-STATIC2: "-lc"

// RUN: %clangxx %s -### --target=arm-liteos -nostdlib++ -fuse-ld=ld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-NOSTDLIBXX
// CHECK-NOSTDLIBXX-NOT: "-lc++"
// CHECK-NOSTDLIBXX: "-lm"
// CHECK-NOSTDLIBXX: "-lc"

// RUN: %clangxx %s -### --target=arm-liteos \
// RUN:     -ccc-install-dir %S/Inputs/ohos_native_tree/llvm/bin \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     --sysroot=%S/Inputs/ohos_native_tree/sysroot \
// RUN:     -march=armv7-a -mfloat-abi=soft 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB,CHECK-MULTILIB-SF,CHECK-MULTILIB-ARM
// RUN: %clangxx %s -### --target=arm-liteos \
// RUN:     -ccc-install-dir %S/Inputs/ohos_native_tree/llvm/bin \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     --sysroot=%S/Inputs/ohos_native_tree/sysroot \
// RUN:     -march=armv7-a -mcpu=cortex-a7 -mfloat-abi=soft 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB,CHECK-MULTILIB-SF,CHECK-MULTILIB-ARM-A7-SOFT
// RUN: %clangxx %s -### --target=arm-liteos  \
// RUN:     -ccc-install-dir %S/Inputs/ohos_native_tree/llvm/bin \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     --sysroot=%S/Inputs/ohos_native_tree/sysroot \
// RUN:     -march=armv7-a -mcpu=cortex-a7 -mfloat-abi=softfp -mfpu=neon-vfpv4 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB,CHECK-MULTILIB-SF,CHECK-MULTILIB-ARM-A7-SOFTFP
// RUN: %clangxx %s -### --target=arm-liteos  \
// RUN:     -ccc-install-dir %S/Inputs/ohos_native_tree/llvm/bin \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     --sysroot=%S/Inputs/ohos_native_tree/sysroot \
// RUN:     -march=armv7-a -mcpu=cortex-a7 -mfloat-abi=hard -mfpu=neon-vfpv4 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB,CHECK-MULTILIB-HF,CHECK-MULTILIB-ARM-A7-HARD
// CHECK-MULTILIB: {{.*}}clang{{.*}}" "-cc1"
// CHECK-MULTILIB: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-MULTILIB: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-MULTILIB: {{.*}}ld.lld{{.*}}" "--sysroot=[[SYSROOT]]"
// CHECK-MULTILIB-SF: "-dynamic-linker" "/lib/ld-musl-arm.so.1"
// CHECK-MULTILIB-HF: "-dynamic-linker" "/lib/ld-musl-armhf.so.1"

// CHECK-MULTILIB-ARM: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}"
// CHECK-MULTILIB-ARM: "-L[[SYSROOT]]{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}"

// CHECK-MULTILIB-ARM-A7-SOFT: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_soft"
// CHECK-MULTILIB-ARM-A7-SOFT: "-L[[SYSROOT]]{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_soft"

// CHECK-MULTILIB-ARM-A7-SOFTFP: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_softfp_neon-vfpv4"
// CHECK-MULTILIB-ARM-A7-SOFTFP: "-L[[SYSROOT]]{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_softfp_neon-vfpv4"

// CHECK-MULTILIB-ARM-A7-HARD: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_hard_neon-vfpv4"
// CHECK-MULTILIB-ARM-A7-HARD: "-L[[SYSROOT]]{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_hard_neon-vfpv4"

// CHECK-MULTILIB-ARM: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-MULTILIB-ARM-A7-SOFT: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_soft{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-MULTILIB-ARM-A7-SOFTFP: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_softfp_neon-vfpv4{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-MULTILIB-ARM-A7-HARD: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_hard_neon-vfpv4{{/|\\\\}}libclang_rt.builtins.a"
