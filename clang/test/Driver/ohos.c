// RUN: %clang %s -### -no-canonical-prefixes --target=arm-liteos \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     --sysroot=%S/Inputs/ohos_native_tree/sysroot -fuse-ld=ld -march=armv7-a 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-ARM %s
// RUN: %clang %s -### -no-canonical-prefixes --target=arm-liteos \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     --sysroot=%S/Inputs/ohos_native_tree/sysroot -fuse-ld=ld -march=armv7-a -mcpu=cortex-a7 -mfloat-abi=soft 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-ARM-A7-SOFT %s
// CHECK: {{.*}}clang{{.*}}" "-cc1"
// CHECK-NOT: "--mrelax-relocations"
// CHECK-NOT: "-munwind-tables"
// CHECK: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK: "-internal-externc-isystem" "[[SYSROOT]]{{/|\\\\}}include"
// CHECK-NOT: "-fsanitize=safe-stack"
// CHECK-NOT: "-stack-protector" "2"
// CHECK-NOT: "-fno-common"
// CHECK: {{.*}}ld.lld{{.*}}" "--sysroot=[[SYSROOT]]"
// CHECK-NOT: "--sysroot=[[SYSROOT]]"
// CHECK-NOT: "--build-id"
// CHECK: "--hash-style=both"
// CHECK: "-pie"
// CHECK: "-dynamic-linker" "/lib/ld-musl-arm.so.1"
// CHECK: Scrt1.o
// CHECK: crti.o
// CHECK: clang_rt.crtbegin.o
// CHECK-ARM: "-L[[SYSROOT]]{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}"
// CHECK-ARM-A7-SOFT: "-L[[SYSROOT]]{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_soft"
// CHECK-ARM: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-ARM-A7-SOFT: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos/a7_soft{{/|\\\\}}libclang_rt.builtins.a"
// CHECK: "-lc"
// CHECK: clang_rt.crtend.o
// CHECK: crtn.o

// RUN: not %clang %s -### --target=arm-liteos -rtlib=libgcc 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-RTLIB
// CHECK-RTLIB: error: invalid runtime library name in argument '-rtlib=libgcc'

// RUN: %clang %s -### --target=arm-liteos -static -fuse-ld=ld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-STATIC
// CHECK-STATIC: "-static"
// CHECK-STATIC-NOT: "-Bdynamic"
// CHECK-STATIC: "-l:libunwind.a"
// CHECK-STATIC: "-lc"

// RUN: %clang %s -### --target=arm-liteos -shared -fuse-ld=ld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-SHARED
// CHECK-SHARED-NOT: "-pie"
// CHECK-SHARED: "-shared"
// CHECK-SHARED: "-lc"
// CHECK-SHARED: "-l:libunwind.a"

// RUN: %clang %s -### --target=arm-linux-ohos -fuse-ld=ld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-RUNTIME
// RUN: %clang %s -### --target=aarch64-linux-ohos -fuse-ld=ld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-RUNTIME
// RUN: %clang %s -### --target=mipsel-linux-ohos -fuse-ld=ld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-RUNTIME
// RUN: %clang %s -### --target=x86_64-linux-ohos -fuse-ld=ld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-RUNTIME
// CHECK-RUNTIME: "{{.*}}libclang_rt.builtins.a"
// CHECK-RUNTIME: "-l:libunwind.a"
// CHECK-LIBM: "-lm"

// RUN: %clang %s -### --target=arm-liteos -r -fuse-ld=ld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-RELOCATABLE
// CHECK-RELOCATABLE-NOT: "-pie"
// CHECK-RELOCATABLE: "-r"

// RUN: %clang %s -### --target=arm-liteos -nodefaultlibs -fuse-ld=ld 2>&1 \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     | FileCheck %s -check-prefix=CHECK-NODEFAULTLIBS
// CHECK-NODEFAULTLIBS: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-NODEFAULTLIBS-NOT: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-NODEFAULTLIBS-NOT: "-lc"

// RUN: %clang %s -### --target=arm-liteos -nostdlib -fuse-ld=ld 2>&1 \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     | FileCheck %s -check-prefix=CHECK-NOSTDLIB
// CHECK-NOSTDLIB: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-NOSTDLIB-NOT: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-NOSTDLIB-NOT: "-lc"

// RUN: %clang %s -### --target=arm-liteos -nolibc -fuse-ld=ld 2>&1 \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     | FileCheck %s -check-prefix=CHECK-NOLIBC
// CHECK-NOLIBC: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-NOLIBC: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-NOLIBC-NOT: "-lc"

// RUN: %clang %s -### --target=arm-liteos \
// RUN:     -fsanitize=safe-stack 2>&1 \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     -fuse-ld=ld \
// RUN:     | FileCheck %s -check-prefix=CHECK-SAFESTACK
// CHECK-SAFESTACK: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-SAFESTACK: "-fsanitize=safe-stack"
// CHECK-SAFESTACK: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.safestack.a"
// CHECK-SAFESTACK: "__safestack_init"

// RUN: %clang %s -### --target=arm-liteos \
// RUN:     -fsanitize=address 2>&1 \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     -fuse-ld=ld \
// RUN:     | FileCheck %s -check-prefix=CHECK-ASAN-ARM
// CHECK-ASAN-ARM: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ASAN-ARM: "-fsanitize=address"
// CHECK-ASAN-ARM: "-fsanitize-address-use-after-scope"
// CHECK-ASAN-ARM: "-dynamic-linker" "/lib/ld-musl-arm.so.1"
// CHECK-ASAN-ARM: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.asan.a"
// CHECK-ASAN-ARM-NOT: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.asan-preinit.a"

// RUN: %clang %s -### --target=arm-liteos \
// RUN:     -fsanitize=address -fPIC -shared 2>&1 \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     -shared-libsan \
// RUN:     -fuse-ld=ld \
// RUN:     | FileCheck %s -check-prefix=CHECK-ASAN-SHARED
// CHECK-ASAN-SHARED: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ASAN-SHARED: "-fsanitize=address"
// CHECK-ASAN-SHARED: "-fsanitize-address-use-after-scope"
// CHECK-ASAN-SHARED: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.asan.so"
// CHECK-ASAN-SHARED-NOT: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.asan-preinit.a"

// RUN: %clang %s -### --target=arm-liteos \
// RUN:     -fsanitize=fuzzer 2>&1 \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     -fuse-ld=ld \
// RUN:     | FileCheck %s -check-prefix=CHECK-FUZZER-ARM
// CHECK-FUZZER-ARM: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-FUZZER-ARM: "-fsanitize=fuzzer,fuzzer-no-link"
// CHECK-FUZZER-ARM: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.fuzzer.a"

// RUN: %clang %s -### --target=arm-liteos \
// RUN:     -fsanitize=scudo 2>&1 \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     -fuse-ld=ld \
// RUN:     | FileCheck %s -check-prefix=CHECK-SCUDO-ARM
// CHECK-SCUDO-ARM: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-SCUDO-ARM: "-fsanitize=scudo"
// CHECK-SCUDO-ARM: "-pie"
// CHECK-SCUDO-ARM: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.scudo_standalone.a"

// RUN: %clang %s -### --target=arm-liteos \
// RUN:     -fsanitize=scudo -fPIC -shared 2>&1 \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     -shared-libsan \
// RUN:     -fuse-ld=ld \
// RUN:     | FileCheck %s -check-prefix=CHECK-SCUDO-SHARED
// CHECK-SCUDO-SHARED: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-SCUDO-SHARED: "-fsanitize=scudo"
// CHECK-SCUDO-SHARED: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.scudo_standalone.so"

// RUN: %clang %s -### --target=arm-liteos \
// RUN:     -fxray-instrument -fxray-modes=xray-basic \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     -fuse-ld=ld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-XRAY-ARM
// CHECK-XRAY-ARM: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-XRAY-ARM: "-fxray-instrument"
// CHECK-XRAY-ARM: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.xray.a"
// CHECK-XRAY-ARM: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.xray-basic.a"

// RUN: %clang %s -### --target=arm-liteos \
// RUN:     -O3 -flto -mcpu=cortex-a53 2>&1 \
// RUN:     -fuse-ld=ld \
// RUN:     | FileCheck %s -check-prefix=CHECK-LTO
// CHECK-LTO: "-plugin-opt=mcpu=cortex-a53"
// CHECK-LTO: "-plugin-opt=O3"

// RUN: %clang %s -### --target=arm-liteos \
// RUN:     -flto=thin -flto-jobs=8 -mcpu=cortex-a7 2>&1 \
// RUN:     -fuse-ld=ld \
// RUN:     | FileCheck %s -check-prefix=CHECK-THINLTO
// CHECK-THINLTO: "-plugin-opt=mcpu=cortex-a7"
// CHECK-THINLTO: "-plugin-opt=thinlto"
// CHECK-THINLTO: "-plugin-opt=jobs=8"

// RUN: %clang %s -### --target=arm-liteos \
// RUN:     -ccc-install-dir %S/Inputs/ohos_native_tree/llvm/bin \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     --sysroot=%S/Inputs/ohos_native_tree/sysroot \
// RUN:     -march=armv7-a -mfloat-abi=soft 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB,CHECK-MULTILIB-SF,CHECK-MULTILIB-ARM
// RUN: %clang %s -### --target=arm-liteos \
// RUN:     -ccc-install-dir %S/Inputs/ohos_native_tree/llvm/bin \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     --sysroot=%S/Inputs/ohos_native_tree/sysroot \
// RUN:     -march=armv7-a -mcpu=cortex-a7 -mfloat-abi=soft 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB,CHECK-MULTILIB-SF,CHECK-MULTILIB-ARM-A7-SOFT
// RUN: %clang %s -### --target=arm-liteos  \
// RUN:     -ccc-install-dir %S/Inputs/ohos_native_tree/llvm/bin \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     --sysroot=%S/Inputs/ohos_native_tree/sysroot \
// RUN:     -march=armv7-a -mcpu=cortex-a7 -mfloat-abi=softfp -mfpu=neon-vfpv4 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB,CHECK-MULTILIB-SF,CHECK-MULTILIB-ARM-A7-SOFTFP
// RUN: %clang %s -### --target=arm-liteos  \
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

// CHECK-MULTILIB-ARM: "-L[[SYSROOT]]{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}"

// CHECK-MULTILIB-ARM-A7-SOFT: "-L[[SYSROOT]]{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_soft"

// CHECK-MULTILIB-ARM-A7-SOFTFP: "-L[[SYSROOT]]{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_softfp_neon-vfpv4"

// CHECK-MULTILIB-ARM-A7-HARD: "-L[[SYSROOT]]{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_hard_neon-vfpv4"

// CHECK-MULTILIB-ARM: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-MULTILIB-ARM-A7-SOFT: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_soft{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-MULTILIB-ARM-A7-SOFTFP: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_softfp_neon-vfpv4{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-MULTILIB-ARM-A7-HARD: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}arm-liteos-ohos{{/|\\\\}}a7_hard_neon-vfpv4{{/|\\\\}}libclang_rt.builtins.a"

// RUN: %clang %s -### -no-canonical-prefixes --target=arm-linux-ohos -fprofile-instr-generate -v \
// RUN:     -resource-dir=%S/Inputs/ohos_native_tree/llvm/lib/clang/x.y.z \
// RUN:     --sysroot=%S/Inputs/ohos_native_tree/sysroot -fuse-ld=ld -march=armv7-a 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK-PROFILE-RTLIB %s

// CHECK-PROFILE-RTLIB: -u__llvm_profile_runtime
// CHECK-PROFILE-RTLIB: libclang_rt.profile

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-ohos -pthread \
// RUN:     --sysroot=%S/Inputs/ohos_native_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-OHOS-PTHREAD %s

// CHECK-OHOS-PTHREAD-NOT: -lpthread

