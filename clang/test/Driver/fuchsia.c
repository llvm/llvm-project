// RUN: %clang %s -### -no-canonical-prefixes --target=x86_64-fuchsia \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --sysroot=%S/platform -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-X86_64 %s
// RUN: %clang %s -### -no-canonical-prefixes --target=aarch64-fuchsia \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --sysroot=%S/platform -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-AARCH64 %s
// CHECK: {{.*}}clang{{.*}}" "-cc1"
// CHECK: "--mrelax-relocations"
// CHECK: "-munwind-tables"
// CHECK: "-fuse-init-array"
// CHECK: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK: "-internal-externc-isystem" "[[SYSROOT]]{{/|\\\\}}include"
// CHECK: "-fsanitize=safe-stack"
// CHECK: "-stack-protector" "2"
// CHECK: "-fno-common"
// CHECK: {{.*}}ld.lld{{.*}}" "-z" "rodynamic"
// CHECK: "--sysroot=[[SYSROOT]]"
// CHECK: "-pie"
// CHECK: "--build-id"
// CHECK: "--hash-style=gnu"
// CHECK: "-dynamic-linker" "ld.so.1"
// CHECK: Scrt1.o
// CHECK-NOT: crti.o
// CHECK-NOT: crtbegin.o
// CHECK: "-L[[SYSROOT]]{{/|\\\\}}lib"
// CHECK-X86_64: "[[RESOURCE_DIR]]{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-AARCH64: "[[RESOURCE_DIR]]{{/|\\\\}}aarch64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.builtins.a"
// CHECK: "-lc"
// CHECK-NOT: crtend.o
// CHECK-NOT: crtn.o

// RUN: %clang %s -### --target=x86_64-fuchsia -rtlib=libgcc -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-RTLIB
// CHECK-RTLIB: error: invalid runtime library name in argument '-rtlib=libgcc'

// RUN: %clang %s -### --target=x86_64-fuchsia -static -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-STATIC
// CHECK-STATIC: "-Bstatic"
// CHECK-STATIC: "-Bdynamic"
// CHECK-STATIC: "-lc"

// RUN: %clang %s -### --target=x86_64-fuchsia -shared -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-SHARED
// CHECK-SHARED-NOT: "-pie"
// CHECK-SHARED: "-shared"

// RUN: %clang %s -### --target=x86_64-fuchsia -r -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-RELOCATABLE
// CHECK-RELOCATABLE-NOT: "-pie"
// CHECK-RELOCATABLE-NOT: "--build-id"
// CHECK-RELOCATABLE: "-r"

// RUN: %clang %s -### --target=x86_64-fuchsia \
// RUN:     -fsanitize=safe-stack 2>&1 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld \
// RUN:     | FileCheck %s -check-prefix=CHECK-SAFESTACK
// CHECK-SAFESTACK: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-SAFESTACK: "-fsanitize=safe-stack"
// CHECK-SAFESTACK-NOT: "[[RESOURCE_DIR]]{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.safestack.a"
// CHECK-SAFESTACK-NOT: "__safestack_init"

// RUN: %clang %s -### --target=x86_64-fuchsia \
// RUN:     -fsanitize=address 2>&1 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld \
// RUN:     | FileCheck %s -check-prefix=CHECK-ASAN-X86
// CHECK-ASAN-X86: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ASAN-X86: "-fsanitize=address"
// CHECK-ASAN-X86: "-fsanitize-address-globals-dead-stripping"
// CHECK-ASAN-X86: "-dynamic-linker" "asan/ld.so.1"
// CHECK-ASAN-X86: "-L[[RESOURCE_DIR]]{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}lib{{/|\\\\}}asan"
// CHECK-ASAN-X86: "-L[[RESOURCE_DIR]]{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}lib"
// CHECK-ASAN-X86: "[[RESOURCE_DIR]]{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.asan.so"
// CHECK-ASAN-X86: "[[RESOURCE_DIR]]{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.asan-preinit.a"

// RUN: %clang %s -### --target=aarch64-fuchsia \
// RUN:     -fsanitize=address 2>&1 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld \
// RUN:     | FileCheck %s -check-prefix=CHECK-ASAN-AARCH64
// CHECK-ASAN-AARCH64: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ASAN-AARCH64: "-fsanitize=address"
// CHECK-ASAN-AARCH64: "-fsanitize-address-globals-dead-stripping"
// CHECK-ASAN-AARCH64: "-dynamic-linker" "asan/ld.so.1"
// CHECK-ASAN-AARCH64: "-L[[RESOURCE_DIR]]{{/|\\\\}}aarch64-fuchsia{{/|\\\\}}lib{{/|\\\\}}asan"
// CHECK-ASAN-AARCH64: "-L[[RESOURCE_DIR]]{{/|\\\\}}aarch64-fuchsia{{/|\\\\}}lib"
// CHECK-ASAN-AARCH64: "[[RESOURCE_DIR]]{{/|\\\\}}aarch64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.asan.so"
// CHECK-ASAN-AARCH64: "[[RESOURCE_DIR]]{{/|\\\\}}aarch64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.asan-preinit.a"

// RUN: %clang %s -### --target=x86_64-fuchsia \
// RUN:     -fsanitize=address -fPIC -shared 2>&1 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld \
// RUN:     | FileCheck %s -check-prefix=CHECK-ASAN-SHARED
// CHECK-ASAN-SHARED: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-ASAN-SHARED: "-fsanitize=address"
// CHECK-ASAN-SHARED: "-fsanitize-address-globals-dead-stripping"
// CHECK-ASAN-SHARED: "[[RESOURCE_DIR]]{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.asan.so"
// CHECK-ASAN-SHARED-NOT: "[[RESOURCE_DIR]]{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.asan-preinit.a"

// RUN: %clang %s -### --target=x86_64-fuchsia \
// RUN:     -fsanitize=fuzzer 2>&1 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld \
// RUN:     | FileCheck %s -check-prefix=CHECK-FUZZER-X86
// CHECK-FUZZER-X86: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-FUZZER-X86: "-fsanitize=fuzzer,fuzzer-no-link,safe-stack"
// CHECK-FUZZER-X86: "[[RESOURCE_DIR]]{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.fuzzer.a"

// RUN: %clang %s -### --target=aarch64-fuchsia \
// RUN:     -fsanitize=fuzzer 2>&1 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld \
// RUN:     | FileCheck %s -check-prefix=CHECK-FUZZER-AARCH64
// CHECK-FUZZER-AARCH64: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-FUZZER-AARCH64: "-fsanitize=fuzzer,fuzzer-no-link,safe-stack"
// CHECK-FUZZER-AARCH64: "[[RESOURCE_DIR]]{{/|\\\\}}aarch64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.fuzzer.a"

// RUN: %clang %s -### --target=x86_64-fuchsia \
// RUN:     -fsanitize=scudo 2>&1 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld \
// RUN:     | FileCheck %s -check-prefix=CHECK-SCUDO-X86
// CHECK-SCUDO-X86: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-SCUDO-X86: "-fsanitize=safe-stack,scudo"
// CHECK-SCUDO-X86: "-pie"
// CHECK-SCUDO-X86: "[[RESOURCE_DIR]]{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.scudo.so"

// RUN: %clang %s -### --target=aarch64-fuchsia \
// RUN:     -fsanitize=scudo 2>&1 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld \
// RUN:     | FileCheck %s -check-prefix=CHECK-SCUDO-AARCH64
// CHECK-SCUDO-AARCH64: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-SCUDO-AARCH64: "-fsanitize=safe-stack,scudo"
// CHECK-SCUDO-AARCH64: "-pie"
// CHECK-SCUDO-AARCH64: "[[RESOURCE_DIR]]{{/|\\\\}}aarch64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.scudo.so"

// RUN: %clang %s -### --target=x86_64-fuchsia \
// RUN:     -fsanitize=scudo -fPIC -shared 2>&1 \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld \
// RUN:     | FileCheck %s -check-prefix=CHECK-SCUDO-SHARED
// CHECK-SCUDO-SHARED: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-SCUDO-SHARED: "-fsanitize=safe-stack,scudo"
// CHECK-SCUDO-SHARED: "[[RESOURCE_DIR]]{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.scudo.so"

// RUN: %clang %s -### --target=x86_64-fuchsia \
// RUN:     -fxray-instrument -fxray-modes=xray-basic \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-XRAY-X86
// CHECK-XRAY-X86: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-XRAY-X86: "-fxray-instrument"
// CHECK-XRAY-X86: "[[RESOURCE_DIR]]{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.xray.a"
// CHECK-XRAY-X86: "[[RESOURCE_DIR]]{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.xray-basic.a"

// RUN: %clang %s -### --target=aarch64-fuchsia \
// RUN:     -fxray-instrument -fxray-modes=xray-basic \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-XRAY-AARCH64
// CHECK-XRAY-AARCH64: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-XRAY-AARCH64: "-fxray-instrument"
// CHECK-XRAY-AARCH64: "[[RESOURCE_DIR]]{{/|\\\\}}aarch64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.xray.a"
// CHECK-XRAY-AARCH64: "[[RESOURCE_DIR]]{{/|\\\\}}aarch64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.xray-basic.a"

// RUN: %clang %s -### --target=aarch64-fuchsia \
// RUN:     -O3 -flto -mcpu=cortex-a53 2>&1 \
// RUN:     -fuse-ld=lld \
// RUN:     | FileCheck %s -check-prefix=CHECK-LTO
// CHECK-LTO: "-plugin-opt=mcpu=cortex-a53"
// CHECK-LTO: "-plugin-opt=O3"

// RUN: %clang %s -### --target=x86_64-fuchsia \
// RUN:     -flto=thin -flto-jobs=8 2>&1 \
// RUN:     -fuse-ld=lld \
// RUN:     | FileCheck %s -check-prefix=CHECK-THINLTO
// CHECK-THINLTO: "-plugin-opt=mcpu=x86-64"
// CHECK-THINLTO: "-plugin-opt=thinlto"
// CHECK-THINLTO: "-plugin-opt=jobs=8"

// RUN: %clang %s -### --target=x86_64-fuchsia \
// RUN:     -gsplit-dwarf -c %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-SPLIT-DWARF
// CHECK-SPLIT-DWARF: "-split-dwarf-file" "fuchsia.dwo"
