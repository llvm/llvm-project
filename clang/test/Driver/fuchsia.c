// RUN: %clang %s -### -no-canonical-prefixes --target=x86_64-unknown-fuchsia \
// RUN:     --sysroot=%S/platform 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-X86_64 %s
// RUN: %clang %s -### -no-canonical-prefixes --target=aarch64-unknown-fuchsia \
// RUN:     --sysroot=%S/platform 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-AARCH64 %s
// CHECK: {{.*}}clang{{.*}}" "-cc1"
// CHECK: "--mrelax-relocations"
// CHECK: "-munwind-tables"
// CHECK: "-fuse-init-array"
// CHECK: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK: "-internal-externc-isystem" "[[SYSROOT]]{{/|\\\\}}include"
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
// CHECK-X86_64: "{{.*[/\\]}}libclang_rt.builtins-x86_64.a"
// CHECK-AARCH64: "{{.*[/\\]}}libclang_rt.builtins-aarch64.a"
// CHECK: "-lc"
// CHECK-NOT: crtend.o
// CHECK-NOT: crtn.o

// RUN: %clang %s -### --target=x86_64-unknown-fuchsia -rtlib=libgcc 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-RTLIB
// CHECK-RTLIB: error: invalid runtime library name in argument '-rtlib=libgcc'

// RUN: %clang %s -### --target=x86_64-unknown-fuchsia -static 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-STATIC
// CHECK-STATIC: "-Bstatic"
// CHECK-STATIC: "-Bdynamic"
// CHECK-STATIC: "-lc"

// RUN: %clang %s -### --target=x86_64-unknown-fuchsia -shared 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-SHARED
// CHECK-SHARED-NOT: "-pie"
// CHECK-SHARED: "-shared"

// RUN: %clang %s -### --target=x86_64-unknown-fuchsia -r 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-RELOCATABLE
// CHECK-RELOCATABLE-NOT: "-pie"
// CHECK-RELOCATABLE-NOT: "--build-id"
// CHECK-RELOCATABLE: "-r"

// RUN: %clang %s -### --target=x86_64-unknown-fuchsia \
// RUN:     -fsanitize=safe-stack 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-SAFESTACK
// CHECK-SAFESTACK: "-fsanitize=safe-stack"
// CHECK-SAFESTACK-NOT: "{{.*[/\\]}}libclang_rt.safestack-x86_64.a"
// CHECK-SAFESTACK-NOT: "__safestack_init"

// RUN: %clang %s -### --target=x86_64-unknown-fuchsia \
// RUN:     -fsanitize=address 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-ASAN-X86
// CHECK-ASAN-X86: "-fsanitize=address"
// CHECK-ASAN-X86: "-fsanitize-address-globals-dead-stripping"
// CHECK-ASAN-X86: "-dynamic-linker" "asan/ld.so.1"
// CHECK-ASAN-X86: "{{.*[/\\]}}libclang_rt.asan-x86_64.so"
// CHECK-ASAN-X86: "{{.*[/\\]}}libclang_rt.asan-preinit-x86_64.a"

// RUN: %clang %s -### --target=aarch64-fuchsia \
// RUN:     -fsanitize=address 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-ASAN-AARCH64
// CHECK-ASAN-AARCH64: "-fsanitize=address"
// CHECK-ASAN-AARCH64: "-fsanitize-address-globals-dead-stripping"
// CHECK-ASAN-AARCH64: "-dynamic-linker" "asan/ld.so.1"
// CHECK-ASAN-AARCH64: "{{.*[/\\]}}libclang_rt.asan-aarch64.so"
// CHECK-ASAN-AARCH64: "{{.*[/\\]}}libclang_rt.asan-preinit-aarch64.a"

// RUN: %clang %s -### --target=x86_64-unknown-fuchsia \
// RUN:     -fsanitize=address -fPIC -shared 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-ASAN-SHARED
// CHECK-ASAN-SHARED: "-fsanitize=address"
// CHECK-ASAN-SHARED: "-fsanitize-address-globals-dead-stripping"
// CHECK-ASAN-SHARED: "{{.*[/\\]}}libclang_rt.asan-x86_64.so"
// CHECK-ASAN-SHARED-NOT: "{{.*[/\\]}}libclang_rt.asan-preinit-x86_64.a"

// RUN: %clang %s -### --target=x86_64-fuchsia \
// RUN:     -fsanitize=fuzzer 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-FUZZER-X86
// CHECK-FUZZER-X86: "-fsanitize=fuzzer,fuzzer-no-link"
// CHECK-FUZZER-X86: "{{.*[/\\]}}libclang_rt.fuzzer-x86_64.a"

// RUN: %clang %s -### --target=aarch64-fuchsia \
// RUN:     -fsanitize=fuzzer 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-FUZZER-AARCH64
// CHECK-FUZZER-AARCH64: "-fsanitize=fuzzer,fuzzer-no-link"
// CHECK-FUZZER-AARCH64: "{{.*[/\\]}}libclang_rt.fuzzer-aarch64.a"

// RUN: %clang %s -### --target=x86_64-fuchsia \
// RUN:     -fsanitize=scudo 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-SCUDO-X86
// CHECK-SCUDO-X86: "-fsanitize=scudo"
// CHECK-SCUDO-X86: "-pie"
// CHECK-SCUDO-X86: "{{.*[/\\]}}libclang_rt.scudo-x86_64.so"

// RUN: %clang %s -### --target=aarch64-fuchsia \
// RUN:     -fsanitize=scudo 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-SCUDO-AARCH64
// CHECK-SCUDO-AARCH64: "-fsanitize=scudo"
// CHECK-SCUDO-AARCH64: "-pie"
// CHECK-SCUDO-AARCH64: "{{.*[/\\]}}libclang_rt.scudo-aarch64.so"

// RUN: %clang %s -### --target=x86_64-fuchsia \
// RUN:     -fsanitize=scudo -fPIC -shared 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-SCUDO-SHARED
// CHECK-SCUDO-SHARED: "-fsanitize=scudo"
// CHECK-SCUDO-SHARED: "{{.*[/\\]}}libclang_rt.scudo-x86_64.so"
