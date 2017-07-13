// RUN: %clang %s -### -no-canonical-prefixes --target=x86_64-unknown-fuchsia \
// RUN:     --sysroot=%S/platform -fuse-ld=ld 2>&1 | FileCheck %s
// CHECK: {{.*}}clang{{.*}}" "-cc1"
// CHECK: "-fuse-init-array"
// CHECK: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK: "-internal-externc-isystem" "[[SYSROOT]]{{/|\\\\}}include"
// CHECK: {{.*}}lld{{.*}}" "-flavor" "gnu"
// CHECK: "--sysroot=[[SYSROOT]]"
// CHECK: "-pie"
// CHECK: "--build-id"
// CHECK: "-dynamic-linker" "ld.so.1"
// CHECK: Scrt1.o
// CHECK-NOT: crti.o
// CHECK-NOT: crtbegin.o
// CHECK: "-L[[SYSROOT]]{{/|\\\\}}lib"
// CHECK: "{{.*[/\\]}}libclang_rt.builtins-x86_64.a"
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
