// RUN: %clangxx %s -### -no-canonical-prefixes --target=x86_64-fuchsia \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --sysroot=%S/platform -fuse-ld=lld 2>&1 | FileCheck %s
// CHECK: {{.*}}clang{{.*}}" "-cc1"
// CHECK: "-triple" "x86_64-fuchsia"
// CHECK: "-fuse-init-array"
// CHECK: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK: "-internal-isystem" "{{.*[/\\]}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK: "-internal-externc-isystem" "[[SYSROOT]]{{/|\\\\}}include"
// CHECK: {{.*}}ld.lld{{.*}}" "-z" "rodynamic"
// CHECK: "--sysroot=[[SYSROOT]]"
// CHECK: "-pie"
// CHECK: "--build-id"
// CHECK: "-dynamic-linker" "ld.so.1"
// CHECK: Scrt1.o
// CHECK-NOT: crti.o
// CHECK-NOT: crtbegin.o
// CHECK: "-L[[SYSROOT]]{{/|\\\\}}lib"
// CHECK: "--push-state"
// CHECK: "--as-needed"
// CHECK: "-lc++"
// CHECK: "-lm"
// CHECK: "--pop-state"
// CHECK: "[[RESOURCE_DIR]]{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}lib{{/|\\\\}}libclang_rt.builtins.a"
// CHECK: "-lc"
// CHECK-NOT: crtend.o
// CHECK-NOT: crtn.o

// RUN: %clangxx %s -### --target=x86_64-unknown-fuchsia -stdlib=libstdc++ \
// RUN:     -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-STDLIB
// CHECK-STDLIB: error: invalid library name in argument '-stdlib=libstdc++'

// RUN: %clangxx %s -### --target=x86_64-unknown-fuchsia -static-libstdc++ \
// RUN:     -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-STATIC
// CHECK-STATIC: "--push-state"
// CHECK-STATIC: "--as-needed"
// CHECK-STATIC: "-Bstatic"
// CHECK-STATIC: "-lc++"
// CHECK-STATIC: "-Bdynamic"
// CHECK-STATIC: "-lm"
// CHECK-STATIC: "--pop-state"
// CHECK-STATIC: "-lc"

// RUN: %clang %s -### --target=x86_64-fuchsia -nostdlib++ -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-NOSTDLIBXX
// CHECK-NOSTDLIBXX-NOT: "-lc++"
// CHECK-NOSTDLIBXX-NOT: "-lm"
// CHECK-NOSTDLIBXX: "-lc"
