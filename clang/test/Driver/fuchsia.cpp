// RUN: %clangxx %s -### -no-canonical-prefixes --target=x86_64-unknown-fuchsia \
// RUN:     --sysroot=%S/platform 2>&1 | FileCheck %s
// CHECK: {{.*}}clang{{.*}}" "-cc1"
// CHECK: "-triple" "x86_64-fuchsia"
// CHECK: "-fuse-init-array"
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
// CHECK: "-lc++" "-lm"
// CHECK: "{{.*[/\\]}}libclang_rt.builtins-x86_64.a"
// CHECK: "-lc"
// CHECK-NOT: crtend.o
// CHECK-NOT: crtn.o

// RUN: %clangxx %s -### --target=x86_64-unknown-fuchsia -stdlib=libstdc++ 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-STDLIB
// CHECK-STDLIB: error: invalid library name in argument '-stdlib=libstdc++'

// RUN: %clangxx %s -### --target=x86_64-unknown-fuchsia -static-libstdc++ 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-STATIC
// CHECK-STATIC: "-Bstatic"
// CHECK-STATIC: "-lc++"
// CHECK-STATIC: "-Bdynamic"
// CHECK-STATIC: "-lm"
// CHECK-STATIC: "-lc"
