// RUN: %clangxx %s -### -no-canonical-prefixes --target=x86_64-fuchsia \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --sysroot=%S/platform -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-X86_64 %s
// RUN: %clangxx %s -### -no-canonical-prefixes --target=aarch64-fuchsia \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --sysroot=%S/platform -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-AARCH64 %s
// RUN: %clangxx %s -### -no-canonical-prefixes --target=riscv64-fuchsia \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --sysroot=%S/platform -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-RISCV64 %s
// CHECK: {{.*}}clang{{.*}}" "-cc1"
// CHECK-X86_64: "-triple" "x86_64-unknown-fuchsia"
// CHECK-AARCH64: "-triple" "aarch64-unknown-fuchsia"
// CHECK-RISCV64: "-triple" "riscv64-unknown-fuchsia"
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
// CHECK-X86_64: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-AARCH64: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aarch64-fuchsia{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-RISCV64: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}riscv64-fuchsia{{/|\\\\}}libclang_rt.builtins.a"
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

// RUN: %clangxx %s -### --target=x86_64-fuchsia -nostdlib++ -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-NOSTDLIBXX
// CHECK-NOSTDLIBXX-NOT: "-lc++"
// CHECK-NOSTDLIBXX-NOT: "-lm"
// CHECK-NOSTDLIBXX: "-lc"

// RUN: %clangxx %s -### --target=x86_64-fuchsia \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB-X86
// RUN: %clangxx %s -### --target=x86_64-fuchsia -fsanitize=address \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB-X86,CHECK-MULTILIB-ASAN-X86
// RUN: %clangxx %s -### --target=x86_64-fuchsia -fno-exceptions \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB-X86,CHECK-MULTILIB-NOEXCEPT-X86
// RUN: %clangxx %s -### --target=x86_64-fuchsia -fsanitize=address -fno-exceptions \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB-X86,CHECK-MULTILIB-ASAN-NOEXCEPT-X86
// CHECK-MULTILIB-X86: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-MULTILIB-ASAN-X86: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}c++{{/|\\\\}}asan"
// CHECK-MULTILIB-NOEXCEPT-X86: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}c++{{/|\\\\}}noexcept"
// CHECK-MULTILIB-ASAN-NOEXCEPT-X86: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}c++{{/|\\\\}}asan+noexcept"
// CHECK-MULTILIB-X86: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}x86_64-fuchsia{{/|\\\\}}c++"
