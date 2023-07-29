// RUN: %clangxx -### %s --target=x86_64-unknown-fuchsia \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --sysroot=%S/platform -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-X86_64 %s
// RUN: %clangxx -### %s --target=aarch64-unknown-fuchsia \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --sysroot=%S/platform -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-AARCH64 %s
// RUN: %clangxx -### %s --target=riscv64-unknown-fuchsia \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --sysroot=%S/platform -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-RISCV64 %s
// RUN: %clangxx -### %s --target=x86_64-fuchsia \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --sysroot=%S/platform -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-X86_64 %s
// RUN: %clangxx -### %s --target=aarch64-fuchsia \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --sysroot=%S/platform -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-AARCH64 %s
// RUN: %clangxx -### %s --target=riscv64-fuchsia \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     --sysroot=%S/platform -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK,CHECK-RISCV64 %s
// CHECK: "-cc1"
// CHECK-X86_64: "-triple" "x86_64-unknown-fuchsia"
// CHECK-AARCH64: "-triple" "aarch64-unknown-fuchsia"
// CHECK-RISCV64: "-triple" "riscv64-unknown-fuchsia"
// CHECK-NOT: "-fno-use-init-array"
// CHECK: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-X86_64: "-internal-isystem" "{{.*[/\\]}}include{{/|\\\\}}x86_64-unknown-fuchsia{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK-AARCH64: "-internal-isystem" "{{.*[/\\]}}include{{/|\\\\}}aarch64-unknown-fuchsia{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK-RISCV64: "-internal-isystem" "{{.*[/\\]}}include{{/|\\\\}}riscv64-unknown-fuchsia{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK: "-internal-isystem" "{{.*[/\\]}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK: "-internal-externc-isystem" "[[SYSROOT]]{{/|\\\\}}include"
// CHECK: {{.*}}ld.lld{{.*}}" "-z" "now" "-z" "rodynamic" "-z" "separate-loadable-segments"
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
// CHECK-X86_64: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}x86_64-unknown-fuchsia{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-AARCH64: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aarch64-unknown-fuchsia{{/|\\\\}}libclang_rt.builtins.a"
// CHECK-RISCV64: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}riscv64-unknown-fuchsia{{/|\\\\}}libclang_rt.builtins.a"
// CHECK: "-lc"
// CHECK-NOT: crtend.o
// CHECK-NOT: crtn.o

// RUN: %clangxx -### %s --target=x86_64-unknown-fuchsia -stdlib=libstdc++ \
// RUN:     -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-STDLIB
// CHECK-STDLIB: error: invalid library name in argument '-stdlib=libstdc++'

// RUN: %clangxx -### %s --target=x86_64-unknown-fuchsia -static-libstdc++ \
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

// RUN: %clangxx -### %s --target=x86_64-unknown-fuchsia -nostdlib++ -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-NOSTDLIBXX
// CHECK-NOSTDLIBXX-NOT: "-lc++"
// CHECK-NOSTDLIBXX-NOT: "-lm"
// CHECK-NOSTDLIBXX: "-lc"

// RUN: %clangxx -### %s --target=x86_64-unknown-fuchsia \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB-X86
// RUN: %clangxx -### %s --target=x86_64-unknown-fuchsia -fsanitize=address \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB-X86,CHECK-MULTILIB-ASAN-X86
// RUN: %clangxx -### %s --target=x86_64-unknown-fuchsia -fno-exceptions \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB-X86,CHECK-MULTILIB-NOEXCEPT-X86
// RUN: %clangxx -### %s --target=x86_64-unknown-fuchsia -fsanitize=address -fno-exceptions \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB-X86,CHECK-MULTILIB-ASAN-NOEXCEPT-X86
// RUN: %clangxx -### %s --target=x86_64-unknown-fuchsia -fsanitize=hwaddress \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB-X86,CHECK-MULTILIB-HWASAN-X86
// RUN: %clangxx -### %s --target=x86_64-unknown-fuchsia -fsanitize=hwaddress -fno-exceptions \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB-X86,CHECK-MULTILIB-HWASAN-NOEXCEPT-X86

// Test compat multilibs.
// RUN: %clangxx -### %s --target=x86_64-unknown-fuchsia -fc++-abi=itanium \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB-X86,CHECK-MULTILIB-COMPAT-X86
// RUN: %clangxx -### %s --target=x86_64-unknown-fuchsia -fc++-abi=itanium -fc++-abi=fuchsia \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB-X86
// RUN: %clangxx -### %s --target=x86_64-unknown-fuchsia -fc++-abi=fuchsia -fc++-abi=itanium \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     -fuse-ld=lld 2>&1\
// RUN:     | FileCheck %s -check-prefixes=CHECK-MULTILIB-X86,CHECK-MULTILIB-COMPAT-X86
// CHECK-MULTILIB-X86: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-MULTILIB-ASAN-X86: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}x86_64-unknown-fuchsia{{/|\\\\}}asan"
// CHECK-MULTILIB-NOEXCEPT-X86: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}x86_64-unknown-fuchsia{{/|\\\\}}noexcept"
// CHECK-MULTILIB-ASAN-NOEXCEPT-X86: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}x86_64-unknown-fuchsia{{/|\\\\}}asan+noexcept"
// CHECK-MULTILIB-HWASAN-X86: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}x86_64-unknown-fuchsia{{/|\\\\}}hwasan"
// CHECK-MULTILIB-HWASAN-NOEXCEPT-X86: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}x86_64-unknown-fuchsia{{/|\\\\}}hwasan+noexcept"
// CHECK-MULTILIB-COMPAT-X86: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}x86_64-unknown-fuchsia{{/|\\\\}}compat"
// CHECK-MULTILIB-X86: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}lib{{/|\\\\}}x86_64-unknown-fuchsia"

// Check that -print-multi-directory only outputs one multilib directory.
// This may be relaxed later but for now preserve existing behaviour.
// RUN: %clangxx -print-multi-directory --target=x86_64-unknown-fuchsia -fsanitize=address -fno-exceptions \
// RUN:     -ccc-install-dir %S/Inputs/basic_fuchsia_tree/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:     | FileCheck %s -check-prefixes=CHECK-PRINT-MULTI-LIB
// CHECK-PRINT-MULTI-LIB-NOT: .
// CHECK-PRINT-MULTI-LIB-NOT: noexcept
// CHECK-PRINT-MULTI-LIB-NOT: asan
// CHECK-PRINT-MULTI-LIB: asan+noexcept
