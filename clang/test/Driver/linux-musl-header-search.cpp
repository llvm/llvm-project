// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-linux-musl -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree --gcc-toolchain= \
// RUN:   | FileCheck --check-prefix=CHECK-X86-64-LIBCXX %s

// RESOURCE_DIR/include comes after /usr/include on linux-musl.
// This is different from a glibc-based distribution.
// CHECK-X86-64-LIBCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-X86-64-LIBCXX: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-X86-64-LIBCXX: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"
// CHECK-X86-64-LIBCXX: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-X86-64-LIBCXX: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// CHECK-X86-64-LIBCXX: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"

// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only -nobuiltininc 2>&1 \
// RUN:     -target x86_64-linux-musl \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree --gcc-toolchain= \
// RUN:   | FileCheck --check-prefix=CHECK-NOBUILTININC %s

// CHECK-NOBUILTININC: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-NOBUILTININC-NOT: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"

// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only -nostdlibinc 2>&1 \
// RUN:     -target x86_64-linux-musl \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree --gcc-toolchain= \
// RUN:   | FileCheck --check-prefix=CHECK-NOSTDLIBINC %s

// CHECK-NOSTDLIBINC: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-NOSTDLIBINC-NOT: "-internal-externc-isystem"
// CHECK-NOSTDLIBINC-NOT: "-internal-isystem"
// CHECK-NOSTDLIBINC: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-NOSTDLIBINC-NOT: "-internal-externc-isystem"
// CHECK-NOSTDLIBINC-NOT: "-internal-isystem"
