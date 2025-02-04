// UNSUPPORTED: system-windows

// -----------------------------------------------------------------------------
// Passing --musl
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -fuse-ld=lld \
// RUN:   --sysroot=%S/Inputs/basic_linux_libcxx_tree %s 2>&1 | FileCheck -check-prefix=CHECK000 %s
// CHECK000-NOT:  {{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crti.o
// CHECK000:      "-dynamic-linker={{/|\\\\}}lib{{/|\\\\}}ld-musl-hexagon.so.1"
// CHECK000:      "{{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crt1.o"
// CHECK000:      "-lc" "-lclang_rt.builtins-hexagon"
// -----------------------------------------------------------------------------
// Passing --musl --shared
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   --sysroot=%S/Inputs/basic_linux_libcxx_tree -shared %s 2>&1 | FileCheck -check-prefix=CHECK001 %s
// CHECK001-NOT:    -dynamic-linker={{/|\\\\}}lib{{/|\\\\}}ld-musl-hexagon.so.1
// CHECK001:        "{{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crti.o"
// CHECK001:        "-lc" "-lclang_rt.builtins-hexagon"
// CHECK001-NOT:    {{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crt1.o
// -----------------------------------------------------------------------------
// Passing --musl -nostdlib
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   --sysroot=%S/Inputs/basic_linux_libcxx_tree -nostdlib %s 2>&1 | FileCheck -check-prefix=CHECK002 %s
// CHECK002:       "-dynamic-linker={{/|\\\\}}lib{{/|\\\\}}ld-musl-hexagon.so.1"
// CHECK002-NOT:   {{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crti.o
// CHECK002-NOT:   {{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crt1.o
// CHECK002-NOT:   "-lc"
// CHECK002-NOT:   "-lclang_rt.builtins-hexagon"
// -----------------------------------------------------------------------------
// Passing --musl -nostartfiles
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   --sysroot=%S/Inputs/basic_linux_libcxx_tree -nostartfiles %s 2>&1 | FileCheck -check-prefix=CHECK003 %s
// CHECK003:       "-dynamic-linker={{/|\\\\}}lib{{/|\\\\}}ld-musl-hexagon.so.1"
// CHECK003-NOT:   {{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}Scrt1.o
// CHECK003-NOT:   {{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crt1.o
// CHECK003:       "-lc" "-lclang_rt.builtins-hexagon"
// -----------------------------------------------------------------------------
// Passing --musl -nodefaultlibs
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   --sysroot=%S/Inputs/basic_linux_libcxx_tree -nodefaultlibs %s 2>&1 | FileCheck -check-prefix=CHECK004 %s
// CHECK004:       "-dynamic-linker={{/|\\\\}}lib{{/|\\\\}}ld-musl-hexagon.so.1"
// CHECK004:       "{{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crt1.o"
// CHECK004-NOT:   "-lc"
// CHECK004-NOT:   "-lclang_rt.builtins-hexagon"
// -----------------------------------------------------------------------------
// Passing --musl -nolibc
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 --sysroot=%S/Inputs/basic_linux_libcxx_tree \
// RUN:   -nolibc %s 2>&1 | FileCheck -check-prefix=CHECK-NOLIBC %s
// CHECK-NOLIBC:       "-dynamic-linker={{/|\\\\}}lib{{/|\\\\}}ld-musl-hexagon.so.1"
// CHECK-NOLIBC-SAME:  "{{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crt1.o"
// CHECK-NOLIBC-NOT:   "-lc"
// -----------------------------------------------------------------------------
// Not Passing -fno-use-init-array when musl is selected
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 %s 2>&1 | FileCheck -check-prefix=CHECK005 %s
// CHECK005-NOT:          -fno-use-init-array
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// c++ when musl is selected
// -----------------------------------------------------------------------------
// RUN: %clangxx -### --target=hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   --sysroot=%S/Inputs/basic_linux_libcxx_tree \
// RUN:   -mcpu=hexagonv60 %s 2>&1 | FileCheck -check-prefix=CHECK006 %s
// CHECK006:          "-internal-isystem" "{{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// -----------------------------------------------------------------------------
// c++ when musl is selected
// -----------------------------------------------------------------------------
// RUN: %clangxx -### --target=hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -stdlib=libc++ \
// RUN:   -mcpu=hexagonv60 %s 2>&1 | FileCheck -check-prefix=CHECK007 %s
// CHECK007:   "-internal-isystem" "{{.*}}hexagon{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// -----------------------------------------------------------------------------
// internal-isystem for linux with and without musl
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -resource-dir=%S/Inputs/resource_dir %s 2>&1 | FileCheck -check-prefix=CHECK008 %s
// CHECK008:   InstalledDir: [[INSTALLED_DIR:.+]]
// CHECK008:   "-resource-dir" "[[RESOURCE:[^"]+]]"
// CHECK008-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]/include"
// CHECK008-SAME: {{^}} "-internal-externc-isystem" "[[INSTALLED_DIR]]/../target/hexagon/include"

// RUN: %clang -### --target=hexagon-unknown-linux \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -resource-dir=%S/Inputs/resource_dir %s 2>&1 | FileCheck -check-prefix=CHECK009 %s
// CHECK009:   InstalledDir: [[INSTALLED_DIR:.+]]
// CHECK009:   "-resource-dir" "[[RESOURCE:[^"]+]]"
// CHECK009-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]/include"
// CHECK009-SAME: {{^}} "-internal-externc-isystem" "[[INSTALLED_DIR]]/../target/hexagon/include"

// RUN: %clang -Werror -L/tmp \
// RUN:    --target=hexagon-unknown-linux-musl %s -### 2>&1 \
// RUN:    | FileCheck -check-prefix=CHECK010 %s
// CHECK010:   InstalledDir: [[INSTALLED_DIR:.+]]
// CHECK010-NOT:  "-lstandalone"
// CHECK010-NOT:  crt0_standalone.o
// CHECK010:   crt1.o
// CHECK010:   "-L/tmp"
// CHECK010-NOT:  "-lstandalone"

// -----------------------------------------------------------------------------
// unwindlib
// -----------------------------------------------------------------------------
// RUN: %clangxx --unwindlib=none \
// RUN:    --target=hexagon-unknown-linux-musl %s -### 2>&1 \
// RUN:    | FileCheck -check-prefix=CHECK011 %s
// CHECK011:   InstalledDir: [[INSTALLED_DIR:.+]]
// CHECK011:   crt1.o
// CHECK011-NOT:  "-lunwind"
// CHECK011-NOT:  "-lgcc_eh"
// CHECK012-NOT:  "-lgcc_s"


// RUN: %clangxx --rtlib=compiler-rt --unwindlib=libunwind \
// RUN:    --target=hexagon-unknown-linux-musl %s -### 2>&1 \
// RUN:    | FileCheck -check-prefix=CHECK012 %s
// RUN: %clangxx \
// RUN:    --target=hexagon-unknown-linux-musl %s -### 2>&1 \
// RUN:    | FileCheck -check-prefix=CHECK012 %s
// CHECK012:   InstalledDir: [[INSTALLED_DIR:.+]]
// CHECK012:   crt1.o
// CHECK012:  "-lunwind"
// CHECK012-NOT:  "-lgcc_eh"
// CHECK012-NOT:  "-lgcc_s"

// RUN: not %clangxx --rtlib=compiler-rt --unwindlib=libgcc \
// RUN:    --target=hexagon-unknown-linux-musl %s -### 2>&1 \
// RUN:    | FileCheck -check-prefix=CHECK013 %s
// CHECK013:  error: unsupported unwind library 'libgcc' for platform 'hexagon-unknown-linux-musl'
// CHECK013-NOT:  "-lgcc_eh"
// CHECK013-NOT:  "-lgcc_s"
// CHECK013-NOT:  "-lunwind"
