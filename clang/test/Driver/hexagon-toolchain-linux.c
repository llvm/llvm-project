// UNSUPPORTED: system-windows

// -----------------------------------------------------------------------------
// Passing --musl
// -----------------------------------------------------------------------------
// RUN: %clang -### -target hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -fuse-ld=lld \
// RUN:   --sysroot=%S/Inputs/basic_linux_libcxx_tree \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK000 %s
// CHECK000-NOT:  {{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crti.o
// CHECK000:      "-dynamic-linker={{/|\\\\}}lib{{/|\\\\}}ld-musl-hexagon.so.1"
// CHECK000:      "{{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crt1.o"
// CHECK000:      "-lclang_rt.builtins-hexagon" "-lc"
// -----------------------------------------------------------------------------
// Passing --musl --shared
// -----------------------------------------------------------------------------
// RUN: %clang -### -target hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   --sysroot=%S/Inputs/basic_linux_libcxx_tree -shared \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK001 %s
// CHECK001-NOT:    -dynamic-linker={{/|\\\\}}lib{{/|\\\\}}ld-musl-hexagon.so.1
// CHECK001:        "{{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crti.o"
// CHECK001:        "-lclang_rt.builtins-hexagon" "-lc"
// CHECK001-NOT:    {{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crt1.o
// -----------------------------------------------------------------------------
// Passing --musl -nostdlib
// -----------------------------------------------------------------------------
// RUN: %clang -### -target hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   --sysroot=%S/Inputs/basic_linux_libcxx_tree -nostdlib \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK002 %s
// CHECK002:       "-dynamic-linker={{/|\\\\}}lib{{/|\\\\}}ld-musl-hexagon.so.1"
// CHECK002-NOT:   {{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crti.o
// CHECK002-NOT:   {{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crt1.o
// CHECK002-NOT:   "-lclang_rt.builtins-hexagon"
// CHECK002-NOT:   "-lc"
// -----------------------------------------------------------------------------
// Passing --musl -nostartfiles
// -----------------------------------------------------------------------------
// RUN: %clang -### -target hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   --sysroot=%S/Inputs/basic_linux_libcxx_tree -nostartfiles \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK003 %s
// CHECK003:       "-dynamic-linker={{/|\\\\}}lib{{/|\\\\}}ld-musl-hexagon.so.1"
// CHECK003-NOT:   {{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}Scrt1.o
// CHECK003-NOT:   {{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crt1.o
// CHECK003:       "-lclang_rt.builtins-hexagon" "-lc"
// -----------------------------------------------------------------------------
// Passing --musl -nodefaultlibs
// -----------------------------------------------------------------------------
// RUN: %clang -### -target hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   --sysroot=%S/Inputs/basic_linux_libcxx_tree -nodefaultlibs \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK004 %s
// CHECK004:       "-dynamic-linker={{/|\\\\}}lib{{/|\\\\}}ld-musl-hexagon.so.1"
// CHECK004:       "{{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}lib{{/|\\\\}}crt1.o"
// CHECK004-NOT:   "-lclang_rt.builtins-hexagon"
// CHECK004-NOT:   "-lc"
// -----------------------------------------------------------------------------
// Not Passing -fno-use-init-array when musl is selected
// -----------------------------------------------------------------------------
// RUN: %clang -### -target hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK005 %s
// CHECK005-NOT:          -fno-use-init-array
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// c++ when musl is selected
// -----------------------------------------------------------------------------
// RUN: %clangxx -### -target hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   --sysroot=%S/Inputs/basic_linux_libcxx_tree \
// RUN:   -mcpu=hexagonv60 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK006 %s
// CHECK006:          "-internal-isystem" "{{.*}}basic_linux_libcxx_tree{{/|\\\\}}usr{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// -----------------------------------------------------------------------------
// c++ when musl is selected
// -----------------------------------------------------------------------------
// RUN: %clangxx -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -stdlib=libc++ \
// RUN:   -mcpu=hexagonv60 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK007 %s
// CHECK007:   "-internal-isystem" "{{.*}}hexagon{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// -----------------------------------------------------------------------------
// internal-isystem for linux with and without musl
// -----------------------------------------------------------------------------
// RUN: %clang -### -target hexagon-unknown-linux-musl \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -resource-dir=%S/Inputs/resource_dir \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK008 %s
// CHECK008:   InstalledDir: [[INSTALLED_DIR:.+]]
// CHECK008:   "-resource-dir" "[[RESOURCE:[^"]+]]"
// CHECK008-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]/include"
// CHECK008-SAME: {{^}} "-internal-externc-isystem" "[[INSTALLED_DIR]]/../target/hexagon/include"

// RUN: %clang -### -target hexagon-unknown-linux \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -resource-dir=%S/Inputs/resource_dir \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK009 %s
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
