// UNSUPPORTED: system-windows
// REQUIRES: hexagon-registered-target

// -----------------------------------------------------------------------------
// Test standard include paths
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-none-picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin %s 2>&1 | FileCheck -check-prefix=CHECK-C-INCLUDES %s
// CHECK-C-INCLUDES: "-cc1" {{.*}} "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon-unknown-none-picolibc/include"

// RUN: %clangxx -### --target=hexagon-none-picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin %s 2>&1 | FileCheck -check-prefix=CHECK-CXX-INCLUDES %s
// CHECK-CXX-INCLUDES: "-cc1" {{.*}} "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon-unknown-none-picolibc/include/c++/v1"
// CHECK-CXX-INCLUDES: "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon-unknown-none-picolibc/include"
// -----------------------------------------------------------------------------
// Passing start files for Picolibc
// -----------------------------------------------------------------------------
// RUN: %clang -target hexagon-none-picolibc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-STARTUP
// CHECK-STARTUP: "{{.*}}crt0-semihost.o"
//
// RUN: %clang -target hexagon-none-picolibc -nostartfiles -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOSTART
// CHECK-NOSTART-NOT: "{{.*}}crt0-semihost.o"
// -----------------------------------------------------------------------------
// Passing  -nostdlib, -nostartfiles, -nodefaultlibs, -nolibc
// -----------------------------------------------------------------------------
// RUN: %clangxx -### --target=hexagon-none-picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -nostdlib %s 2>&1 | FileCheck -check-prefix=CHECK-NOSTDLIB %s
// CHECK-NOSTDLIB: "-cc1"
// CHECK-NOSTDLIB: {{hexagon-link|ld}}
// CHECK-NOSTDLIB-NOT: {{.*}}crt0-semihost.o
// CHECK-NOSTDLIB-NOT: "-lc++"
// CHECK-NOSTDLIB-NOT: "-lm"
// CHECK-NOSTDLIB-NOT: "--start-group"
// CHECK-NOSTDLIB-NOT: "-lsemihost"
// CHECK-NOSTDLIB-NOT: "-lc"
// CHECK-NOSTDLIB-NOT: "-lclang_rt.builtins"
// CHECK-NOSTDLIB-NOT: "--end-group"

// RUN: %clangxx -### --target=hexagon-none-picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -nostartfiles %s 2>&1 | FileCheck -check-prefix=CHECK-NOSTARTFILES %s
// CHECK-NOSTARTFILES: "-cc1"
// CHECK-NOSTARTFILES: {{hexagon-link|ld}}
// CHECK-NOSTARTFILES-NOT: {{.*}}crt0-semihost.o
// CHECK-NOSTARTFILES: "-lc++" "-lc++abi" "-lunwind" "-lm" "--start-group" "-lsemihost" "-lc" "-lclang_rt.builtins" "--end-group"

// RUN: %clangxx -### --target=hexagon-none-picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -nodefaultlibs %s 2>&1 | FileCheck -check-prefix=CHECK-NODEFAULTLIBS %s
// CHECK-NODEFAULTLIBS: "-cc1"
// CHECK-NODEFAULTLIBS: {{hexagon-link|ld}}
// CHECK-NODEFAULTLIBS: "{{.*}}crt0-semihost.o"
// CHECK-NODEFAULTLIBS-NOT: "-lc++"
// CHECK-NODEFAULTLIBS-NOT: "-lm"
// CHECK-NODEFAULTLIBS-NOT: "--start-group"
// CHECK-NODEFAULTLIBS-NOT: "-lsemihost"
// CHECK-NODEFAULTLIBS-NOT: "-lc"
// CHECK-NODEFAULTLIBS-NOT: "-lclang_rt.builtins"
// CHECK-NODEFAULTLIBS-NOT: "--end-group"

// RUN: %clangxx -### --target=hexagon-none-picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin -mcpu=hexagonv60 \
// RUN:   -nolibc %s 2>&1 | FileCheck -check-prefix=CHECK-NOLIBC %s
// CHECK-NOLIBC: "-cc1"
// CHECK-NOLIBC: hexagon-link
// CHECK-NOLIBC-SAME: "{{.*}}crt0-semihost.o"
// CHECK-NOLIBC-SAME: "-lc++"
// CHECK-NOLIBC-SAME: "-lm"
// CHECK-NOLIBC-SAME: "--start-group"
// CHECK-NOLIBC-SAME: "-lsemihost"
// CHECK-NOLIBC-NOT: "-lc"
// CHECK-NOLIBC-SAME: "-lclang_rt.builtins"
// CHECK-NOLIBC-SAME: "--end-group"
// -----------------------------------------------------------------------------
// Force compiler-rt when Picolibc is selected
// -----------------------------------------------------------------------------
// RUN: %clang -target hexagon-none-picolibc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-RTLIB
// RUN: %clangxx -target hexagon-none-picolibc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-RTLIB
// CHECK-RTLIB: "-lclang_rt.builtins"
// -----------------------------------------------------------------------------
// Force libunwind when Picolibc is selected
// -----------------------------------------------------------------------------
// RUN: %clang -target hexagon-none-picolibc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-C-UNWIND
// RUN: %clangxx -target hexagon-none-picolibc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-CXX-UNWIND
// CHECK-C-UNWIND-NOT: "-lunwind"
// CHECK-CXX-UNWIND: "-lunwind"
// -----------------------------------------------------------------------------
// Force G0 for Picolibc
// -----------------------------------------------------------------------------
// RUN: %clang -target hexagon-none-picolibc -### \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv68 %s 2>&1 | FileCheck %s --check-prefix=CHECK-G0
// RUN: %clangxx -target hexagon-none-picolibc -### \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv68 %s 2>&1 | FileCheck %s --check-prefix=CHECK-G0
// CHECK-G0: "{{.*}}/G0/crt0-semihost.o"
// CHECK-G0-SAME: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon-unknown-none-picolibc/lib/v68/G0"
// -----------------------------------------------------------------------------
// Libc++ experimental library linkage
// -----------------------------------------------------------------------------
// RUN: %clangxx -### --target=hexagon-none-picolibc -fexperimental-library %s 2>&1 | FileCheck %s --check-prefix=CHECK-EXPERIMENTAL
// CHECK-EXPERIMENTAL: "-lc++experimental"
// -----------------------------------------------------------------------------
// Custom -L forwarding
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-none-picolibc -L/foo/bar %s 2>&1 | FileCheck %s --check-prefix=CHECK-CUSTOM-L
// CHECK-CUSTOM-L: "-L/foo/bar"
// -----------------------------------------------------------------------------
// Link arch flags propagation
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-none-picolibc -mcpu=hexagonv68 %s 2>&1 | FileCheck %s --check-prefix=CHECK-LINK-ARCH
// CHECK-LINK-ARCH: "-march=hexagon"
// CHECK-LINK-ARCH: "-mcpu=hexagonv68"
// -----------------------------------------------------------------------------
// No standard includes when -nostdinc (C only); -nostdinc++ blocks C++ headers
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-none-picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin -nostdinc %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOSTDINC-C
// CHECK-NOSTDINC-C-NOT: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon-unknown-none-picolibc/include"
// RUN: %clangxx -### --target=hexagon-none-picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin -nostdinc++ %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOSTDINCXX
// CHECK-NOSTDINCXX-NOT: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon-unknown-none-picolibc/include/c++/v1"
// CHECK-NOSTDINCXX: "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon-unknown-none-picolibc/include"
// -----------------------------------------------------------------------------
// C linking does not include -lm by default
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-none-picolibc %s 2>&1 | FileCheck %s --check-prefix=CHECK-C-NO-LM
// CHECK-C-NO-LM-NOT: "-lm"
// -----------------------------------------------------------------------------
