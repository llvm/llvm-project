// REQUIRES: hexagon-registered-target

// -----------------------------------------------------------------------------
// Test standard include paths
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-none-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin %s 2>&1 | FileCheck -check-prefix=CHECK-C-INCLUDES %s
// CHECK-C-INCLUDES: "-cc1" {{.*}} "-internal-isystem" "{{.*}}{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]+}}{{/|\\\\}}include"
// CHECK-C-INCLUDES: "-internal-externc-isystem" "{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-none-elf{{/|\\\\}}include"

// RUN: %clangxx -### --target=hexagon-none-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin %s 2>&1 | FileCheck -check-prefix=CHECK-CXX-INCLUDES %s
// CHECK-CXX-INCLUDES: "-cc1" {{.*}} "-internal-isystem" "{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-none-elf{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK-CXX-INCLUDES: "-internal-isystem" "{{.*}}{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]+}}{{/|\\\\}}include"
// CHECK-CXX-INCLUDES: "-internal-externc-isystem" "{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-none-elf{{/|\\\\}}include"
// -----------------------------------------------------------------------------
// Passing start files for Picolibc
// -----------------------------------------------------------------------------
// RUN: %clang --target=hexagon-none-elf --cstdlib=picolibc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-STARTUP
// CHECK-STARTUP: "{{.*}}crt0-semihost.o"
//
// RUN: %clang --target=hexagon-none-elf --cstdlib=picolibc -nostartfiles -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOSTART
// CHECK-NOSTART-NOT: "{{.*}}crt0-semihost.o"
// -----------------------------------------------------------------------------
// Passing  -nostdlib, -nostartfiles, -nodefaultlibs, -nolibc
// -----------------------------------------------------------------------------
// RUN: %clangxx -### --target=hexagon-none-elf --cstdlib=picolibc \
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
// CHECK-NOSTDLIB-NOT: "-l{{(clang_rt\.builtins)}}"
// CHECK-NOSTDLIB-NOT: "--end-group"

// RUN: %clangxx -### --target=hexagon-none-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -nostartfiles %s 2>&1 | FileCheck -check-prefix=CHECK-NOSTARTFILES %s
// CHECK-NOSTARTFILES: "-cc1"
// CHECK-NOSTARTFILES: {{hexagon-link|ld}}
// CHECK-NOSTARTFILES-NOT: {{.*}}crt0-semihost.o
// CHECK-NOSTARTFILES: "-lc++" "-lc++abi" "-lunwind" "-lm" "--start-group" "-lsemihost" "-lc" "-lclang_rt.builtins" "--end-group"

// RUN: %clangxx -### --target=hexagon-none-elf --cstdlib=picolibc \
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

// RUN: %clangxx -### --target=hexagon-none-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin -mcpu=hexagonv60 \
// RUN:   -nolibc %s 2>&1 | FileCheck -check-prefix=CHECK-NOLIBC %s
// CHECK-NOLIBC: "-cc1"
// CHECK-NOLIBC: {{hexagon-link|ld}}
// CHECK-NOLIBC: "{{.*}}crt0-semihost.o"
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
// RUN: %clang --target=hexagon-none-elf --cstdlib=picolibc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-RTLIB
// RUN: %clangxx --target=hexagon-none-elf --cstdlib=picolibc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-RTLIB
// CHECK-RTLIB: "-lclang_rt.builtins"
// CHECK-RTLIB-NOT: "-lgcc"
// -----------------------------------------------------------------------------
// Allow --rtlib to override the default compiler-rt when Picolibc is selected
// -----------------------------------------------------------------------------
// RUN: %clang --target=hexagon-none-elf --cstdlib=picolibc --rtlib=libgcc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-RTLIB-OVERRIDE
// CHECK-RTLIB-OVERRIDE: "-lgcc"
// CHECK-RTLIB-OVERRIDE-NOT: "-lclang_rt.builtins"
// -----------------------------------------------------------------------------
// libunwind is linked by default for C++ when Picolibc is selected; user can
// override with --unwindlib=
// -----------------------------------------------------------------------------
// RUN: %clangxx --target=hexagon-none-elf --cstdlib=picolibc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-CXX-UNWIND
// CHECK-CXX-UNWIND: "-lunwind"
// RUN: %clangxx --target=hexagon-none-elf --cstdlib=picolibc --unwindlib=none -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-UNWIND-OVERRIDE
// CHECK-UNWIND-OVERRIDE-NOT: "-lunwind"
// -----------------------------------------------------------------------------
// Library search paths
// -----------------------------------------------------------------------------
// RUN: %clang --target=hexagon-none-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv68 -### %s 2>&1 | FileCheck -check-prefix=CHECK-LIBPATHS %s
// CHECK-LIBPATHS: "-L{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-none-elf{{/|\\\\}}lib{{/|\\\\}}v68"
// CHECK-LIBPATHS-NOT: "-L{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-none-elf{{/|\\\\}}lib{{/|\\\\}}v68{{/|\\\\}}G0"

// RUN: %clang --target=hexagon-none-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv68 -G0 -### %s 2>&1 | FileCheck -check-prefix=CHECK-LIBPATHS-G0 %s
// CHECK-LIBPATHS-G0: "-L{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-none-elf{{/|\\\\}}lib{{/|\\\\}}v68{{/|\\\\}}G0"
// CHECK-LIBPATHS-G0: "-L{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-none-elf{{/|\\\\}}lib{{/|\\\\}}v68"

// RUN: %clang --target=hexagon-none-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv68 -fpic -### %s 2>&1 | FileCheck -check-prefix=CHECK-LIBPATHS-PIC %s
// CHECK-LIBPATHS-PIC: "-L{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-none-elf{{/|\\\\}}lib{{/|\\\\}}v68{{/|\\\\}}G0{{/|\\\\}}pic"
// CHECK-LIBPATHS-PIC: "-L{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-none-elf{{/|\\\\}}lib{{/|\\\\}}v68{{/|\\\\}}G0"
// CHECK-LIBPATHS-PIC: "-L{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-none-elf{{/|\\\\}}lib{{/|\\\\}}v68"

// =============================================================================
// H2 OS tests (--target=hexagon-h2-elf --cstdlib=picolibc)
// Differences from hexagon-none-elf: crt0-noflash-hosted.o, -lh2 -lsyscall_wrapper
// =============================================================================

// -----------------------------------------------------------------------------
// Test standard include paths for H2
// -----------------------------------------------------------------------------
// RUN: %clang -### --target=hexagon-h2-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin %s 2>&1 | FileCheck -check-prefix=CHECK-H2-C-INCLUDES %s
// CHECK-H2-C-INCLUDES: "-cc1" {{.*}} "-internal-isystem" "{{.*}}{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]+}}{{/|\\\\}}include"
// CHECK-H2-C-INCLUDES: "-internal-externc-isystem" "{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-h2-elf{{/|\\\\}}include"

// RUN: %clangxx -### --target=hexagon-h2-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin %s 2>&1 | FileCheck -check-prefix=CHECK-H2-CXX-INCLUDES %s
// CHECK-H2-CXX-INCLUDES: "-cc1" {{.*}} "-internal-isystem" "{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-h2-elf{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK-H2-CXX-INCLUDES: "-internal-isystem" "{{.*}}{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]+}}{{/|\\\\}}include"
// CHECK-H2-CXX-INCLUDES: "-internal-externc-isystem" "{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-h2-elf{{/|\\\\}}include"

// -----------------------------------------------------------------------------
// H2 start files: crt0-noflash-hosted.o (not crt0-semihost.o)
// -----------------------------------------------------------------------------
// RUN: %clang --target=hexagon-h2-elf --cstdlib=picolibc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-H2-STARTUP
// CHECK-H2-STARTUP: "{{.*}}crt0-noflash-hosted.o"
// CHECK-H2-STARTUP-NOT: "{{.*}}crt0-semihost.o"

// RUN: %clang --target=hexagon-h2-elf --cstdlib=picolibc -nostartfiles -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-H2-NOSTART
// CHECK-H2-NOSTART-NOT: "{{.*}}crt0-noflash-hosted.o"

// -----------------------------------------------------------------------------
// H2: -nostdlib, -nostartfiles, -nodefaultlibs, -nolibc
// -----------------------------------------------------------------------------
// RUN: %clangxx -### --target=hexagon-h2-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv68 \
// RUN:   -nostdlib %s 2>&1 | FileCheck -check-prefix=CHECK-H2-NOSTDLIB %s
// CHECK-H2-NOSTDLIB: "-cc1"
// CHECK-H2-NOSTDLIB: {{hexagon-link|ld}}
// CHECK-H2-NOSTDLIB-NOT: "{{.*}}crt0-noflash-hosted.o"
// CHECK-H2-NOSTDLIB-NOT: "-lc++"
// CHECK-H2-NOSTDLIB-NOT: "-lm"
// CHECK-H2-NOSTDLIB-NOT: "--start-group"
// CHECK-H2-NOSTDLIB-NOT: "-lh2"
// CHECK-H2-NOSTDLIB-NOT: "-lsyscall_wrapper"
// CHECK-H2-NOSTDLIB-NOT: "-lc"
// CHECK-H2-NOSTDLIB-NOT: "-lclang_rt.builtins"
// CHECK-H2-NOSTDLIB-NOT: "--end-group"

// RUN: %clangxx -### --target=hexagon-h2-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv68 \
// RUN:   -nostartfiles %s 2>&1 | FileCheck -check-prefix=CHECK-H2-NOSTARTFILES %s
// CHECK-H2-NOSTARTFILES: "-cc1"
// CHECK-H2-NOSTARTFILES: {{hexagon-link|ld}}
// CHECK-H2-NOSTARTFILES-NOT: "{{.*}}crt0-noflash-hosted.o"
// CHECK-H2-NOSTARTFILES: "-lc++" "-lc++abi" "-lunwind" "-lm" "--start-group" "-lh2" "-lsyscall_wrapper" "-lc" "-lclang_rt.builtins" "--end-group"

// RUN: %clangxx -### --target=hexagon-h2-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv68 \
// RUN:   -nodefaultlibs %s 2>&1 | FileCheck -check-prefix=CHECK-H2-NODEFAULTLIBS %s
// CHECK-H2-NODEFAULTLIBS: "-cc1"
// CHECK-H2-NODEFAULTLIBS: {{hexagon-link|ld}}
// CHECK-H2-NODEFAULTLIBS: "{{.*}}crt0-noflash-hosted.o"
// CHECK-H2-NODEFAULTLIBS-NOT: "-lc++"
// CHECK-H2-NODEFAULTLIBS-NOT: "-lm"
// CHECK-H2-NODEFAULTLIBS-NOT: "--start-group"
// CHECK-H2-NODEFAULTLIBS-NOT: "-lh2"
// CHECK-H2-NODEFAULTLIBS-NOT: "-lsyscall_wrapper"
// CHECK-H2-NODEFAULTLIBS-NOT: "-lc"
// CHECK-H2-NODEFAULTLIBS-NOT: "-lclang_rt.builtins"
// CHECK-H2-NODEFAULTLIBS-NOT: "--end-group"

// RUN: %clangxx -### --target=hexagon-h2-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv68 \
// RUN:   -nolibc %s 2>&1 | FileCheck -check-prefix=CHECK-H2-NOLIBC %s
// CHECK-H2-NOLIBC: "-cc1"
// CHECK-H2-NOLIBC: {{hexagon-link|ld}}
// CHECK-H2-NOLIBC: "{{.*}}crt0-noflash-hosted.o"
// CHECK-H2-NOLIBC-SAME: "-lc++"
// CHECK-H2-NOLIBC-SAME: "-lm"
// CHECK-H2-NOLIBC-SAME: "--start-group"
// CHECK-H2-NOLIBC-SAME: "-lh2"
// CHECK-H2-NOLIBC-SAME: "-lsyscall_wrapper"
// CHECK-H2-NOLIBC-NOT: "-lc"
// CHECK-H2-NOLIBC-SAME: "-lclang_rt.builtins"
// CHECK-H2-NOLIBC-SAME: "--end-group"

// -----------------------------------------------------------------------------
// H2: compiler-rt is forced (not -lgcc)
// -----------------------------------------------------------------------------
// RUN: %clang --target=hexagon-h2-elf --cstdlib=picolibc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-H2-RTLIB
// RUN: %clangxx --target=hexagon-h2-elf --cstdlib=picolibc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-H2-RTLIB
// CHECK-H2-RTLIB: "-lclang_rt.builtins"
// CHECK-H2-RTLIB-NOT: "-lgcc"

// -----------------------------------------------------------------------------
// H2: libunwind linked for C++ but not C
// -----------------------------------------------------------------------------
// RUN: %clangxx --target=hexagon-h2-elf --cstdlib=picolibc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-H2-CXX-UNWIND
// CHECK-H2-CXX-UNWIND: "-lunwind"

// -----------------------------------------------------------------------------
// H2: library search paths use target/picolibc/hexagon-unknown-h2-elf/
// -----------------------------------------------------------------------------
// RUN: %clang --target=hexagon-h2-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv68 -### %s 2>&1 | FileCheck -check-prefix=CHECK-H2-LIBPATHS %s
// CHECK-H2-LIBPATHS: "-L{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-h2-elf{{/|\\\\}}lib{{/|\\\\}}v68"
// CHECK-H2-LIBPATHS-NOT: "-L{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-h2-elf{{/|\\\\}}lib{{/|\\\\}}v68{{/|\\\\}}G0"

// RUN: %clang --target=hexagon-h2-elf --cstdlib=picolibc \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv68 -G0 -### %s 2>&1 | FileCheck -check-prefix=CHECK-H2-LIBPATHS-G0 %s
// CHECK-H2-LIBPATHS-G0: "-L{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-h2-elf{{/|\\\\}}lib{{/|\\\\}}v68{{/|\\\\}}G0"
// CHECK-H2-LIBPATHS-G0: "-L{{.*}}{{/|\\\\}}Inputs{{/|\\\\}}hexagon_tree{{/|\\\\}}Tools{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-h2-elf{{/|\\\\}}lib{{/|\\\\}}v68"
