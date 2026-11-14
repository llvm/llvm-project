// REQUIRES: hexagon-registered-target

// Every Hexagon toolchain searches <resource-dir>/lib/hexagon/<variant> for
// builtins, whatever the target OS or libc. The variant is derived from the
// CPU and the -G0 / -fpic flags in the same way for all of them.

// -----------------------------------------------------------------------------
// Default libc (no --cstdlib) for each Hexagon OS
// -----------------------------------------------------------------------------
// RUN: %clang --target=hexagon-none-elf -mcpu=hexagonv68 --sysroot= \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin -### %s 2>&1 | FileCheck -check-prefix=CHECK-NONE %s
// CHECK-NONE: "-L{{.*}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]+}}{{/|\\\\}}lib{{/|\\\\}}hexagon{{/|\\\\}}v68"

// RUN: %clang --target=hexagon-unknown-elf -mcpu=hexagonv68 --sysroot= \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin -### %s 2>&1 | FileCheck -check-prefix=CHECK-UNKNOWN-ELF %s
// CHECK-UNKNOWN-ELF: "-L{{.*}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]+}}{{/|\\\\}}lib{{/|\\\\}}hexagon{{/|\\\\}}v68"

// -----------------------------------------------------------------------------
// Picolibc, for each Hexagon OS
// -----------------------------------------------------------------------------
// RUN: %clang --target=hexagon-none-elf --cstdlib=picolibc -mcpu=hexagonv68 --sysroot= \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin -### %s 2>&1 | FileCheck -check-prefix=CHECK-PICOLIBC %s
// CHECK-PICOLIBC: "-L{{.*}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]+}}{{/|\\\\}}lib{{/|\\\\}}hexagon{{/|\\\\}}v68"

// RUN: %clang --target=hexagon-h2-elf --cstdlib=picolibc -mcpu=hexagonv68 --sysroot= \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin -### %s 2>&1 | FileCheck -check-prefix=CHECK-H2 %s
// CHECK-H2: "-L{{.*}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]+}}{{/|\\\\}}lib{{/|\\\\}}hexagon{{/|\\\\}}v68"

// -----------------------------------------------------------------------------
// Linux (musl)
// -----------------------------------------------------------------------------
// RUN: %clang --target=hexagon-unknown-linux-musl -mcpu=hexagonv68 -fuse-ld=lld \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   --sysroot=%S/Inputs/basic_linux_libcxx_tree -### %s 2>&1 | FileCheck -check-prefix=CHECK-LINUX %s
// CHECK-LINUX: "-L{{.*}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]+}}{{/|\\\\}}lib{{/|\\\\}}hexagon{{/|\\\\}}v68"

// -----------------------------------------------------------------------------
// Variant selection: -G0 and -fpic select the matching packaged directory
// -----------------------------------------------------------------------------
// RUN: %clang --target=hexagon-none-elf -mcpu=hexagonv68 -G0 -fpic --sysroot= \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin -### %s 2>&1 | FileCheck -check-prefix=CHECK-VARIANT-NONE %s
// CHECK-VARIANT-NONE: "-L{{.*}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]+}}{{/|\\\\}}lib{{/|\\\\}}hexagon{{/|\\\\}}v68-G0-pic"

// RUN: %clang --target=hexagon-none-elf --cstdlib=picolibc -mcpu=hexagonv68 -G0 --sysroot= \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin -### %s 2>&1 | FileCheck -check-prefix=CHECK-VARIANT-PICOLIBC %s
// CHECK-VARIANT-PICOLIBC: "-L{{.*}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]+}}{{/|\\\\}}lib{{/|\\\\}}hexagon{{/|\\\\}}v68-G0"

// RUN: %clang --target=hexagon-unknown-linux-musl -mcpu=hexagonv68 -G0 -fpic -fuse-ld=lld \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   --sysroot=%S/Inputs/basic_linux_libcxx_tree -### %s 2>&1 | FileCheck -check-prefix=CHECK-VARIANT-LINUX %s
// CHECK-VARIANT-LINUX: "-L{{.*}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]+}}{{/|\\\\}}lib{{/|\\\\}}hexagon{{/|\\\\}}v68-G0-pic"

// -----------------------------------------------------------------------------
// Resource dir is searched before the sysroot
// -----------------------------------------------------------------------------
// RUN: %clang --target=hexagon-none-elf -mcpu=hexagonv68 --sysroot= \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin -### %s 2>&1 | FileCheck -check-prefix=CHECK-ORDER-NONE %s
// CHECK-ORDER-NONE: "-L{{.*}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]+}}{{/|\\\\}}lib{{/|\\\\}}hexagon{{/|\\\\}}v68"
// CHECK-ORDER-NONE: "-L{{.*}}{{/|\\\\}}target{{/|\\\\}}hexagon{{/|\\\\}}lib{{/|\\\\}}v68"

// RUN: %clang --target=hexagon-none-elf --cstdlib=picolibc -mcpu=hexagonv68 --sysroot= \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin -### %s 2>&1 | FileCheck -check-prefix=CHECK-ORDER-PICOLIBC %s
// CHECK-ORDER-PICOLIBC: "-L{{.*}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]+}}{{/|\\\\}}lib{{/|\\\\}}hexagon{{/|\\\\}}v68"
// CHECK-ORDER-PICOLIBC: "-L{{.*}}{{/|\\\\}}target{{/|\\\\}}picolibc{{/|\\\\}}hexagon-unknown-none-elf{{/|\\\\}}lib{{/|\\\\}}v68"

// -----------------------------------------------------------------------------
// Runtime selection: -lgcc by default, -lclang_rt.builtins with --rtlib=compiler-rt
// -----------------------------------------------------------------------------
// RUN: %clang --target=hexagon-none-elf -mcpu=hexagonv68 --sysroot= \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin -### %s 2>&1 | FileCheck -check-prefix=CHECK-RT-DEFAULT %s
// CHECK-RT-DEFAULT: "-lgcc"
// CHECK-RT-DEFAULT-NOT: "-lclang_rt.builtins"

// RUN: %clang --target=hexagon-none-elf -mcpu=hexagonv68 --rtlib=compiler-rt --sysroot= \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin -### %s 2>&1 | FileCheck -check-prefix=CHECK-RT-COMPILER-RT %s
// CHECK-RT-COMPILER-RT: "-lclang_rt.builtins"
// CHECK-RT-COMPILER-RT-NOT: "-lgcc"
