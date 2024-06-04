// Check passing Mips ABI options to the backend.
//
// REQUIRES: mips-registered-target
//
// RUN: %clang --target=mips-linux-gnu -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS32R2-O32 %s
// RUN: %clang --target=mips64-linux-gnu -mips32r2 -mabi=32 -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS32R2-O32 %s
// MIPS32R2-O32: "-target-cpu" "mips32r2"
// MIPS32R2-O32: "-target-abi" "o32"
//
// RUN: %clang --target=mips64-linux-gnu -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64R2-N64 %s
// RUN: %clang --target=mips-img-linux-gnu -mips64r2 -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64R2-N64 %s
// RUN: %clang --target=mips-mti-linux-gnu -mips64r2 -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64R2-N64 %s
// RUN: %clang --target=mips-linux-gnu -mips64r2 -mabi=64 -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64R2-N64 %s
// MIPS64R2-N64: "-target-cpu" "mips64r2"
// MIPS64R2-N64: "-target-abi" "n64"
//
// RUN: %clang --target=mips64-linux-gnu -### -mips64r3 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64R3-N64 %s
// RUN: %clang --target=mips-img-linux-gnu -mips64r3 -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64R3-N64 %s
// RUN: %clang --target=mips-mti-linux-gnu -mips64r3 -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64R3-N64 %s
// MIPS64R3-N64: "-target-cpu" "mips64r3"
// MIPS64R3-N64: "-target-abi" "n64"
//
// RUN: %clang --target=mips-linux-gnu -### -c %s \
// RUN:        -mabi=32 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-32 %s
// MIPS-ABI-32: "-target-cpu" "mips32r2"
// MIPS-ABI-32: "-target-abi" "o32"
//
// RUN: %clang --target=mips-linux-gnu -### -c %s \
// RUN:        -mabi=o32 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-O32 %s
// MIPS-ABI-O32: "-target-cpu" "mips32r2"
// MIPS-ABI-O32: "-target-abi" "o32"
//
// RUN: %clang --target=mips-linux-gnu -### -c %s \
// RUN:        -mabi=n32 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-N32 %s
// MIPS-ABI-N32: "-target-cpu" "mips64r2"
// MIPS-ABI-N32: "-target-abi" "n32"
//
// RUN: %clang --target=mips64-linux-gnu -### -c %s \
// RUN:        -mabi=64 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-64 %s
// MIPS-ABI-64: "-target-cpu" "mips64r2"
// MIPS-ABI-64: "-target-abi" "n64"
//
// RUN: %clang --target=mips64-linux-gnu -### -c %s \
// RUN:        -mabi=n64 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-N64 %s
// MIPS-ABI-N64: "-target-cpu" "mips64r2"
// MIPS-ABI-N64: "-target-abi" "n64"
//
// RUN: not %clang --target=mips64-linux-gnu -c %s \
// RUN:        -mabi=o64 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-O64 %s
// MIPS-ABI-O64: error: unknown target ABI 'o64'
//
// RUN: not %clang --target=mips-linux-gnu -c %s \
// RUN:        -mabi=unknown 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-UNKNOWN %s
// MIPS-ABI-UNKNOWN: error: unknown target ABI 'unknown'
//
// RUN: %clang --target=mips-linux-gnu -### -c %s \
// RUN:        -march=mips1 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-1 %s
// MIPS-ARCH-1: "-target-cpu" "mips1"
// MIPS-ARCH-1: "-target-abi" "o32"
//
// RUN: %clang --target=mips-linux-gnu -### -c %s \
// RUN:        -march=mips2 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-2 %s
// MIPS-ARCH-2: "-target-cpu" "mips2"
// MIPS-ARCH-2: "-target-abi" "o32"
//
// RUN: %clang --target=mips-linux-gnu -### -c %s \
// RUN:        -march=mips3 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-3 %s
// MIPS-ARCH-3: "-target-cpu" "mips3"
// MIPS-ARCH-3: "-target-abi" "o32"
//
// RUN: %clang --target=mips-linux-gnu -### -c %s \
// RUN:        -march=mips4 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-4 %s
// MIPS-ARCH-4: "-target-cpu" "mips4"
// MIPS-ARCH-4: "-target-abi" "o32"
//
// RUN: %clang --target=mips-linux-gnu -### -c %s \
// RUN:        -march=mips5 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-5 %s
// MIPS-ARCH-5: "-target-cpu" "mips5"
// MIPS-ARCH-5: "-target-abi" "o32"
//
// RUN: %clang --target=mips-linux-gnu -### -c %s \
// RUN:        -march=mips32 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-32 %s
// MIPS-ARCH-32: "-target-cpu" "mips32"
// MIPS-ARCH-32: "-target-abi" "o32"
//
// RUN: %clang --target=mips-linux-gnu -### -c %s \
// RUN:        -march=mips32r2 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-32R2 %s
// MIPS-ARCH-32R2: "-target-cpu" "mips32r2"
// MIPS-ARCH-32R2: "-target-abi" "o32"
//
// RUN: %clang --target=mips-linux-gnu -### -c %s \
// RUN:        -march=p5600 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-P5600 %s
// MIPS-ARCH-P5600: "-target-cpu" "p5600"
// MIPS-ARCH-P5600: "-target-abi" "o32"
//
// RUN: not %clang --target=mips-linux-gnu -c %s \
// RUN:        -march=p5600 -mabi=64 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-P5600-N64 %s
// MIPS-ARCH-P5600-N64: error: ABI 'n64' is not supported on CPU 'p5600'
//
// RUN: %clang --target=mips-linux-gnu -### -c %s \
// RUN:        -march=mips64 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-3264 %s
// MIPS-ARCH-3264: "-target-cpu" "mips64"
// MIPS-ARCH-3264: "-target-abi" "o32"
//
// RUN: %clang --target=mips64-linux-gnu -### -c %s \
// RUN:        -march=mips64 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-64 %s
// MIPS-ARCH-64: "-target-cpu" "mips64"
// MIPS-ARCH-64: "-target-abi" "n64"
//
// RUN: %clang --target=mips64-linux-gnu -### -c %s \
// RUN:        -march=mips64r2 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-64R2 %s
// MIPS-ARCH-64R2: "-target-cpu" "mips64r2"
// MIPS-ARCH-64R2: "-target-abi" "n64"
//
// RUN: %clang --target=mips64-linux-gnu -### -c %s \
// RUN:        -march=octeon 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-OCTEON %s
// MIPS-ARCH-OCTEON: "-target-cpu" "octeon"
// MIPS-ARCH-OCTEON: "-target-abi" "n64"
//
// RUN: %clang --target=mips64-linux-gnu -### -c %s \
// RUN:        -march=octeon+ 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-OCTEONP %s
// MIPS-ARCH-OCTEONP: "-target-cpu" "octeon+"
// MIPS-ARCH-OCTEONP: "-target-abi" "n64"
//
// RUN: not %clang --target=mips64-linux-gnu -c %s \
// RUN:        -march=mips32 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-6432 %s
// MIPS-ARCH-6432: error: ABI 'n64' is not supported on CPU 'mips32'
//
// RUN: not %clang --target=mips-linux-gnu -c %s \
// RUN:        -march=unknown 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ARCH-UNKNOWN %s
// MIPS-ARCH-UNKNOWN: error: unknown target CPU 'unknown'

// Check adjusting of target triple accordingly to `-mabi` option.
// RUN: %clang --target=mips64-linux-gnuabi64 -mabi=32 -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET-O32 %s
// TARGET-O32: "-triple" "mips-unknown-linux-gnu"
// TARGET-O32: "-target-cpu" "mips32r2"
// TARGET-O32: "-target-abi" "o32"
// TARGET-O32: ld{{(.exe)?}}"
// TARGET-O32: "-m" "elf32btsmip"

// RUN: %clang --target=mips-linux-gnu -mabi=n32 -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET-N32 %s
// TARGET-N32: "-triple" "mips64-unknown-linux-gnuabin32"
// TARGET-N32: "-target-cpu" "mips64r2"
// TARGET-N32: "-target-abi" "n32"
// TARGET-N32: ld{{(.exe)?}}"
// TARGET-N32: "-m" "elf32btsmipn32"

// RUN: %clang --target=mips-linux-gnu -mabi=64 -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET-N64 %s
// TARGET-N64: "-triple" "mips64-unknown-linux-gnuabi64"
// TARGET-N64: "-target-cpu" "mips64r2"
// TARGET-N64: "-target-abi" "n64"
// TARGET-N64: ld{{(.exe)?}}"
// TARGET-N64: "-m" "elf64btsmip"
