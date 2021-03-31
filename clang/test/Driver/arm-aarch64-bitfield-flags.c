/// check -faapcs-bitfield-width/-fno-aapcs-bitfield-width
// RUN: %clang -target arm-arm-none-eabi -march=armv8-m.main -faapcs-bitfield-width -### %s 2>&1 | FileCheck --check-prefixes=WIDTH,INVERSE-WIDTH %s
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8-m.main -faapcs-bitfield-width -### %s 2>&1 | FileCheck --check-prefixes=WIDTH,INVERSE-WIDTH %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8-m.main -fno-aapcs-bitfield-width -### %s 2>&1 | FileCheck --check-prefixes=NO-WIDTH,WIDTH %s
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8-m.main -fno-aapcs-bitfield-width -### %s 2>&1 | FileCheck --check-prefixes=NO-WIDTH,WIDTH %s
// WIDTH-NOT: -faapcs-bitfield-width
// NO-WIDTH: -fno-aapcs-bitfield-width

/// check -faapcs-bitfield-load
// RUN: %clang -target arm-arm-none-eabi -march=armv8-m.main -faapcs-bitfield-load -### %s 2>&1 | FileCheck --check-prefix=LOAD %s
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8-m.main -faapcs-bitfield-load -### %s 2>&1 | FileCheck --check-prefix=LOAD %s
// LOAD: -faapcs-bitfield-load

/// check absence of the above argument when not given
// RUN: %clang -target arm-arm-none-eabi -march=armv8-m.main -### %s 2>&1 | FileCheck --check-prefixes=INVERSE-WIDTH,INVERSE-LOAD %s
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8-m.main -### %s 2>&1 | FileCheck --check-prefixes=INVERSE-WIDTH,INVERSE-LOAD %s
// INVERSE-WIDTH-NOT: -fno-aapcs-bitfield-width
// INVERSE-LOAD-NOT: -fno-aapcs-bitfield-load
