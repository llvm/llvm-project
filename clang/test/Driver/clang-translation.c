// RUN: %clang -target i386-unknown-unknown -### -S -O0 -Os %s -o %t.s -fverbose-asm -funwind-tables -fvisibility=hidden 2>&1 | FileCheck -check-prefix=I386 %s
// I386: "-triple" "i386-unknown-unknown"
// I386: "-S"
// I386: "-disable-free"
// I386: "-mrelocation-model" "static"
// I386: "-mdisable-fp-elim"
// I386: "-masm-verbose"
// I386: "-munwind-tables"
// I386: "-Os"
// I386: "-fvisibility"
// I386: "hidden"
// I386: "-o"
// I386: clang-translation
// RUN: %clang -target i386-apple-darwin9 -### -S %s -o %t.s 2>&1 | \
// RUN: FileCheck -check-prefix=YONAH %s
// YONAH: "-target-cpu"
// YONAH: "yonah"
// RUN: %clang -target x86_64-apple-darwin9 -### -S %s -o %t.s 2>&1 | \
// RUN: FileCheck -check-prefix=CORE2 %s
// CORE2: "-target-cpu"
// CORE2: "core2"

// RUN: %clang -target x86_64-apple-darwin10 -### -S %s -arch armv7 2>&1 | \
// RUN: FileCheck -check-prefix=ARMV7_DEFAULT %s
// ARMV7_DEFAULT: clang
// ARMV7_DEFAULT: "-cc1"
// ARMV7_DEFAULT-NOT: "-msoft-float"
// ARMV7_DEFAULT: "-mfloat-abi" "soft"
// ARMV7_DEFAULT-NOT: "-msoft-float"
// ARMV7_DEFAULT: "-x" "c"

// RUN: %clang -target x86_64-apple-darwin10 -### -S %s -arch armv7 \
// RUN: -msoft-float 2>&1 | FileCheck -check-prefix=ARMV7_SOFTFLOAT %s
// ARMV7_SOFTFLOAT: clang
// ARMV7_SOFTFLOAT: "-cc1"
// ARMV7_SOFTFLOAT: "-msoft-float"
// ARMV7_SOFTFLOAT: "-mfloat-abi" "soft"
// ARMV7_SOFTFLOAT: "-target-feature"
// ARMV7_SOFTFLOAT: "-neon"
// ARMV7_SOFTFLOAT: "-x" "c"

// RUN: %clang -target x86_64-apple-darwin10 -### -S %s -arch armv7 \
// RUN: -mhard-float 2>&1 | FileCheck -check-prefix=ARMV7_HARDFLOAT %s
// ARMV7_HARDFLOAT: clang
// ARMV7_HARDFLOAT: "-cc1"
// ARMV7_HARDFLOAT-NOT: "-msoft-float"
// ARMV7_HARDFLOAT: "-mfloat-abi" "hard"
// ARMV7_HARDFLOAT-NOT: "-msoft-float"
// ARMV7_HARDFLOAT: "-x" "c"

// RUN: %clang -target arm-linux -### -S %s -march=armv5e 2>&1 | \
// RUN: FileCheck -check-prefix=ARMV5E %s
// ARMV5E: clang
// ARMV5E: "-cc1"
// ARMV5E: "-target-cpu" "arm1022e"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=G5 2>&1 | FileCheck -check-prefix=PPCG5 %s
// PPCG5: clang
// PPCG5: "-cc1"
// PPCG5: "-target-cpu" "g5"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=power7 2>&1 | FileCheck -check-prefix=PPCPWR7 %s
// PPCPWR7: clang
// PPCPWR7: "-cc1"
// PPCPWR7: "-target-cpu" "pwr7"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s 2>&1 | FileCheck -check-prefix=PPC64NS %s
// PPC64NS: clang
// PPC64NS: "-cc1"
// PPC64NS: "-target-cpu" "ppc64"

// RUN: %clang -target powerpc-fsl-linux -### -S %s \
// RUN: -mcpu=e500mc 2>&1 | FileCheck -check-prefix=PPCE500MC %s
// PPCE500MC: clang
// PPCE500MC: "-cc1"
// PPCE500MC: "-target-cpu" "e500mc"

// RUN: %clang -target powerpc64-fsl-linux -### -S \
// RUN: %s -mcpu=e5500 2>&1 | FileCheck -check-prefix=PPCE5500 %s
// PPCE5500: clang
// PPCE5500: "-cc1"
// PPCE5500: "-target-cpu" "e5500"

// RUN: %clang -target amd64-unknown-openbsd5.2 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=AMD64 %s
// AMD64: clang
// AMD64: "-cc1"
// AMD64: "-triple"
// AMD64: "amd64-unknown-openbsd5.2"
// AMD64: "-munwind-tables"

// RUN: %clang -target amd64--mingw32 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=AMD64-MINGW %s
// AMD64-MINGW: clang
// AMD64-MINGW: "-cc1"
// AMD64-MINGW: "-triple"
// AMD64-MINGW: "amd64--mingw32"
// AMD64-MINGW: "-munwind-tables"

// RUN: %clang -target i386-linux-android -### -S %s 2>&1 \
// RUN:        --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=ANDROID-X86 %s
// ANDROID-X86: clang
// ANDROID-X86: "-target-cpu" "core2"
