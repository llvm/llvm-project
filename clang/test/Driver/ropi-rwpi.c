// RUN: %clang -target arm-none-eabi               -### -c %s 2>&1 | FileCheck --check-prefix=STATIC %s
// RUN: %clang -target arm-none-eabi -fropi        -### -c %s 2>&1 | FileCheck --check-prefix=ROPI %s
// RUN: %clang -target arm-none-eabi        -frwpi -### -c %s 2>&1 | FileCheck --check-prefix=RWPI %s
// RUN: %clang -target arm-none-eabi -fropi -frwpi -### -c %s 2>&1 | FileCheck --check-prefix=ROPI-RWPI %s

// RUN: %clang -target armeb-none-eabi   -fropi        -### -c %s 2>&1 | FileCheck --check-prefix=ROPI %s
// RUN: %clang -target thumb-none-eabi   -fropi        -### -c %s 2>&1 | FileCheck --check-prefix=ROPI %s
// RUN: %clang -target thumbeb-none-eabi -fropi        -### -c %s 2>&1 | FileCheck --check-prefix=ROPI %s

// RUN: %clang -target x86_64-linux-gnu -fropi        -### -c %s 2>&1 | FileCheck --check-prefix=ROPI-NON-ARM %s
// RUN: %clang -target x86_64-linux-gnu        -frwpi -### -c %s 2>&1 | FileCheck --check-prefix=RWPI-NON-ARM %s
// RUN: %clang -target x86_64-linux-gnu -fropi -frwpi -### -c %s 2>&1 | FileCheck --check-prefix=ROPI-NON-ARM --check-prefix=RWPI-NON-ARM %s

// RUN: %clang -target arm-none-eabi -fpic    -fropi        -### -c %s 2>&1 | FileCheck --check-prefix=PIC %s
// RUN: %clang -target arm-none-eabi -fpie           -frwpi -### -c %s 2>&1 | FileCheck --check-prefix=PIC %s
// RUN: %clang -target arm-none-eabi -fPIC    -fropi -frwpi -### -c %s 2>&1 | FileCheck --check-prefix=PIC %s
// RUN: %clang -target arm-none-eabi -fno-pic -fropi        -### -c %s 2>&1 | FileCheck --check-prefix=ROPI %s

// RUN: %clang -target arm-none-eabi -x c++ -fropi        -### -c %s 2>&1 | FileCheck --check-prefix=CXX %s
// RUN: %clang -target arm-none-eabi -x c++        -frwpi -### -c %s 2>&1 | FileCheck --check-prefix=RWPI %s
// RUN: %clang -target arm-none-eabi -x c++ -fropi -frwpi -### -c %s 2>&1 | FileCheck --check-prefix=CXX %s
// RUN: %clang -target arm-none-eabi -x c++ -fallow-unsupported -fropi        -### -c %s 2>&1 | FileCheck --check-prefix=ROPI %s


// STATIC: "-mrelocation-model" "static"

// ROPI: "-mrelocation-model" "ropi"

// RWPI: "-mrelocation-model" "rwpi"

// ROPI-RWPI: "-mrelocation-model" "ropi-rwpi"

// ROPI-NON-ARM: error: unsupported option '-fropi' for target 'x86_64-unknown-linux-gnu'
// RWPI-NON-ARM: error: unsupported option '-frwpi' for target 'x86_64-unknown-linux-gnu'

// PIC: error: embedded and GOT-based position independence are incompatible

// CXX: error: ROPI is not compatible with c++
