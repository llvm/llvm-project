// UNSUPPORTED: ps4, system-windows

// RUN: %clang -### -fcrash-diagnostics-dir=mydumps -c %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=OPTION
// OPTION: "-crash-diagnostics-dir=mydumps"
// RUN: %clang -### -c %s 2>&1 | FileCheck %s --check-prefix=NOOPTION
// NOOPTION-NOT: "-crash-diagnostics-dir

// RUN: %clang -### -fcrash-diagnostics-dir=mydumps -flto %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=LTO-OPTION
// LTO-OPTION: "-cc1"
// LTO-OPTION: "-plugin-opt=-crash-diagnostics-dir=mydumps"

// RUN: %clang -### -flto %s 2>&1 | FileCheck %s --check-prefix=LTO-NOOPTION
// LTO-NOOPTION: "-cc1"
// LTO-NOOPTION-NOT: "-plugin-opt=-crash-diagnostics-dir=mydumps"
