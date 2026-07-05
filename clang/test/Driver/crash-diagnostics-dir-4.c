// RUN: %clang -target x86_64-linux-unknown -### -fcrash-diagnostics-dir=mydumps -flto %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=LTO-OPTION
// LTO-OPTION: "-cc1"
// LTO-OPTION: "-plugin-opt=-crash-diagnostics-dir=mydumps"

// RUN: %clang -target x86_64-linux-unknown -### -flto %s 2>&1 | FileCheck %s --check-prefix=LTO-NOOPTION
// LTO-NOOPTION: "-cc1"
// LTO-NOOPTION-NOT: "-plugin-opt=-crash-diagnostics-dir=mydumps"
