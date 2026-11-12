// RUN: %clang -### -fcrash-diagnostics-dir=mydumps -c %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=OPTION
// OPTION: "-crash-diagnostics-dir=mydumps"
// RUN: %clang -### -c %s 2>&1 | FileCheck %s --check-prefix=NOOPTION
// NOOPTION-NOT: "-crash-diagnostics-dir
