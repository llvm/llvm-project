// RUN: %clang -target %itanium_abi_triple -### -c %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -target %itanium_abi_triple -fno-common -### -c %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -target %itanium_abi_triple -fno-common -fcommon -### -c %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=COMMON

// DEFAULT-NOT: "-fcommon"
// COMMON:      "-fcommon"
