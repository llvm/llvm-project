// REQUIRES: powerpc-registered-target
// RUN: %clang -### -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -mregnames \
// RUN:   %s 2>&1 >/dev/null | FileCheck %s --check-prefix=FULLNAMES
// RUN: %clang -### -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -mno-regnames \
// RUN:   %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOFULLNAMES

// FULLNAMES: -mregnames
// NOFULLNAMES-NOT: -mregnames
