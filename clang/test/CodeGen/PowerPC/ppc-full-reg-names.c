// REQUIRES: powerpc-registered-target
// RUN: %clang -### -target powerpc-ibm-aix-xcoff -mcpu=pwr8 -O3 -mregnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=FULLNAMES
// RUN: %clang -### -target powerpc64-ibm-aix-xcoff -mcpu=pwr8 -O3 -mregnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=FULLNAMES
// RUN: %clang -### -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -O3 -mregnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=FULLNAMES
// RUN: %clang -### -target powerpc-ibm-aix-xcoff -mcpu=pwr8 -O3 -mno-regnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=NOFULLNAMES
// RUN: %clang -### -target powerpc64-ibm-aix-xcoff -mcpu=pwr8 -O3 -mno-regnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=NOFULLNAMES
// RUN: %clang -### -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -O3 -mno-regnames \
// RUN:   -maltivec %s -o - | FileCheck %s --check-prefix=NOFULLNAMES

// FULLNAMES: clang
// FULLNAMES-SAME: -mregnames
// NOFULLNAMES: clang
// NOFULLNAMES-SAME-NOT: -mregnames


