// RUN: %clang -target s390x-linux-gnu -### -c %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang -target s390x-linux-gnu -mno-unaligned-symbols -### -c %s 2>&1 | FileCheck -check-prefix=ALIGNED %s
// RUN: %clang -target s390x-linux-gnu -munaligned-symbols -### -c %s 2>&1 | FileCheck -check-prefix=UNALIGN %s

// DEFAULT-NOT: unaligned-symbols"
// ALIGNED: "-target-feature" "-unaligned-symbols"
// UNALIGN: "-target-feature" "+unaligned-symbols"
