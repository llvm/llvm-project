// RUN: %clang -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-DEVIRT-SPECULATIVE %s
// RUN: %clang -fdevirtualize-speculatively -### %s 2>&1 | FileCheck -check-prefix=CHECK-DEVIRT-SPECULATIVE %s
// RUN: %clang -fno-devirtualize-speculatively -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-DEVIRT-SPECULATIVE %s

// CHECK-DEVIRT-SPECULATIVE: "-fdevirtualize-speculatively"
// CHECK-DEVIRT-SPECULATIVE: "-mllvm" "-devirtualize-speculatively"

// CHECK-NO-DEVIRT-SPECULATIVE-NOT: "-fdevirtualize-speculatively"
// CHECK-NO-DEVIRT-SPECULATIVE-NOT: "-mllvm" "-devirtualize-speculatively"
