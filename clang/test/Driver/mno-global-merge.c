// RUN: %clang -target armv7-apple-darwin10 \
// RUN:   -mno-global-merge -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NGM < %t %s

// CHECK-NGM: "-mno-global-merge"

// RUN: %clang -target armv7-apple-darwin10 \
// RUN:   -mglobal-merge -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-GM < %t %s

// CHECK-GM-NOT: "-mglobal-merge"

