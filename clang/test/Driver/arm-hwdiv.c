// Test that different values of -mhwdiv pick correct ARM hwdiv target-feature(s).

// RUN: %clang -### -target arm %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-target-feature" "+hwdiv"
// CHECK-DEFAULT-NOT: "-target-feature" "+hwdiv-arm"

// RUN: %clang -### -target arm %s -mhwdiv=arm -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM %s
// CHECK-ARM: "-target-feature" "+hwdiv-arm"
// CHECK-ARM: "-target-feature" "-hwdiv"

// RUN: %clang -### -target arm %s -mhwdiv=thumb -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-THUMB %s
// CHECK-THUMB: "-target-feature" "-hwdiv-arm"
// CHECK-THUMB: "-target-feature" "+hwdiv"

// RUN: %clang  -### -target arm %s -mhwdiv=arm,thumb -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM-THUMB %s
// CHECK-ARM-THUMB: "-target-feature" "+hwdiv-arm"
// CHECK-ARM-THUMB: "-target-feature" "+hwdiv"

// RUN: %clang  -### -target arm %s -mhwdiv=thumb,arm -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-THUMB-ARM %s
// CHECK-THUMB-ARM: "-target-feature" "+hwdiv-arm"
// CHECK-THUMB-ARM: "-target-feature" "+hwdiv"

// RUN: %clang -### -target arm %s -mhwdiv=none -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NONE %s
// CHECK-NONE: "-target-feature" "-hwdiv-arm"
// CHECK-NONE: "-target-feature" "-hwdiv"

// Also check the alternative syntax.

// RUN: %clang -### -target arm %s --mhwdiv arm -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ALT %s
// CHECK-ALT: "-target-feature" "+hwdiv-arm"
// CHECK-ALT: "-target-feature" "-hwdiv"

// RUN: %clang -### -target arm %s --mhwdiv=arm -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ALT-EQ %s
// CHECK-ALT-EQ: "-target-feature" "+hwdiv-arm"
// CHECK-ALT-EQ: "-target-feature" "-hwdiv"

