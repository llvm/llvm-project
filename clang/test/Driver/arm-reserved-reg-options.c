// ## FP ARM + Thumb
// RUN: %clang -target arm-arm-none-eabi -### -ffixed-r11 -c %s 2>&1 | FileCheck -check-prefix=CHECK-ERROR-R11 %s
// RUN: %clang -target arm-arm-none-eabi -### -ffixed-r7 -c %s 2>&1 | FileCheck -check-prefix=CHECK-NO-ERROR %s

// RUN: %clang -target arm-arm-none-eabi -### -ffixed-r7 -mthumb -c %s 2>&1 | FileCheck -check-prefix=CHECK-ERROR-R7 %s
// RUN: %clang -target arm-arm-none-eabi -### -ffixed-r11 -mthumb -c %s 2>&1 | FileCheck -check-prefix=CHECK-NO-ERROR %s

// RUN: %clang -target thumbv6m-none-eabi -### -ffixed-r7 -c %s 2>&1 | FileCheck -check-prefix=CHECK-ERROR-R7 %s
// RUN: %clang -target thumbv6m-none-eabi -### -ffixed-r11 -c %s 2>&1 | FileCheck -check-prefix=CHECK-NO-ERROR %s

// ## FP Darwin (R7)
// RUN: %clang -target armv6-apple-darwin9 -### -ffixed-r7 -c %s 2>&1 | FileCheck -check-prefix=CHECK-ERROR-R7 %s
// RUN: %clang -target armv6-apple-darwin9 -### -ffixed-r11 -c %s 2>&1 | FileCheck -check-prefix=CHECK-NO-ERROR %s

// RUN: %clang -target armv6-apple-ios3 -### -ffixed-r7 -c %s 2>&1 | FileCheck -check-prefix=CHECK-ERROR-R7 %s
// RUN: %clang -target armv6-apple-ios3 -### -ffixed-r11 -c %s 2>&1 | FileCheck -check-prefix=CHECK-NO-ERROR %s

// RUN: %clang -target armv7s-apple-darwin10 -### -ffixed-r7 -c %s 2>&1 | FileCheck -check-prefix=CHECK-ERROR-R7 %s
// RUN: %clang -target armv7s-apple-darwin10 -### -ffixed-r11 -c %s 2>&1 | FileCheck -check-prefix=CHECK-NO-ERROR %s

// ## FP Windows (R11)
// RUN: %clang -target armv7-windows -### -ffixed-r11 -c %s 2>&1 | FileCheck -check-prefix=CHECK-ERROR-R11 %s
// RUN: %clang -target armv7-windows -### -ffixed-r7 -c %s 2>&1 | FileCheck -check-prefix=CHECK-NO-ERROR %s

// ## FRWPI (R9)
// RUN: %clang -target arm-arm-none-eabi -### -frwpi -ffixed-r9 -c %s 2>&1 | FileCheck -check-prefix=CHECK-RESERVED-FRWPI-CONFLICT %s
// RUN: %clang -target arm-arm-none-eabi -### -ffixed-r9 -c %s 2>&1 | FileCheck -check-prefix=CHECK-RESERVED-FRWPI-VALID %s
// RUN: %clang -target arm-arm-none-eabi -### -frwpi -c %s 2>&1 | FileCheck -check-prefix=CHECK-RESERVED-FRWPI-VALID %s

// CHECK-ERROR-R11: error: '-ffixed-r11' has been specified but 'r11' is used as the frame pointer for this target
// CHECK-ERROR-R7: error: '-ffixed-r7' has been specified but 'r7' is used as the frame pointer for this target
// CHECK-NO-ERROR-NOT: may still be used as a frame pointer

// CHECK-RESERVED-FRWPI-CONFLICT: option '-ffixed-r9' cannot be specified with '-frwpi'
// CHECK-RESERVED-FRWPI-VALID-NOT: option '-ffixed-r9' cannot be specified with '-frwpi'
