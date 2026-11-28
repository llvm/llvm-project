// RUN: %clang -### --target=unknown-windows-unknown %s 2>&1 | FileCheck %s -check-prefix=STACKPROBE
// RUN: %clang -### -mno-stack-arg-probe -mstack-arg-probe --target=unknown-windows-unknown %s 2>&1 | FileCheck %s -check-prefix=STACKPROBE
// RUN: %clang -### -mstack-arg-probe -mno-stack-arg-probe --target=unknown-windows-unknown %s 2>&1 | FileCheck %s -check-prefix=NO-STACKPROBE
// RUN: not %clang -mno-stack-arg-probe --target=unknown-apple-darwin %s 2>&1 | FileCheck %s --check-prefix=ERROR

// NO-STACKPROBE: -mno-stack-arg-probe
// STACKPROBE-NOT: -mno-stack-arg-probe
// ERROR: stack-arg-probe flags are unsupported on Darwin platforms