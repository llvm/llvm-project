// RUN: %clang -### %s 2>&1 | FileCheck %s -check-prefix=STACKPROBE
// RUN: %clang -### -mno-stack-arg-probe -mstack-arg-probe %s 2>&1 | FileCheck %s -check-prefix=STACKPROBE
// RUN: %clang -### -mstack-arg-probe -mno-stack-arg-probe %s 2>&1 | FileCheck %s -check-prefix=NO-STACKPROBE

// NO-STACKPROBE: -mno-stack-arg-probe
// STACKPROBE-NOT: -mno-stack-arg-probe
