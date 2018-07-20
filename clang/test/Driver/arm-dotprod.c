// RUN: %clang -### -target arm %s 2>&1 | FileCheck %s --check-prefix=CHECK-NONE
// RUN: %clang -### -target arm -march=armv8.1a %s 2>&1 | FileCheck %s --check-prefix=CHECK-NONE
// RUN: %clang -### -target arm -march=armv8.2a %s 2>&1 | FileCheck %s --check-prefix=CHECK-NONE
// RUN: %clang -### -target arm -march=armv8.3a %s 2>&1 | FileCheck %s --check-prefix=CHECK-NONE
// CHECK-NONE-NOT: "-target-feature" "+dotprod"

// RUN: %clang -### -target arm-linux-eabi -march=armv8.2a+dotprod %s 2>&1 | FileCheck %s
// RUN: %clang -### -target arm-linux-eabi -march=armv8.3a+dotprod %s 2>&1 | FileCheck %s
// RUN: %clang -### -target arm-linux-eabi -mcpu=cortex-a75 %s 2>&1 | FileCheck %s
// RUN: %clang -### -target arm-linux-eabi -mcpu=cortex-a55 %s 2>&1 | FileCheck %s
// CHECK: "+dotprod"

// The following default to -msoft-float
// RUN: %clang -### -target arm -march=armv8.2a+dotprod %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-NO-DOTPROD
// RUN: %clang -### -target arm -march=armv8.3a+dotprod %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-NO-DOTPROD
// RUN: %clang -### -target arm -mcpu=cortex-a75 %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-NO-DOTPROD
// RUN: %clang -### -target arm -mcpu=cortex-a55 %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-NO-DOTPROD
// We rely on the backend disabling dotprod as it depends on neon, so check that
// neon is disabled after the dotprod was enabled.
// CHECK-NO-DOTPROD-NOT: "+dotprod"
