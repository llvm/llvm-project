// RUN: %clang -target arm-none-none-eabi -march=armv8m.main %s -### -c 2>&1 | FileCheck %s --check-prefixes=CHECK-NOCDE,CHECK-NOCDE-V8
// RUN: %clang -target arm-none-none-eabi -march=armv8.1m.main %s -### -c 2>&1 | FileCheck %s --check-prefixes=CHECK-NOCDE,CHECK-NOCDE-V81
// CHECK-NOCDE-V8: "-triple" "thumbv8m.main-none-none-eabi"
// CHECK-NOCDE-V81: "-triple" "thumbv8.1m.main-none-none-eabi"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp0"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp1"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp2"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp3"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp4"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp5"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp6"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp7"

// RUN: %clang -target arm-none-none-eabi -march=armv8m.main+cdecp0+cdecp3 %s -### -c 2>&1 | FileCheck %s --check-prefixes=CHECK-CDE1,CHECK-CDE1-V8
// RUN: %clang -target arm-none-none-eabi -march=armv8.1m.main+cdecp0+cdecp3 %s -### -c 2>&1 | FileCheck %s --check-prefixes=CHECK-CDE1,CHECK-CDE1-V81
// RUN: %clang -target arm-none-none-eabi -march=armv8.1m.main+mve.fp+cdecp0+cdecp3 %s -### -c 2>&1 | FileCheck %s --check-prefixes=CHECK-CDE1,CHECK-CDE1-V81MVE
// CHECK-CDE1-V8: "-triple" "thumbv8m.main-none-none-eabi"
// CHECK-CDE1-V81: "-triple" "thumbv8.1m.main-none-none-eabi"
// CHECK-CDE1-V81MVE: "-triple" "thumbv8.1m.main-none-none-eabi"
// CHECK-CDE1-V81MVE-DAG: "-target-feature" "+mve.fp"
// CHECK-CDE1-DAG: "-target-feature" "+cdecp0"
// CHECK-CDE1-DAG: "-target-feature" "+cdecp3"

// RUN: %clang -target arm-none-none-eabi -march=armv8m.main+cdecp0+cdecp3 %s -### -c 2>&1 | FileCheck %s --check-prefixes=CHECK-CDE2,CHECK-CDE2-V8
// RUN: %clang -target arm-none-none-eabi -march=armv8.1m.main+cdecp0+cdecp3 %s -### -c 2>&1 | FileCheck %s --check-prefixes=CHECK-CDE2,CHECK-CDE2-V81
// CHECK-CDE2-V8: "-triple" "thumbv8m.main-none-none-eabi"
// CHECK-CDE2-V81: "-triple" "thumbv8.1m.main-none-none-eabi"
// CHECK-CDE2-NOT: "-target-feature" "+cdecp1"
// CHECK-CDE2-NOT: "-target-feature" "+cdecp2"
// CHECK-CDE2-NOT: "-target-feature" "+cdecp4"
// CHECK-CDE2-NOT: "-target-feature" "+cdecp5"
// CHECK-CDE2-NOT: "-target-feature" "+cdecp6"
// CHECK-CDE2-NOT: "-target-feature" "+cdecp7"
