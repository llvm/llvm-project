// RUN: %clang -target arm-none-none-eabi -march=armv8m.main %s -### -c 2>&1 | FileCheck %s --check-prefix=CHECK-NOCDE
// CHECK-NOCDE: "-triple" "thumbv8m.main-none-none-eabi"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp0"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp1"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp2"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp3"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp4"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp5"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp6"
// CHECK-NOCDE-NOT: "-target-feature" "+cdecp7"

// RUN: %clang -target arm-none-none-eabi -march=armv8m.main+cdecp0+cdecp3 %s -### -c 2>&1 | FileCheck %s --check-prefix=CHECK-CDE1
// CHECK-CDE1: "-triple" "thumbv8m.main-none-none-eabi"
// CHECK-CDE1-DAG: "-target-feature" "+cdecp0"
// CHECK-CDE1-DAG: "-target-feature" "+cdecp3"

// RUN: %clang -target arm-none-none-eabi -march=armv8m.main+cdecp0+cdecp3 %s -### -c 2>&1 | FileCheck %s --check-prefix=CHECK-CDE2
// CHECK-CDE2: "-triple" "thumbv8m.main-none-none-eabi"
// CHECK-CDE2-NOT: "-target-feature" "+cdecp1"
// CHECK-CDE2-NOT: "-target-feature" "+cdecp2"
// CHECK-CDE2-NOT: "-target-feature" "+cdecp4"
// CHECK-CDE2-NOT: "-target-feature" "+cdecp5"
// CHECK-CDE2-NOT: "-target-feature" "+cdecp6"
// CHECK-CDE2-NOT: "-target-feature" "+cdecp7"
