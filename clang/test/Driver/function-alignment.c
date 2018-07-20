// RUN: %clang -### %s 2>&1 | FileCheck %s -check-prefix CHECK-0
// RUN: %clang -### -falign-functions %s 2>&1 | FileCheck %s -check-prefix CHECK-1
// RUN: %clang -### -falign-functions=1 %s 2>&1 | FileCheck %s -check-prefix CHECK-1
// RUN: %clang -### -falign-functions=2 %s 2>&1 | FileCheck %s -check-prefix CHECK-2
// RUN: %clang -### -falign-functions=3 %s 2>&1 | FileCheck %s -check-prefix CHECK-3
// RUN: %clang -### -falign-functions=4 %s 2>&1 | FileCheck %s -check-prefix CHECK-4
// RUN: %clang -### -falign-functions=65537 %s 2>&1 | FileCheck %s -check-prefix CHECK-ERR-65537
// RUN: %clang -### -falign-functions=a %s 2>&1 | FileCheck %s -check-prefix CHECK-ERR-A

// CHECK-0-NOT: "-function-alignment"
// CHECK-1-NOT: "-function-alignment"
// CHECK-2: "-function-alignment" "1"
// CHECK-3: "-function-alignment" "2"
// CHECK-4: "-function-alignment" "2"
// CHECK-ERR-65537: error: invalid integral value '65537' in '-falign-functions=65537'
// CHECK-ERR-A: error: invalid integral value 'a' in '-falign-functions=a'

