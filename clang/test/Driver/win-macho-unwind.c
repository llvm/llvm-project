// RUN: %clang -target x86_64-pc-win32-macho -### -S %s -o %t.s 2>&1 | FileCheck %s
 
// Do not add function attribute "uwtable" for macho targets.
// CHECK-NOT: -funwind-tables=2
