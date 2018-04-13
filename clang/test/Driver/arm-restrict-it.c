// RUN: %clang -target arm-none-gnueabi -mrestrict-it -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-RESTRICTED < %t %s

// RUN: %clang -target armv8a-none-gnueabi -mrestrict-it -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-RESTRICTED < %t %s

// CHECK-RESTRICTED: "-mllvm" "-arm-restrict-it"

// RUN: %clang -target arm-none-gnueabi -mno-restrict-it -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-RESTRICTED < %t %s

// RUN: %clang -target armv8a-none-gnueabi -mno-restrict-it -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-RESTRICTED < %t %s

// CHECK-NO-RESTRICTED: "-mllvm" "-arm-no-restrict-it"
