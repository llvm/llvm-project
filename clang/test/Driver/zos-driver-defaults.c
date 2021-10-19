// RUN: %clang -### --target=s390x-none-zos -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-SHORT-ENUMS %s
// RUN: %clang -### --target=s390x-none-zos -fno-short-enums -fsyntax-only %s 2>&1 | FileCheck %s
// REQUIRES: clang-driver

//CHECK-SHORT-ENUMS: -fshort-enums
//CHECK-SHORT-ENUMS: -fno-signed-char

//CHECK-NOT: -fshort-enums
//CHECK: -fno-signed-char
