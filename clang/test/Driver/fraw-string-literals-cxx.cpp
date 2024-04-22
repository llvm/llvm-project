// RUN: %clang -fraw-string-literals    -fsyntax-only -std=c++03   %s 2>&1 | FileCheck --check-prefix=CHECK-POS %s
// RUN: %clang -fraw-string-literals    -fsyntax-only -std=gnu++03 %s 2>&1 | FileCheck --check-prefix=CHECK-POS %s
// RUN: %clang -fno-raw-string-literals -fsyntax-only -std=c++03   %s 2>&1 | FileCheck --check-prefix=CHECK-NEG %s
// RUN: %clang -fno-raw-string-literals -fsyntax-only -std=gnu++03 %s 2>&1 | FileCheck --check-prefix=CHECK-NEG %s
// RUN: %clang -fraw-string-literals    -fsyntax-only -std=c++11   %s 2>&1 | FileCheck --check-prefix=CHECK-POS %s
// RUN: %clang -fraw-string-literals    -fsyntax-only -std=gnu++11 %s 2>&1 | FileCheck --check-prefix=CHECK-POS %s
// RUN: %clang -fno-raw-string-literals -fsyntax-only -std=c++11   %s 2>&1 | FileCheck --check-prefix=CHECK-NEG %s
// RUN: %clang -fno-raw-string-literals -fsyntax-only -std=gnu++11 %s 2>&1 | FileCheck --check-prefix=CHECK-NEG %s

// CHECK-POS: ignoring '-fraw-string-literals', which is only valid for C
// CHECK-NEG: ignoring '-fno-raw-string-literals', which is only valid for C
