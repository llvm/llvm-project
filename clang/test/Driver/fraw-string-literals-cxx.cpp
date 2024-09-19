// RUN: %clang -fraw-string-literals    -fsyntax-only -std=c++03   %s 2>&1 | FileCheck --check-prefix=CHECK-PRE-CXX11 --allow-empty %s
// RUN: %clang -fraw-string-literals    -fsyntax-only -std=gnu++03 %s 2>&1 | FileCheck --check-prefix=CHECK-PRE-CXX11 --allow-empty %s
// RUN: %clang -fno-raw-string-literals -fsyntax-only -std=c++03   %s 2>&1 | FileCheck --check-prefix=CHECK-PRE-CXX11 --allow-empty %s
// RUN: %clang -fno-raw-string-literals -fsyntax-only -std=gnu++03 %s 2>&1 | FileCheck --check-prefix=CHECK-PRE-CXX11 --allow-empty %s
// RUN: %clang -fraw-string-literals    -fsyntax-only -std=c++11   %s 2>&1 | FileCheck --check-prefix=CHECK-POS %s
// RUN: %clang -fraw-string-literals    -fsyntax-only -std=gnu++11 %s 2>&1 | FileCheck --check-prefix=CHECK-POS %s
// RUN: %clang -fno-raw-string-literals -fsyntax-only -std=c++11   %s 2>&1 | FileCheck --check-prefix=CHECK-NEG %s
// RUN: %clang -fno-raw-string-literals -fsyntax-only -std=gnu++11 %s 2>&1 | FileCheck --check-prefix=CHECK-NEG %s
// RUN: %clang -fraw-string-literals    -fsyntax-only -std=c++20   %s 2>&1 | FileCheck --check-prefix=CHECK-POS %s
// RUN: %clang -fraw-string-literals    -fsyntax-only -std=gnu++20 %s 2>&1 | FileCheck --check-prefix=CHECK-POS %s
// RUN: %clang -fno-raw-string-literals -fsyntax-only -std=c++20   %s 2>&1 | FileCheck --check-prefix=CHECK-NEG %s
// RUN: %clang -fno-raw-string-literals -fsyntax-only -std=gnu++20 %s 2>&1 | FileCheck --check-prefix=CHECK-NEG %s

// CHECK-PRE-CXX11-NOT: ignoring '-fraw-string-literals'
// CHECK-PRE-CXX11-NOT: ignoring '-fno-raw-string-literals'
// CHECK-POS: ignoring '-fraw-string-literals', which is only valid for C and C++ standards before C++11
// CHECK-NEG: ignoring '-fno-raw-string-literals', which is only valid for C and C++ standards before C++11
