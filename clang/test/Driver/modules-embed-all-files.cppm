// Tests that we'll enable -fmodules-embed-all-files for C++20 module units.

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang %t/a.cppm -### 2>&1 | FileCheck %t/a.cppm --check-prefix=PRE20
// RUN: %clang -std=c++20 %t/a.cppm -### 2>&1 | FileCheck %t/a.cppm

// RUN: %clang %t/a.cpp -### 2>&1 | FileCheck %t/a.cpp --check-prefix=NO-CXX-MODULE
// RUN: %clang -std=c++20 %t/a.cpp -### 2>&1 | FileCheck %t/a.cpp --check-prefix=NO-CXX-MODULE
// RUN: %clang -std=c++20 -x c++-module %t/a.cpp -### 2>&1 | FileCheck %t/a.cpp

//--- a.cppm

// PRE20-NOT: -fmodules-embed-all-files
// CHECK: -fmodules-embed-all-files

//--- a.cpp

// NO-CXX-MODULE-NOT: -fmodules-embed-all-files
// CHECK: -fmodules-embed-all-files
