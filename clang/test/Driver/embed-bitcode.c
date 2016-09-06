// RUN: %clang -ccc-print-bindings -c %s -fembed-bitcode 2>&1 | FileCheck %s
// CHECK: clang
// CHECK: clang

// RUN: %clang %s -c -fembed-bitcode -fintegrated-as 2>&1 -### | FileCheck %s -check-prefix=CHECK-CC
// CHECK-CC: -cc1
// CHECK-CC: -emit-llvm-bc
// CHECK-CC: -cc1
// CHECK-CC: -emit-obj
// CHECK-CC: -fembed-bitcode=all

// RUN: %clang %s -c -fembed-bitcode=bitcode -fintegrated-as 2>&1 -### | FileCheck %s -check-prefix=CHECK-BITCODE
// CHECK-BITCODE: -cc1
// CHECK-BITCODE: -emit-llvm-bc
// CHECK-BITCODE: -cc1
// CHECK-BITCODE: -emit-obj
// CHECK-BITCODE: -fembed-bitcode=bitcode
//
// RUN: %clang %s -c -save-temps -fembed-bitcode -fintegrated-as 2>&1 -### | FileCheck %s -check-prefix=CHECK-SAVE-TEMP
// CHECK-SAVE-TEMP: -cc1
// CHECK-SAVE-TEMP: -E
// CHECK-SAVE-TEMP: -cc1
// CHECK-SAVE-TEMP: -emit-llvm-bc
// CHECK-SAVE-TEMP: -cc1
// CHECK-SAVE-TEMP: -S
// CHECK-SAVE-TEMP: -fembed-bitcode=all
// CHECK-SAVE-TEMP: -cc1as

// RUN: %clang -c %s -flto -fembed-bitcode 2>&1 -### | FileCheck %s -check-prefix=CHECK-LTO
// RUN: %clang -c %s -flto=full -fembed-bitcode 2>&1 -### | FileCheck %s -check-prefix=CHECK-LTO
// RUN: %clang -c %s -flto=thin -fembed-bitcode 2>&1 -### | FileCheck %s -check-prefix=CHECK-LTO
// CHECK-LTO: -cc1
// CHECK-LTO: -emit-llvm-bc
// CHECK-LTO-NOT: warning: argument unused during compilation: '-fembed-bitcode'
// CHECK-LTO-NOT: -cc1
// CHECK-LTO-NOT: -fembed-bitcode=all

// RUN: %clang -c %s -fembed-bitcode-marker -fintegrated-as 2>&1 -### | FileCheck %s -check-prefix=CHECK-MARKER
// CHECK-MARKER: -cc1
// CHECK-MARKER: -emit-obj
// CHECK-MARKER: -fembed-bitcode=marker
// CHECK-MARKER-NOT: -cc1

