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
// RUN: %clang %s -c -target powerpc-ibm-aix -save-temps -fembed-bitcode -fintegrated-as 2>&1 -### | FileCheck %s -check-prefix=CHECK-SAVE-TEMP
// CHECK-SAVE-TEMP: -cc1
// CHECK-SAVE-TEMP: -E
// CHECK-SAVE-TEMP: -cc1
// CHECK-SAVE-TEMP: -emit-llvm-bc
// CHECK-SAVE-TEMP: -cc1
// CHECK-SAVE-TEMP: -S
// CHECK-SAVE-TEMP: -fembed-bitcode=all
// CHECK-SAVE-TEMP: "{{.*}}as{{(.exe)?}}"

// RUN: %clang -c %s -flto -fembed-bitcode 2>&1 -### | FileCheck %s -check-prefix=CHECK-LTO
// RUN: %clang -c %s -flto=full -fembed-bitcode 2>&1 -### | FileCheck %s -check-prefix=CHECK-LTO
// RUN: %clang -c %s -flto=thin -fembed-bitcode 2>&1 -### | FileCheck %s -check-prefix=CHECK-LTO
// CHECK-LTO: -cc1
// CHECK-LTO: -emit-llvm-bc
// CHECK-LTO-NOT: warning: argument unused during compilation: '-fembed-bitcode'
// CHECK-LTO-NOT: -cc1
// CHECK-LTO-NOT: -fembed-bitcode=all
// RUN: touch %t.o
// RUN: %clang  -target armv7-apple-darwin -miphoneos-version-min=6.0 %t.o -fembed-bitcode  -fembed-bitcode-marker -mlinker-version=277  2>&1 -### | FileCheck %s -check-prefix=CHECK-LTO-MARKER-277
// RUN: %clang  -target armv7-apple-darwin -miphoneos-version-min=6.0 %t.o -fembed-bitcode  -fembed-bitcode-marker -mlinker-version=278  2>&1 -### | FileCheck %s -check-prefix=CHECK-LTO-MARKER-278
// CHECK-LTO-MARKER-277-NOT: bitcode_process_mode
// CHECK-LTO-MARKER-278: bitcode_process_mode



// RUN: %clang -c %s -fembed-bitcode-marker -fintegrated-as 2>&1 -### | FileCheck %s -check-prefix=CHECK-MARKER
// CHECK-MARKER: -cc1
// CHECK-MARKER: -emit-obj
// CHECK-MARKER: -fembed-bitcode=marker
// CHECK-MARKER-NOT: -cc1

// RUN: %clang -target armv7-apple-darwin -miphoneos-version-min=6.0 %s -fembed-bitcode=all -fintegrated-as 2>&1 -### | FileCheck %s -check-prefix=CHECK-LINKER
// RUN: %clang -target armv7-apple-darwin -miphoneos-version-min=6.0 %s -fembed-bitcode=marker -fintegrated-as 2>&1 -### | FileCheck %s -check-prefix=CHECK-LINKER
// RUN: %clang -target armv7-apple-darwin -miphoneos-version-min=6.0 %s -flto=full -fembed-bitcode=bitcode -fintegrated-as 2>&1 -### | FileCheck %s -check-prefix=CHECK-LINKER
// RUN: %clang -target armv7-apple-darwin -miphoneos-version-min=6.0 %s -flto=thin -fembed-bitcode=bitcode -fintegrated-as 2>&1 -### | FileCheck %s -check-prefix=CHECK-LINKER
// RUN: %clang -target armv7-apple-darwin -miphoneos-version-min=6.0 %s -fembed-bitcode=off -fintegrated-as 2>&1 -### | FileCheck %s -check-prefix=CHECK-NO-LINKER
// CHECK-LINKER: ld
// CHECK-LINKER: -bitcode_bundle
// CHECK-NO-LINKER-NOT: -bitcode_bundle

// RUN: not %clang -target armv7-apple-darwin -miphoneos-version-min=5.0 %s -fembed-bitcode -### 2>&1 | \
// RUN:   FileCheck %s -check-prefix=CHECK-PLATFORM-NOTSUPPORTED
// CHECK-PLATFORM-NOTSUPPORTED: -fembed-bitcode is not supported on versions of iOS prior to 6.0
