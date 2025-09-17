// UNSUPPORTED: system-windows
//   Windows is unsupported because we use the Unix path separator `\`.

// RUN: %clang %s -target x86_64-apple-driverkit19.0 -mlinker-version=0 \
// RUN:   -isysroot %S/Inputs/DriverKit19.0.sdk -### 2>&1               \
// RUN: | FileCheck %s --check-prefix=LD64-OLD
// RUN: %clang %s -target x86_64-apple-driverkit19.0 -mlinker-version=604.99 \
// RUN:   -isysroot %S/Inputs/DriverKit19.0.sdk -### 2>&1                    \
// RUN: | FileCheck %s --check-prefix=LD64-OLD
// RUN: %clang %s -target x86_64-apple-driverkit19.0 -mlinker-version=605.0 \
// RUN:   -isysroot %S/Inputs/DriverKit19.0.sdk -### 2>&1                   \
// RUN: | FileCheck %s --check-prefix=LD64-OLD
// RUN: %clang %s -target x86_64-apple-driverkit19.0 -mlinker-version=605.1 \
// RUN:   -isysroot %S/Inputs/DriverKit19.0.sdk -### 2>&1                   \
// RUN: | FileCheck %s --check-prefix=LD64-NEW

int main() { return 0; }
// LD64-OLD: "-isysroot" "[[PATH:[^"]*]]Inputs/DriverKit19.0.sdk"
// LD64-OLD: "-L[[PATH]]Inputs/DriverKit19.0.sdk/System/DriverKit/usr/lib"
// LD64-OLD: "-F[[PATH]]Inputs/DriverKit19.0.sdk/System/DriverKit/System/Library/Frameworks"
// LD64-NEW: "-isysroot" "[[PATH:[^"]*]]Inputs/DriverKit19.0.sdk"
// LD64-NEW-NOT: "-L[[PATH]]Inputs/DriverKit19.0.sdk/System/DriverKit/usr/lib"
// LD64-NEW-NOT: "-F[[PATH]]Inputs/DriverKit19.0.sdk/System/DriverKit/System/Library/Frameworks"
// LD64-NEW-NOT: "-F[[PATH]]Inputs/DriverKit19.0.sdk/System/DriverKit/System/Library/SubFrameworks"


// RUN: %clang %s -target x86_64-apple-driverkit19.0 -isysroot %S/Inputs/DriverKit19.0.sdk -x c++ -### 2>&1 \
// RUN: | FileCheck %s -DSDKROOT=%S/Inputs/DriverKit19.0.sdk --check-prefix=INC
//
// INC: "-isysroot" "[[SDKROOT]]"
// INC: "-internal-isystem" "[[SDKROOT]]/System/DriverKit/usr/local/include"
// INC: "-internal-isystem" "{{.+}}/lib{{(64)?}}/clang/{{[^/ ]+}}/include"
// INC: "-internal-externc-isystem" "[[SDKROOT]]/System/DriverKit/usr/include"
// INC: "-internal-iframework" "[[SDKROOT]]/System/DriverKit/System/Library/Frameworks"
// INC: "-internal-iframework" "[[SDKROOT]]/System/DriverKit/System/Library/SubFrameworks"
