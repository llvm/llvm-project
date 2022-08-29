// REQUIRES: x86-registered-target
// REQUIRES: darwin
// FIXME: Breaks on non-macOS:
//        http://45.33.8.238/linux/85125/step_7.txt
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


// RUN: %clang %s -target x86_64-apple-driverkit19.0 -isysroot %S/Inputs/DriverKit19.0.sdk -E -v -x c++ 2>&1 | FileCheck %s --check-prefix=INC
// RUN: %clang %s -arch x86_64                       -isysroot %S/Inputs/DriverKit19.0.sdk -E -v -x c++ 2>&1 | FileCheck %s --check-prefix=INC
//
// INC:       -isysroot [[PATH:[^ ]*/Inputs/DriverKit19.0.sdk]]
// INC-LABEL: #include <...> search starts here:
// INC:       [[PATH]]/System/DriverKit/usr/local/include
// INC:       /lib/clang/{{[0-9\.]+}}/include
// INC:       [[PATH]]/System/DriverKit/usr/include
// INC:       [[PATH]]/System/DriverKit/System/Library/Frameworks (framework directory)
