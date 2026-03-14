// RUN: %clang %s -target x86_64-apple-driverkit19.0 \
// RUN:   -isysroot %S/Inputs/DriverKit19.0.sdk -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-DEFAULT

// RUN: %clang %s -target x86_64-apple-driverkit19.0 -nodriverkitlib \
// RUN:   -isysroot %S/Inputs/DriverKit19.0.sdk -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-NO-DRIVERKIT

int main() { return 0; }

// CHECK-DEFAULT: "-framework" "DriverKit"

// CHECK-NO-DRIVERKIT-NOT: "-framework" "DriverKit"
