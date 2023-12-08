// REQUIRES: x86-registered-target
// RUN: %clang %s -target x86_64-apple-driverkit19.0 -### 2>&1 | FileCheck %s -check-prefix=DEFAULT
// RUN: %clang %s -target x86_64-apple-driverkit19.0 -fexceptions -### 2>&1 | FileCheck %s -check-prefix=USERPROVIDED

int main() { return 0; }
// DEFAULT-NOT: "-fcxx-exceptions"
// DEFAULT-NOT: "-fexceptions"
// USERPROVIDED: "-fcxx-exceptions"
// USERPROVIDED: "-fexceptions"
