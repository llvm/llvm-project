// RUN: touch %t.o

// RUN: %clang -target arm64_32-apple-watchos5.2 -isysroot %S/Inputs/WatchOS6.0.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -target x86_64-apple-watchos6-simulator -isysroot %S/Inputs/WatchOS6.0.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=SIMUL %s

// CHECK: "-platform_version" "watchos" "5.2.0" "6.0.0"
// SIMUL: "-platform_version" "watchos-simulator" "6.0.0" "6.0.0"
