// RUN: touch %t.o

// RUN: %clang -target arm64-apple-tvos12.3 -isysroot %S/Inputs/iPhoneOS13.0.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -target x86_64-apple-tvos13-simulator -isysroot %S/Inputs/iPhoneOS13.0.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=SIMUL %s

// CHECK: "-platform_version" "tvos" "12.3.0" "13.0"
// SIMUL: "-platform_version" "tvos-simulator" "13.0.0" "13.0"
