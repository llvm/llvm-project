// RUN: touch %t.o

// RUN: %clang -target arm64-apple-ios12.3 -isysroot %S/Inputs/iPhoneOS13.0.sdk -mlinker-version=0 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=LINKER-OLD %s
// RUN: %clang -target arm64-apple-ios12.3 -isysroot %S/Inputs/iPhoneOS13.0.sdk -mlinker-version=400 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=LINKER-OLD %s
// RUN: %clang -target arm64-apple-ios12.3 -isysroot %S/Inputs/iPhoneOS13.0.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=LINKER-NEW  %s
// RUN: %clang -target x86_64-apple-ios13-simulator -isysroot %S/Inputs/iPhoneOS13.0.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=SIMUL %s

// LINKER-OLD: "-iphoneos_version_min" "12.3.0"
// LINKER-NEW: "-platform_version" "ios" "12.3.0" "13.0"
// SIMUL: "-platform_version" "ios-simulator" "13.0.0" "13.0"
