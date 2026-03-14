// Test triple manipulations.

// RUN: %clang -### -c %s \
// RUN:   --target=i386-apple-darwin10 -mappletvsimulator-version-min=9.0 -arch x86_64 2>&1 | \
// RUN:   FileCheck %s -DARCH=x86_64 -DOS=tvos9.0.0-simulator
// RUN: %clang -### -c %s \
// RUN:   --target=armv7s-apple-darwin10 -mappletvos-version-min=9.0 -arch arm64 2>&1 | \
// RUN:   FileCheck %s -DARCH=arm64 -DOS=tvos9.0.0
// RUN: env TVOS_DEPLOYMENT_TARGET=9.0 %clang -### -c %s \
// RUN:   -isysroot SDKs/MacOSX10.9.sdk -target i386-apple-darwin10  -arch x86_64 2>&1 | \
// RUN:   FileCheck %s -DARCH=x86_64 -DOS=tvos9.0.0

// RUN: %clang -### -c %s \
// RUN:   --target=x86_64-apple-driverkit19.0 2>&1 | \
// RUN:   FileCheck %s -DARCH=x86_64 -DOS=driverkit19.0.0

// RUN: %clang -### -c %s \
// RUN:   --target=i386-apple-darwin10 -miphonesimulator-version-min=7.0 -arch i386 2>&1 | \
// RUN:   FileCheck %s -DARCH=i386 -DOS=ios7.0.0-simulator
// RUN: %clang -### -c %s \
// RUN:   --target=armv7s-apple-darwin10 -miphoneos-version-min=7.0 -arch armv7s 2>&1 | \
// RUN:   FileCheck %s -DARCH=thumbv7s -DOS=ios7.0.0

// RUN: %clang -### -c %s \
// RUN:   --target=i386-apple-darwin10 -mwatchsimulator-version-min=2.0 -arch i386 2>&1 | \
// RUN:   FileCheck %s -DARCH=i386 -DOS=watchos2.0.0-simulator
// RUN: %clang -### -c %s \
// RUN:   --target=armv7s-apple-darwin10 -mwatchos-version-min=2.0 -arch armv7k 2>&1 | \
// RUN:   FileCheck %s -DARCH=thumbv7k -DOS=watchos2.0.0

// CHECK: "-cc1" "-triple" "[[ARCH]]-apple-[[OS]]"
