// RUN: touch %t.o

// RUN: %clang -target arm64_32-apple-watchos5.2 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/WatchOS6.0.sdk -mlinker-version=400 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=LINKER-OLD %s
// RUN: %clang -target arm64_32-apple-watchos5.2 -fuse-ld=lld \
// RUN:   -isysroot %S/Inputs/WatchOS6.0.sdk -mlinker-version=0 \
// RUN:   -### %t.o -B%S/Inputs/lld 2>&1 \
// RUN:   | FileCheck --check-prefix=LINKER-NEW %s
// RUN: %clang -target arm64_32-apple-watchos5.2 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/WatchOS6.0.sdk -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=LINKER-NEW %s
// RUN: %clang -target x86_64-apple-watchos6-simulator -fuse-ld= \
// RUN:   -isysroot %S/Inputs/WatchOS6.0.sdk -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=SIMUL %s

// RUN: %clang -target arm64-apple-watchos6.3 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/WatchOS6.0.sdk -mlinker-version=400 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64-LINKER-OLD %s

// RUN: %clang -target arm64e-apple-watchos6.3 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/WatchOS6.0.sdk -mlinker-version=400 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64-LINKER-OLD %s

// RUN: %clang -target arm64-apple-watchos26.1 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/WatchOS6.0.sdk -mlinker-version=400 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64-LINKER-OLD-261 %s

// RUN: %clang -target arm64-apple-watchos6.3 -fuse-ld=lld \
// RUN:   -isysroot %S/Inputs/WatchOS6.0.sdk -mlinker-version=0 \
// RUN:   -### %t.o -B%S/Inputs/lld 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64-LINKER-NEW %s

// RUN: %clang -target arm64e-apple-watchos6.3 -fuse-ld=lld \
// RUN:   -isysroot %S/Inputs/WatchOS6.0.sdk -mlinker-version=0 \
// RUN:   -### %t.o -B%S/Inputs/lld 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64-LINKER-NEW %s

// RUN: %clang -target arm64-apple-watchos6.3 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/WatchOS6.0.sdk -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64-LINKER-NEW %s

// RUN: %clang -target arm64-apple-watchos26.1 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/WatchOS6.0.sdk -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64-LINKER-NEW-261 %s

// RUN: %clang -target arm64-apple-watchos6-simulator -fuse-ld= \
// RUN:   -isysroot %S/Inputs/WatchOS6.0.sdk -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64-SIMUL %s

// LINKER-OLD: "-watchos_version_min" "5.2.0"
// LINKER-NEW: "-platform_version" "watchos" "5.2.0" "6.0"
// SIMUL: "-platform_version" "watchos-simulator" "6.0.0" "6.0"

// ARM64-LINKER-OLD: "-watchos_version_min" "26.0.0"
// ARM64-LINKER-OLD-261: "-watchos_version_min" "26.1.0"

// ARM64-LINKER-NEW: "-platform_version" "watchos" "26.0.0" "6.0"
// ARM64-LINKER-NEW-261: "-platform_version" "watchos" "26.1.0" "6.0"

// ARM64-SIMUL: "-platform_version" "watchos-simulator" "7.0.0" "6.0"
