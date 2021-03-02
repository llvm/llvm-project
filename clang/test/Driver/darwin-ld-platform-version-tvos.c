// RUN: touch %t.o

// RUN: %clang -target arm64-apple-tvos12.3 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/iPhoneOS13.0.sdk -mlinker-version=0 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=LINKER-OLD %s
// RUN: %clang -target arm64-apple-tvos12.3 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/iPhoneOS13.0.sdk -mlinker-version=400 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=LINKER-OLD %s
// RUN: %clang -target arm64-apple-tvos12.3 -fuse-ld=lld.darwinnew \
// RUN:   -isysroot %S/Inputs/iPhoneOS13.0.sdk -mlinker-version=0 \
// RUN:   -### %t.o -B%S/Inputs/lld 2>&1 \
// RUN:   | FileCheck --check-prefix=LINKER-NEW %s
// RUN: %clang -target arm64-apple-tvos12.3 -fuse-ld= \
// RUN:   -isysroot %S/Inputs/iPhoneOS13.0.sdk -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=LINKER-NEW %s
// RUN: %clang -target x86_64-apple-tvos13-simulator -fuse-ld= \
// RUN:   -isysroot %S/Inputs/iPhoneOS13.0.sdk -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=SIMUL %s

// LINKER-OLD: "-tvos_version_min" "12.3.0"
// LINKER-NEW: "-platform_version" "tvos" "12.3.0" "13.0"
// SIMUL: "-platform_version" "tvos-simulator" "13.0.0" "13.0"
