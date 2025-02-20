// UNSUPPORTED: xcselect
// CLANG_USE_XCSELECT will always have an SDK inferred.

// RUN: touch %t.o

// RUN: %clang -target x86_64-apple-macos10.13 -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=NOSDK %s
// RUN: %clang -target x86_64-apple-darwin17 -mlinker-version=520 \
// RUN:   -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=NOSDK %s
// NOSDK: "-platform_version" "macos" "10.13.0" "10.13.0"
