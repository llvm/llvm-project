// -O option is only present in ld64 version 900 and later.

// RUN: %clang -target x86_64-apple-macos11 -### %s \
// RUN:   -mlinker-version=900 -O0 -fuse-ld= 2>&1 | FileCheck -check-prefixes=CHECK,O0 %s
// RUN: %clang -target arm64-apple-ios11 -### %s \
// RUN:   -mlinker-version=900 -O1 -fuse-ld= 2>&1 | FileCheck -check-prefixes=CHECK,O1 %s
// RUN: %clang -target x86_64-apple-macos11 -### %s \
// RUN:   -mlinker-version=900 -O2 -fuse-ld= 2>&1 | FileCheck -check-prefixes=CHECK,O2 %s
// RUN: %clang -target x86_64-apple-macos11 -### %s \
// RUN:   -mlinker-version=900 -O3 -fuse-ld= 2>&1 | FileCheck -check-prefixes=CHECK,O3 %s
// RUN: %clang -target x86_64-apple-macos11 -### %s \
// RUN:   -mlinker-version=900 -Oz -fuse-ld= 2>&1 | FileCheck -check-prefixes=CHECK,OZ %s

// RUN: %clang -target x86_64-apple-macos11 -### %s \
// RUN:   -mlinker-version=800 -O0 -fuse-ld= 2>&1 | FileCheck -check-prefix=NOOPT %s
// RUN: %clang -target x86_64-apple-macos11 -### %s \
// RUN:   -mlinker-version=900 -fuse-ld= 2>&1 | FileCheck -check-prefix=NOOPT %s

// CHECK: {{ld(.exe)?"}}
// O0-SAME: "-O0"
// O1-SAME: "-O1"
// O2-SAME: "-O2"
// O3-SAME: "-O3"
// OZ-SAME: "-Oz"

// NOOPT: {{ld(.exe)?"}}
// NOOPT-NOT: -O
