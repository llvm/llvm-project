// RUN: %clang -x c++ -### -c -target arm64-apple-ios -fno-rtti %s 2>&1 | FileCheck -check-prefix=WARN %s
// RUN: %clang -x c++ -### -c -target x86_64-apple-macosx -fno-rtti %s 2>&1 | FileCheck -check-prefix=WARN %s
// RUN: %clang -x c++ -### -c -target arm64-apple-ios %s 2>&1 | FileCheck -check-prefix=OK %s
// RUN: %clang -x c++ -### -c -target x86_64-apple-macosx %s 2>&1 | FileCheck -check-prefix=OK %s
// RUN: %clang -x c++ -### -c -target x86_64-linux-gnu -fexceptions -fno-rtti %s 2>&1 | FileCheck -check-prefix=OK %s
// RUN: %clang -x c++ -### -c -target x86_64-pc-windows -fexceptions -fno-rtti %s 2>&1 | FileCheck -check-prefix=OK %s

// WARN: warning: using exceptions without RTTI is not recommended
// OK-NOT: {{warning:|error:}}
