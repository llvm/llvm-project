// Check the -fmax-type-align=N flag
// rdar://16254558
//
// RUN: %clang --target=x86_64-apple-macosx10.7.0 %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST0 %s
// TEST0: -fmax-type-align=16
// RUN: %clang -fmax-type-align=32 --target=x86_64-apple-macosx10.7.0 %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST1 %s
// TEST1: -fmax-type-align=32
// RUN: %clang -fmax-type-align=32 -fno-max-type-align --target=x86_64-apple-macosx10.7.0 %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST2 %s
// TEST2-NOT: -fmax-type-align
// RUN: %clang -fno-max-type-align --target=x86_64-apple-macosx10.7.0 %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST3 %s
// TEST3-NOT: -fmax-type-align
