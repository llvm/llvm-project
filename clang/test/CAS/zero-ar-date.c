// Test that clang tells the linker to create reproducible executables, in
// spite of the Darwin executable's debug map having a field for each object
// file's timestamp.
//
// FIXME: Should be split up into various tests outside of clang/test/CAS.

// Needs 'shell' for touch and 'system-darwin' for calling through to the
// linker.
// REQUIRES: shell
// REQUIRES: system-darwin

// RUN: rm -rf %t
// RUN: mkdir %t

// Confirm "-greproducible" is passed through to -cc1.
// RUN: %clang -greproducible -target x86_64-apple-macos11.0 -x c -o %t/t.o -c %s -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-CC1
// CHECK-CC1: "-greproducible"

// Build an object file with debug info. Confirm the link command sets
// ZERO_AR_DATE to make the output reproducible.
// RUN: %clang -greproducible -target x86_64-apple-macos11.0 -x c -o %t/t.o -c %s
// RUN: %clang -greproducible -target x86_64-apple-macos11.0 -o %t/exec %t/t.o -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-LINK
// CHECK-LINK: env "ZERO_AR_DATE=1" "{{[^"]*}}/ld"

// Set the modtime to the start of January 1st and build an executable.
// RUN: touch -t 202001010000.00 %t/t.o
// RUN: %clang -greproducible -target x86_64-apple-macos11.0 -o %t/exec %t/t.o
// RUN: cp %t/exec %t/exec1

// Set the modtime to the end of January 31st and build an executable.
// RUN: touch -t 202012312359.59 %t/t.o
// RUN: %clang -greproducible -target x86_64-apple-macos11.0 -o %t/exec %t/t.o
// RUN: cp %t/exec %t/exec2

// Confirm the executables match.
// RUN: diff %t/exec1 %t/exec2

int main(void) { return 0; }
