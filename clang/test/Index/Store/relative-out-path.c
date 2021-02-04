// Needs 'find'.
// REQUIRES: shell

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang %s -index-store-path %t/idx1 -c -o %t/outfile.o
// RUN: cd %t
// RUN: %clang %s -index-store-path %t/idx2 -c -o outfile.o
// RUN: cd ..
// RUN: %clang %s -index-store-path %t/idx3 -fsyntax-only -o outfile.o -working-directory=%t
// RUN: diff -r -u %t/idx2 %t/idx3

// RUN: find %t/idx1 -name '*outfile.o*' > %t/hashes.txt
// RUN: find %t/idx3 -name '*outfile.o*' >> %t/hashes.txt
// RUN: FileCheck %s --input-file=%t/hashes.txt
// CHECK:      outfile.o[[OUT_HASH:.*$]]
// CHECK-NEXT: outfile.o[[OUT_HASH]]

// Repeat with custom unit output path:
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang %s -index-store-path %t/idx1 -c -o %t/outfile1.o -index-unit-output-path %t/custom.o
// RUN: cd %t
// RUN: %clang %s -index-store-path %t/idx2 -c -o outfile2.o -index-unit-output-path custom.o
// RUN: cd ..
// RUN: %clang %s -index-store-path %t/idx3 -fsyntax-only -o outfile3.o -working-directory=%t -index-unit-output-path custom.o
// RUN: diff -r -u %t/idx2 %t/idx3

// RUN: find %t/idx1 -name '*custom.o*' > %t/hashes.txt
// RUN: find %t/idx3 -name '*custom.o*' >> %t/hashes.txt
// RUN: FileCheck %s --input-file=%t/hashes.txt --check-prefix=CUSTOM
// CUSTOM:      custom.o[[OUT_HASH:.*$]]
// CUSTOM-NEXT: custom.o[[OUT_HASH]]

void foo();
