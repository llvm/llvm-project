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

void foo();
