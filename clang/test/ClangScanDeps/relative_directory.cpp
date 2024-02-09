// UNSUPPORTED:  target={{.*}}-aix{{.*}}

// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: mkdir %t.dir/Inputs
// RUN: cp %s %t.dir/Inputs/relative_directory_input1.cpp
// RUN: cp %s %t.dir/Inputs/relative_directory_input2.cpp
// RUN: touch %t.dir/Inputs/header.h
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/relative_directory.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 | FileCheck --check-prefixes=CHECK1,CHECK2 %s

// The output order is non-deterministic when using more than one thread,
// so check the output using two runs.
// RUN: clang-scan-deps -compilation-database %t.cdb -j 2 | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-scan-deps -compilation-database %t.cdb -j 2 | FileCheck --check-prefix=CHECK2 %s

#include <header.h>

// CHECK1: relative_directory_input1.o:
// CHECK1-NEXT: relative_directory_input1.cpp
// CHECK1-NEXT: header.h

// CHECK2: relative_directory_input2.o:
// CHECK2-NEXT: relative_directory_input2.cpp
// CHECK2-NEXT: header.h
