// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: rm -rf %t.module-cache
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/modules_cdb_input.cpp
// RUN: cp %s %t.dir/modules_cdb_input2.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: cp %S/Inputs/header2.h %t.dir/Inputs/header2.h
// RUN: cp %S/Inputs/module.modulemap %t.dir/Inputs/module.modulemap
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/modules_cdb.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -mode preprocess-minimized-sources | \
// RUN:   FileCheck --check-prefixes=CHECK1,CHECK2,CHECK2NO %s
//
// The output order is non-deterministic when using more than one thread,
// so check the output using two runs. Note that the 'NOT' check is not used
// as it might fail if the results for `modules_cdb_input.cpp` are reported before
// `modules_cdb_input2.cpp`.
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 2 -mode preprocess-minimized-sources | \
// RUN:   FileCheck --check-prefix=CHECK1 %s
// RUN: clang-scan-deps -compilation-database %t.cdb -j 2 -mode preprocess | \
// RUN:   FileCheck --check-prefix=CHECK1 %s
// RUN: clang-scan-deps -compilation-database %t.cdb -j 2 -mode preprocess-minimized-sources | \
// RUN:   FileCheck --check-prefix=CHECK2 %s
// RUN: clang-scan-deps -compilation-database %t.cdb -j 2 -mode preprocess | \
// RUN:   FileCheck --check-prefix=CHECK2 %s

#include "header.h"

// CHECK1: modules_cdb_input2.cpp
// CHECK1-NEXT: modules_cdb_input2.cpp
// CHECK1-NEXT: Inputs{{/|\\}}module.modulemap
// CHECK1-NEXT: Inputs{{/|\\}}header2.h
// CHECK1: Inputs{{/|\\}}header.h

// CHECK2: modules_cdb_input.cpp
// CHECK2-NEXT: Inputs{{/|\\}}module.modulemap
// CHECK2-NEXT: Inputs{{/|\\}}header.h
// CHECK2NO-NOT: header2
