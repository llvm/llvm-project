// RUN: mkdir -p %T/compilation-database-test/include
// RUN: mkdir -p %T/compilation-database-test/a
// RUN: mkdir -p %T/compilation-database-test/b
// RUN: echo 'int *AA = 0;' > %T/compilation-database-test/a/a.cpp
// RUN: echo 'int *AB = 0;' > %T/compilation-database-test/a/b.cpp
// RUN: echo 'int *BB = 0;' > %T/compilation-database-test/b/b.cpp
// RUN: echo 'int *BC = 0;' > %T/compilation-database-test/b/c.cpp
// RUN: echo 'int *HP = 0;' > %T/compilation-database-test/include/header.h
// RUN: echo '#include "header.h"' > %T/compilation-database-test/b/d.cpp
// RUN: sed 's|test_dir|%/T/compilation-database-test|g' %S/Inputs/compilation-database/template.json > %T/compile_commands.json

// Regression test: shouldn't crash.
// RUN: not clang-tidy --checks=-*,modernize-use-nullptr -p %T %T/compilation-database-test/b/not-exist -header-filter=.* 2>&1 | FileCheck %s -check-prefix=CHECK-NOT-EXIST
// CHECK-NOT-EXIST: Error while processing {{.*[/\\]}}not-exist.
// CHECK-NOT-EXIST: unable to handle compilation
// CHECK-NOT-EXIST: Found compiler error

// RUN: clang-tidy --checks=-*,modernize-use-nullptr -p %T %T/compilation-database-test/a/a.cpp %T/compilation-database-test/a/b.cpp %T/compilation-database-test/b/b.cpp %T/compilation-database-test/b/c.cpp %T/compilation-database-test/b/d.cpp -header-filter=.* -fix
// RUN: FileCheck -input-file=%T/compilation-database-test/a/a.cpp %s -check-prefix=CHECK-FIX1
// RUN: FileCheck -input-file=%T/compilation-database-test/a/b.cpp %s -check-prefix=CHECK-FIX2
// RUN: FileCheck -input-file=%T/compilation-database-test/b/b.cpp %s -check-prefix=CHECK-FIX3
// RUN: FileCheck -input-file=%T/compilation-database-test/b/c.cpp %s -check-prefix=CHECK-FIX4
// RUN: FileCheck -input-file=%T/compilation-database-test/include/header.h %s -check-prefix=CHECK-FIX5

// CHECK-FIX1: int *AA = nullptr;
// CHECK-FIX2: int *AB = nullptr;
// CHECK-FIX3: int *BB = nullptr;
// CHECK-FIX4: int *BC = nullptr;
// CHECK-FIX5: int *HP = nullptr;
