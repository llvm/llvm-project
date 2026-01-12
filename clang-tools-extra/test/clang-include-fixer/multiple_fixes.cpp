// RUN: sed -e 's#//.*$##' %s > %t.cpp
// RUN: mkdir -p %t.dir/clang-include-fixer/multiple-fixes
// RUN: echo 'foo f;' > %t.dir/clang-include-fixer/multiple-fixes/foo.cpp
// RUN: echo 'bar b;' > %t.dir/clang-include-fixer/multiple-fixes/bar.cpp
// RUN: clang-include-fixer -db=fixed -input='foo= "foo.h";bar= "bar.h"' %t.dir/clang-include-fixer/multiple-fixes/*.cpp --
// RUN: FileCheck -input-file=%t.dir/clang-include-fixer/multiple-fixes/bar.cpp %s -check-prefix=CHECK-BAR
// RUN: FileCheck -input-file=%t.dir/clang-include-fixer/multiple-fixes/foo.cpp %s -check-prefix=CHECK-FOO
//
// CHECK-FOO: #include "foo.h"
// CHECK-FOO: foo f;
// CHECK-BAR: #include "bar.h"
// CHECK-BAR: bar b;
