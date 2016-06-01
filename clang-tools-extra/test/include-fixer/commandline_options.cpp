// REQUIRES: shell
// RUN: sed -e 's#//.*$##' %s > %t.cpp
// RUN: clang-include-fixer -db=fixed -input='foo= "foo.h","bar.h"' -output-headers %t.cpp -- | FileCheck %s -check-prefix=CHECK-HEADERS
// RUN: cat %t.cpp | clang-include-fixer -stdin -insert-header='{SymbolIdentifier: foo, Headers: ["\"foo.h\""]}' %t.cpp | FileCheck %s -check-prefix=CHECK
//
// CHECK-HEADERS: "Headers": [ "\"foo.h\"", "\"bar.h\"" ]
//
// CHECK: #include "foo.h"
// CHECK: foo f;

foo f;
