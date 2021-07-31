// Enabling MS extensions should allow us to add BAR definitions.
// RUN: %clang_cc1 -DMSEXT -fms-extensions -DBAZ="\"Inputs/pch-through1.h\"" -emit-pch -o %t1.pch
// RUN: %clang_cc1 -DMSEXT -fms-extensions -include-pch %t1.pch -verify %s

#include BAZ
// expected-no-diagnostics
