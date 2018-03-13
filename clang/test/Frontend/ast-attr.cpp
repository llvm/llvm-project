// RUN: %clang -emit-ast -o %t.ast %S/../Sema/attr-print.cpp
// RUN: %clang_cc1 %t.ast -ast-print | FileCheck %S/../Sema/attr-print.cpp

// %S/../Sema/attr-print.cpp exercises many different attributes, so we reuse
// it here to check -emit-ast for attributes.
