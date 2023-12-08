// RUN: %clang_cc1 -emit-pch -o %t %S/line-directive-nofilename.h
// RUN: not %clang_cc1 -include-pch %t -fsyntax-only %s 2>&1 | FileCheck %s

// This causes an "error: redefinition" diagnostic. The notes will have the
// locations of the declarations from the PCH file.
double foo, bar;

// CHECK: line-directive-nofilename.h:42:5: note: previous definition is here
// CHECK: foobar.h:100:5: note: previous definition is here
