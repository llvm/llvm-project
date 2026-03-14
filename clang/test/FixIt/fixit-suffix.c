// RUN: cp %s %t.extrasuffix
// RUN: %clang_cc1 -fixit=fixed -x c %t.extrasuffix
// RUN: %clang_cc1 -Werror -pedantic -x c %t.fixed.extrasuffix

_Complex cd;
