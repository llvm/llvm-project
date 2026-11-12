// RUN: %clang_cc1 -Werror -pedantic %s -fixit-recompile -fixit-to-temporary -E -o - | FileCheck %s
// RUN: not %clang_cc1 -Werror -pedantic %s -fixit-recompile -fixit-to-temporary -fix-only-warnings

_Complex cd;

// CHECK: _Complex double cd;
