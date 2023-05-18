// RUN: %clang -fsyntax-only -Werror -xc %s
// RUN: %clang -fsyntax-only -Werror %s -xc %s

// RUN: %clang -fsyntax-only %s -xc++ -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -fsyntax-only -xc %s -xc++ -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -fsyntax-only %s -xc %s -xc++ -fsyntax-only 2>&1 | FileCheck %s
// CHECK: '-x c++' after last input file has no effect

// RUN: not %clang_cl /WX /clang:-xc /clang:-E /clang:-dM -- %s 2>&1 | FileCheck --implicit-check-not="error:" -check-prefix=CL %s
// RUN: not %clang_cl /TC /WX /clang:-xc /clang:-E /clang:-dM -- %s 2>&1 | FileCheck --implicit-check-not="error:" -check-prefix=CL %s
// CL: error: unsupported option '-x c'; did you mean '/TC' or '/TP'?
