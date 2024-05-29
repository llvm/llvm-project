// RUN: %clang -fsyntax-only -Werror -xc %s
// RUN: %clang -fsyntax-only -Werror %s -xc %s

// RUN: %clang -fsyntax-only %s -xc++ -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -fsyntax-only -xc %s -xc++ -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -fsyntax-only %s -xc %s -xc++ -fsyntax-only 2>&1 | FileCheck %s
// CHECK: '-x c++' after last input file has no effect

// RUN: %clang_cl -fsyntax-only /WX -xc++ -- %s
