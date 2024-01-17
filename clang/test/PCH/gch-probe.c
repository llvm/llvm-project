// For GCC compatibility, clang should probe also with the .gch extension.
// RUN: %clang -x c-header -c %s -o %t.h.gch
// RUN: %clang -fsyntax-only -include %t.h %s

// -gmodules embeds the Clang AST file in an object file.
// RUN: %clang -x c-header -c %s -gmodules -o %t.h.gch
// RUN: %clang -fsyntax-only -include %t.h %s

// gch probing should ignore files which are not clang pch files.
// RUN: %clang -fsyntax-only -include %S/Inputs/gch-probe.h %s 2>&1 | FileCheck %s
// CHECK: warning: precompiled header '{{.*}}gch-probe.h.gch' was ignored because it is not a clang PCH file

void f(void);
