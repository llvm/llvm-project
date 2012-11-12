//  %clang_cc1 -remap-file "%s;%S/Inputs/remapped-file" -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-EXIST %s

// RUN: %clang_cc1 -x c++-header %s.h -emit-pch -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -remap-file "%s.h;%s.remap.h" -fsyntax-only 2>&1 | FileCheck %s

const char *str = STR;
int ge = zool;

// CHECK: file '{{.*[/\\]}}remap-file-from-pch.cpp.h' from the precompiled header has been overridden
// CHECK: use of undeclared identifier 'zool'
