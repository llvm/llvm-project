// Test without PCH
// RUN: %clang_cc1 %s -include %S/reinclude1.h -include %S/reinclude2.h -fsyntax-only -verify

// RUN: %clang_cc1 -x c++-header %S/reinclude1.h -emit-pch -o %t1
// RUN: %clang_cc1 -x c++-header %S/reinclude2.h -include-pch %t1 -emit-pch -o %t2
// RUN: %clang_cc1 %s -include-pch %t2 -fsyntax-only -verify
// RUN: %clang_cc1 -x c++-header %S/reinclude2.h -include-pch %t1 -emit-pch -o %t2 
// RUN: %clang_cc1 %s -include-pch %t2 -fsyntax-only -verify

// expected-no-diagnostics

int q2 = A::y;
