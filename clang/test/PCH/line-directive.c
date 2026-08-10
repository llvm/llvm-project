// Test this without pch.
// RUN: not %clang_cc1 -include %S/line-directive.h -fsyntax-only %s 2>&1|grep "25:5"

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %S/line-directive.h
// RUN: not %clang_cc1 -include-pch %t -fsyntax-only %s 2>&1|grep "25:5"  

double x; // expected-error{{redefinition of 'x' with a different type}}
















// expected-note{{previous definition is here}}
