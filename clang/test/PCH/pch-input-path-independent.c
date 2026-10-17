// Testing `-include-pch` canonicalization.
// When PCHs are chained, the dependent PCHs produced are identical
// whether the included PCH is specified through a relative path or an absolute
// path (bridging1.h.pch vs bridging2.h.pch).
// The dependent PCHs are also identical regardless of the working
// directory where clang is invoked (bridging1.h.pch vs bridging3.h.pch).

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -triple x86_64-apple-macos11 -emit-pch h1.h -o %t/h1.h.pch
// RUN: %clang_cc1 -triple x86_64-apple-macos11 -emit-pch %t/bridging.h \
// RUN:  -o %t/bridging1.h.pch -include-pch %t/h1.h.pch
// RUN: %clang_cc1 -triple x86_64-apple-macos11 -emit-pch %t/bridging.h \
// RUN:  -o %t/bridging2.h.pch -include-pch ./h1.h.pch
// RUN: mkdir %t/wd/
// RUN: cd %t/wd/
// RUN: %clang_cc1 -triple x86_64-apple-macos11 -emit-pch %t/bridging.h \
// RUN:  -o %t/bridging3.h.pch -include-pch ../h1.h.pch

// RUN: diff %t/bridging1.h.pch %t/bridging2.h.pch
// RUN: diff %t/bridging1.h.pch %t/bridging3.h.pch

//--- h1.h
int bar1() { return 42; }

//--- bridging.h
int bar() { return bar1(); }

