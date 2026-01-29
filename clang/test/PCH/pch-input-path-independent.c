// Tests that when PCHs are chained, the dependent PCHs produced are identical
// whether the input PCH is specified through a relative path or an absolute path.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -triple x86_64-apple-macos11 -emit-pch h1.h -o %t/h1.h.pch
// RUN: %clang_cc1 -triple x86_64-apple-macos11 -emit-pch bridging.h \
// RUN:  -o %t/bridging1.h.pch -include-pch %t/h1.h.pch
// RUN: %clang_cc1 -triple x86_64-apple-macos11 -emit-pch bridging.h \
// RUN:  -o %t/bridging2.h.pch -include-pch ./h1.h.pch

// RUN: diff %t/bridging1.h.pch %t/bridging2.h.pch

//--- h1.h
int bar1() { return 42; }

//--- bridging.h
int bar() { return bar1(); }
