// Check any combination of delayed-template-parsing between PCH and TU works.
// RUN: %clang_cc1 %s -emit-pch -o %t.pch
// RUN: %clang_cc1 -fdelayed-template-parsing %s -emit-pch -o %t.delayed.pch
// RUN: %clang_cc1 -DMAIN_FILE -include-pch %t.pch %s
// RUN: %clang_cc1 -DMAIN_FILE -fdelayed-template-parsing -include-pch %t.pch %s
// RUN: %clang_cc1 -DMAIN_FILE -include-pch %t.delayed.pch %s
// RUN: %clang_cc1 -DMAIN_FILE -fdelayed-template-parsing -include-pch %t.delayed.pch %s

#ifndef MAIN_FILE
template <typename T>
T successor(T Value) { return Value + 1; }
#else
int x = successor(42);
#endif
