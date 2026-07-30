// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %S/cxx-offsetof-base.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only %s 
