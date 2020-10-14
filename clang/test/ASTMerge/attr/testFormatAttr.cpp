// RUN: %clang -x c++-header -o %t.a.ast %S/Inputs/FormatAttr.cpp
// RUN: %clang_cc1 -x c++ -ast-merge %t.a.ast /dev/null -ast-dump
