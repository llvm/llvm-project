// RUN: echo "#include <stddef.h>" > %t.h
// RUN: %clang -S -g -include %t.h %s -emit-llvm -o %t.ll
// RUN: grep "i32 5" %t.ll
// outer is at line number 5.
int outer = 42;
