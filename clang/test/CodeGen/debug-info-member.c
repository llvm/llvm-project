// RUN: %clang_cc1 -emit-llvm -g < %s | grep DW_TAG_member 

struct A { int x; } a;
