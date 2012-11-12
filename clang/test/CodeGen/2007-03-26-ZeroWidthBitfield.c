// RUN: %clang_cc1 %s -emit-llvm -o -
struct Z { int :0; } z;
