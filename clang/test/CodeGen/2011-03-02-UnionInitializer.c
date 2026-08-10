// RUN: %clang_cc1 -emit-llvm %s -o -
union { int :3; double f; } u17_017 = {17.17};
