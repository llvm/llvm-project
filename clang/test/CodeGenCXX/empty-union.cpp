// RUN: %clang_cc1 -emit-llvm -o - %s

union sigval { };
union sigval Test1;

union NonPODUnion { ~NonPODUnion(); };
union NonPODUnion Test2;
