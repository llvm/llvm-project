// RUN: %clang_cc1 -emit-llvm %s -o - -O1 -triple=x86_64-gnu-linux | grep "i64 -1"

// PR3709
long long a() { return (long long)(int*)-1;}

