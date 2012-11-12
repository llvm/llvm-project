// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-apple-darwin9 | grep x86_fp80

long double x = 0;
int checksize[sizeof(x) == 16 ? 1 : -1];
