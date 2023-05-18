// RUN: %clang_cc1 -emit-llvm %s -o - | grep "icmp ult"

int a(char* a, char* b) {return a<b;}
