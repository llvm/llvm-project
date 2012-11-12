// RUN: %clang_cc1 %s -emit-llvm -o -
struct W {};
struct Y {
  struct W w;
  int i:1;
} __attribute__ ((packed)) y;
