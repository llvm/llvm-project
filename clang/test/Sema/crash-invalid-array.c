// RUN: not %clang_cc1 -O1 %s -emit-llvm
// PR6913

#include <stdio.h>

int main()
{
   int x[10][10];
   int (*p)[] = x; // expected-error {{invalid use of array with unspecified bounds}

   int i;

   for(i = 0; i < 10; ++i)
   {
       p[i][i] = i;
   }
}
