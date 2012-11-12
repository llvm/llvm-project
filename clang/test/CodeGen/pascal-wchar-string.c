// RUN: %clang_cc1 -emit-llvm -o -  %s -fpascal-strings -fshort-wchar  | FileCheck %s
// rdar://8020384

#include <stddef.h>

extern void abort (void);

typedef unsigned short UInt16;

typedef UInt16 UniChar;

int main(int argc, char* argv[])
{

        char st[] = "\pfoo";            // pascal string
        UniChar wt[] = L"\pbar";        // pascal Unicode string
	UniChar wt1[] = L"\p";
	UniChar wt2[] = L"\pgorf";

        if (st[0] != 3)
          abort ();
        if (wt[0] != 3)
          abort ();
        if (wt1[0] != 0)
          abort ();
        if (wt2[0] != 4)
          abort ();
        
        return 0;
}

// CHECK: [i16 3, i16 98, i16 97, i16 114, i16 0]
// CHECK: [i16 4, i16 103, i16 111, i16 114, i16 102, i16 0]


// PR8856 - -fshort-wchar makes wchar_t be unsigned.
// CHECK: @test2
// CHECK: store volatile i32 1, i32* %isUnsigned
void test2() {
  volatile int isUnsigned = (wchar_t)-1 > (wchar_t)0;
}
