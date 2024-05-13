// RUN: %clang_cc1 -std=c99 %s -emit-llvm -o - | \
// RUN:    opt -O3 -disable-output
// PR580

int X, Y;
int foo(void) {
  int i;
        for (i=0; i<100; i++ )
        {
                break;
                i = ( X || Y ) ;
        }
}

