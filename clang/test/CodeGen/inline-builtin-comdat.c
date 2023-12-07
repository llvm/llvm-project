// RUN: %clang_cc1 -triple x86_64-windows -S -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s
// Inline builtin are not supported for odr linkage
// CHECK-NOT: .inline

double __cdecl frexp( double _X, int* _Y);
inline __attribute__((always_inline))  long double __cdecl frexpl( long double __x, int *__exp ) {
  return (long double) frexp((double)__x, __exp );
}

long double pain(void)
{
    long double f = 123.45;
    int i;
    long double f2 = frexpl(f, &i);
    return f2;
}
