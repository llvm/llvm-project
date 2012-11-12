// RUN: %clang_cc1 -DA=1 -E %s | grep 'int a = 927 == 927'
// RUN: %clang_cc1 -E %s | grep 'int a = 37 == 37'
#if not defined(A)
#define X 37
#else
#define X 927
#endif

#if ! defined(A)
#define Y 37
#else
#define Y 927
#endif

int a = X == Y;
