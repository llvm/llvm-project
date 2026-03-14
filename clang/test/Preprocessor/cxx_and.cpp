// RUN: %clang_cc1 -DA -DB -E %s | grep 'int a = 37 == 37'
// RUN: %clang_cc1 -DA -E %s | grep 'int a = 927 == 927'
// RUN: %clang_cc1 -DB -E %s | grep 'int a = 927 == 927'
// RUN: %clang_cc1 -E %s | grep 'int a = 927 == 927'
#if defined(A) and defined(B)
#define X 37
#else
#define X 927
#endif

#if defined(A) && defined(B)
#define Y 37
#else
#define Y 927
#endif

int a = X == Y;
