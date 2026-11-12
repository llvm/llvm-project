// RUN: %clang_cc1 -DA=1 -DB=2 -E %s | grep 'int a = 927 == 927'
// RUN: %clang_cc1 -DA=1 -DB=1 -E %s | grep 'int a = 37 == 37'
// RUN: %clang_cc1 -E %s | grep 'int a = 927 == 927'
#if A bitand B
#define X 37
#else
#define X 927
#endif

#if A & B
#define Y 37
#else
#define Y 927
#endif

int a = X == Y;
