// RUN: %clang_cc1 -DA=1 -E %s | grep 'int a = 37 == 37'
// RUN: %clang_cc1 -DA=0 -E %s | grep 'int a = 927 == 927'
// RUN: %clang_cc1 -E %s | grep 'int a = 927 == 927'
#if compl 0 bitand A
#define X 37
#else
#define X 927
#endif

#if ~0 & A
#define Y 37
#else
#define Y 927
#endif

int a = X == Y;
