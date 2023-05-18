// RUN: %clang_cc1 -E %s | grep '!!'

#define A(B,C) B ## C

!A(,)!

