// RUN: %clang_cc1 %s -E | grep '! ,'

#define XX
! XX,

