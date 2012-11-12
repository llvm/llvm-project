// The 1 and # should not go on the same line.
// RUN: %clang_cc1 %s -E | not grep "1 #"
// RUN: %clang_cc1 %s -E | grep '^1$'
// RUN: %clang_cc1 %s -E | grep '^      #$'
1
#define EMPTY
EMPTY #

