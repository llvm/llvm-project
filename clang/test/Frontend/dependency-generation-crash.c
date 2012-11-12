// RUN: touch %t
// RUN: chmod 0 %t
// %clang -E -dependency-file bla -MT %t -MP -o %t -x c /dev/null
// rdar://9286457
