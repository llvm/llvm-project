// RUN: %clang_cc1 %s -Eonly 2>&1 | grep error

#define COMM1 / ## /
COMM1

