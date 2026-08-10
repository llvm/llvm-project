// RUN: %clang_cc1 -dump-tokens %s 2>&1 | grep "Spelling=.*dumptokens_phyloc.c:3:20"

#define TESTPHYLOC 10

TESTPHYLOC
