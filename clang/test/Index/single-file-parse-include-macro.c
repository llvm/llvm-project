// RUN: split-file %s %t
// RUN: c-index-test -single-file-parse %t/tu.c 2>&1 | FileCheck --allow-empty %t/tu.c

//--- header1.h
#define HEADER2_H "header2.h"
//--- header2.h
//--- tu.c
#include "header1.h"
// CHECK-NOT: tu.c:[[@LINE+1]]:10: error: expected "FILENAME" or <FILENAME>
#include HEADER2_H
