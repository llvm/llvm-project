// RUN: %clang -M -MG -include nonexistent-preinclude.h %s | FileCheck %s
// CHECK: nonexistent-preinclude.h
// CHECK: nonexistent-ppinclude.h

#include "nonexistent-ppinclude.h"
