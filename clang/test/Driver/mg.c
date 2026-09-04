// RUN: %clang -M -MG -include nonexistent-preinclude.h -std=c23 %s | FileCheck %s
// CHECK: nonexistent-preinclude.h
// CHECK: nonexistent-ppinclude.h
// CHECK: nonexistent-embed

#include "nonexistent-ppinclude.h"
#embed "nonexistent-embed"
