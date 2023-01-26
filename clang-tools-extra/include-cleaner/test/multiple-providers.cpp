// RUN: clang-include-cleaner --print=changes %s -- -I %S/Inputs | FileCheck --allow-empty %s
#include "foo.h"
#include "foo2.h"

int n = foo();
// Make sure both providers are preserved.
// CHECK-NOT: - "foo.h"
// CHECK-NOT: - "foo2.h"
