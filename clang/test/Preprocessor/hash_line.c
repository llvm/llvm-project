// The 1 and # should not go on the same line.
// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s
// CHECK: {{^1$}}
// CHECK-NEXT: {{^      #$}}
// CHECK-NEXT: {{^2$}}
// CHECK-NEXT: {{^           #$}}
#define EMPTY
#define IDENTITY(X) X
1
EMPTY #
2
IDENTITY() #
