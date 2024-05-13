// The 1 and # should not go on the same line.
// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s
// CHECK: {{^1$}}
// CHECK-NEXT: {{^      #$}}
// CHECK-NEXT: {{^2$}}
// CHECK-NEXT: {{^           #$}}

// RUN: %clang_cc1 -E -P -fminimize-whitespace %s | FileCheck --strict-whitespace %s --check-prefix=MINWS
// MINWS:  {{^}}1#2#{{$}}

#define EMPTY
#define IDENTITY(X) X
1
EMPTY #
2
IDENTITY() #
