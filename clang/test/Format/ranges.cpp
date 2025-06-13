// RUN: grep -Ev "// *[A-Z-]+:" %s \
// RUN:   | clang-format -style=LLVM -offset=2 -length=1 -offset=28 -length=1 -offset=35 -length=8 \
// RUN:   | FileCheck -strict-whitespace %s
// CHECK: {{^int\ \*i;$}}
int*i;

// CHECK: {{^int\ \ \*\ \ i;$}}
int  *  i; 

// CHECK: {{^int\ \*i;$}}
int   *   i;

// CHECK: int I;
// CHECK-NEXT: int J ;
int I ;
int J ;

// RUN: not clang-format -length=0 %s 2>&1 \
// RUN:   | FileCheck -strict-whitespace -check-prefix=CHECK0 %s
// CHECK0: error: length should be at least 1
