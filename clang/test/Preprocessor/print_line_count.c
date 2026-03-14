/* RUN: %clang -E -C -P %s | FileCheck --strict-whitespace %s
   PR2741
   comment */ 
y
// CHECK: {{^}}   comment */{{$}}
// CHECK-NEXT: {{^}}y{{$}}

