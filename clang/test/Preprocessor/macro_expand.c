// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s

#define X() Y
#define Y() X

A: X()()()
// CHECK: {{^}}A: Y{{$}}

// PR3927
#define f(x) h(x
#define for(x) h(x
#define h(x) x()
B: f(f))
C: for(for))

// CHECK: {{^}}B: f(){{$}}
// CHECK: {{^}}C: for(){{$}}

#define f(x,y...) y
f()

// CHECK: #pragma omp parallel for
#define FOO parallel
#define Streaming _Pragma("omp FOO for")
Streaming

