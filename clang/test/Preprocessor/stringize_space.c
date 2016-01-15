// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s

#define A(b) -#b  ,  - #b  ,  -# b  ,  - # b
A()

// CHECK: {{^}}-"" , - "" , -"" , - ""{{$}}


#define t(x) #x
t(a
c)

// CHECK: {{^}}"a c"{{$}}

#define str(x) #x
#define f(x) str(-x)
f(
    1)

// CHECK: {{^}}"-1"
