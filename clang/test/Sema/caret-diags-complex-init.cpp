// RUN: not %clang_cc1 -std=c++11 -fsyntax-only -fno-diagnostics-show-line-numbers -fcaret-diagnostics-max-lines=5 %s 2>&1 | FileCheck %s -strict-whitespace


//CHECK: {{.*}}: error: excess elements in scalar initializer
//CHECK-NEXT: {{^}}_Complex double gz1 = {1, 2, 3};
//CHECK-NEXT: {{^}}                             ^{{$}}
_Complex double gz1 = {1, 2, 3}; 

//CHECK: {{.*}}: error: excess elements in scalar initializer
//CHECK-NEXT: {{^}}_Complex double dd = {1.0, 2.0, 3.0};
//CHECK-NEXT: {{^}}                                ^~~{{$}}
_Complex double dd = {1.0, 2.0, 3.0};

//CHECK: {{.*}}: error: excess elements in scalar initializer
//CHECK-NEXT: {{^}}_Complex float fd = {1.0, 2.0, 3.0, 4.0, 5.0};
//CHECK-NEXT: {{^}}                               ^~~{{$}}
_Complex float fd = {1.0, 2.0, 3.0, 4.0, 5.0};

//CHECK: {{.*}}: error: no viable conversion from 'foo' to 'double'
//CHECK-NEXT: {{^}}_Complex double ds = {f, 1.0, b};
//CHECK-NEXT: {{^}}                      ^{{$}}
struct foo{};
struct bar{};

foo f;
bar b;
_Complex double ds = {f, 1.0, b};

//CHECK: {{.*}}: error: no viable conversion from 'foo' to 'double'
//CHECK-NEXT: {{^}}_Complex double fg = {1.0, f};
//CHECK-NEXT: {{^}}                           ^{{$}}
_Complex double fg = {1.0, f};


//CHECK: {{.*}}: error: excess elements in scalar initializer
//CHECK-NEXT: {{^}}_Complex double gg = {1.0, 2.0, f};
//CHECK-NEXT: {{^}}                                ^{{$}}
//CHECK-NEXT: {{^}}6 errors generated.
_Complex double gg = {1.0, 2.0, f};
