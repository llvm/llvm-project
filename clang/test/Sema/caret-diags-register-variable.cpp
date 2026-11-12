// RUN: not %clang_cc1 -triple i386-pc-linux-gnu -std=c++11 -fsyntax-only -fno-diagnostics-show-line-numbers -fcaret-diagnostics-max-lines=5 %s 2>&1 | FileCheck %s -strict-whitespace

struct foo {
  int a;
};

//CHECK: {{.*}}: error: unsupported type for named register variable
//CHECK-NEXT: {{^}}register struct foo bar asm("esp");
//CHECK-NEXT: {{^}}         ^~~~~~~~~~{{$}}
register struct foo bar asm("esp");

//CHECK: {{.*}}: error: register 'edi' unsuitable for global register variables on this target
//CHECK-NEXT: {{^}}register int r0 asm ("edi");
//CHECK-NEXT: {{^}}                     ^{{$}}
register int r0 asm ("edi");

//CHECK: {{.*}}: error: size of register 'esp' does not match variable size
//CHECK-NEXT: {{^}}register long long r1 asm ("esp");
//CHECK-NEXT: {{^}}                           ^{{$}}
register long long r1 asm ("esp");
