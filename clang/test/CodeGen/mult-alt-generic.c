// RUN: %clang_cc1 -triple i686 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple arm %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple mips %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple mipsel %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple s390x %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple sparc %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple sparcv9 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple thumb %s -emit-llvm -o - | FileCheck %s

int mout0;
int min1;
int marray[2];

// CHECK: @single_m
void single_m(void)
{
  // CHECK: call void asm "foo $1,$0", "=*m,*m[[CLOBBERS:[a-zA-Z0-9@%{},~_$ ]*\"]](ptr elementtype(i32) {{[a-zA-Z0-9@%]+}}, ptr elementtype(i32) {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=m" (mout0) : "m" (min1));
}

// CHECK: @single_o
void single_o(void)
{
  register int out0 = 0;
  register int index = 1;
  // Doesn't really do an offset...
  //asm("foo %1, %2,%0" : "=r" (out0) : "o" (min1));
}

// CHECK: @single_V
void single_V(void)
{
//  asm("foo %1,%0" : "=m" (mout0) : "V" (min1));
}

// CHECK: @single_lt
void single_lt(void)
{
  register int out0 = 0;
  register int in1 = 1;
  // CHECK: call i32 asm "foo $1,$0", "=r,<r[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r" (out0) : "<r" (in1));
  // CHECK: call i32 asm "foo $1,$0", "=r,r<[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r" (out0) : "r<" (in1));
}

// CHECK: @single_gt
void single_gt(void)
{
  register int out0 = 0;
  register int in1 = 1;
  // CHECK: call i32 asm "foo $1,$0", "=r,>r[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r" (out0) : ">r" (in1));
  // CHECK: call i32 asm "foo $1,$0", "=r,r>[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r" (out0) : "r>" (in1));
}

// CHECK: @single_r
void single_r(void)
{
  register int out0 = 0;
  register int in1 = 1;
  // CHECK: call i32 asm "foo $1,$0", "=r,r[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r" (out0) : "r" (in1));
}

// CHECK: @single_i
void single_i(void)
{
  register int out0 = 0;
  // CHECK: call i32 asm "foo $1,$0", "=r,i[[CLOBBERS]](i32 1)
  asm("foo %1,%0" : "=r" (out0) : "i" (1));
}

// CHECK: @single_n
void single_n(void)
{
  register int out0 = 0;
  // CHECK: call i32 asm "foo $1,$0", "=r,n[[CLOBBERS]](i32 1)
  asm("foo %1,%0" : "=r" (out0) : "n" (1));
}

// CHECK: @single_E
void single_E(void)
{
  register double out0 = 0.0;
  // CHECK: call double asm "foo $1,$0", "=r,E[[CLOBBERS]](double {{[0-9.eE+-]+}})
  asm("foo %1,%0" : "=r" (out0) : "E" (1.0e+01));
}

// CHECK: @single_F
void single_F(void)
{
  register double out0 = 0.0;
  // CHECK: call double asm "foo $1,$0", "=r,F[[CLOBBERS]](double {{[0-9.eE+-]+}})
  asm("foo %1,%0" : "=r" (out0) : "F" (1.0));
}

// CHECK: @single_s
void single_s(void)
{
  register int out0 = 0;
  //asm("foo %1,%0" : "=r" (out0) : "s" (single_s));
}

// CHECK: @single_g
void single_g(void)
{
  register int out0 = 0;
  register int in1 = 1;
  // CHECK: call i32 asm "foo $1,$0", "=r,imr[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r" (out0) : "g" (in1));
  // CHECK: call i32 asm "foo $1,$0", "=r,imr[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r" (out0) : "g" (min1));
  // CHECK: call i32 asm "foo $1,$0", "=r,imr[[CLOBBERS]](i32 1)
  asm("foo %1,%0" : "=r" (out0) : "g" (1));
}

// CHECK: @single_X
void single_X(void)
{
  register int out0 = 0;
  register int in1 = 1;
  // CHECK: call i32 asm "foo $1,$0", "=r,X[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r" (out0) : "X" (in1));
  // CHECK: call i32 asm "foo $1,$0", "=r,X[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r" (out0) : "X" (min1));
  // CHECK: call i32 asm "foo $1,$0", "=r,X[[CLOBBERS]](i32 1)
  asm("foo %1,%0" : "=r" (out0) : "X" (1));
  // CHECK: call i32 asm "foo $1,$0", "=r,X[[CLOBBERS]](ptr {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r" (out0) : "X" (marray));
  // CHECK: call i32 asm "foo $1,$0", "=r,X[[CLOBBERS]](double {{[0-9.eE+-]+}})
  asm("foo %1,%0" : "=r" (out0) : "X" (1.0e+01));
  // CHECK: call i32 asm "foo $1,$0", "=r,X[[CLOBBERS]](double {{[0-9.eE+-]+}})
  asm("foo %1,%0" : "=r" (out0) : "X" (1.0));
}

// CHECK: @single_p
void single_p(void)
{
  register int out0 = 0;
  // Constraint converted differently on different platforms moved to platform-specific.
  // : call i32 asm "foo $1,$0", "=r,im[[CLOBBERS]](ptr getelementptr inbounds ([2 x i32], ptr {{[a-zA-Z0-9@%]+}}, i{{32|64}} 0, i{{32|64}} 0))
  asm("foo %1,%0" : "=r" (out0) : "p" (marray));
}

// CHECK: @multi_m
void multi_m(void)
{
  // CHECK: call void asm "foo $1,$0", "=*m|r,m|r[[CLOBBERS]](ptr elementtype(i32) {{[a-zA-Z0-9@%]+}}, i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=m,r" (mout0) : "m,r" (min1));
}

// CHECK: @multi_o
void multi_o(void)
{
  register int out0 = 0;
  register int index = 1;
  // Doesn't really do an offset...
  //asm("foo %1, %2,%0" : "=r,r" (out0) : "r,o" (min1));
}

// CHECK: @multi_V
void multi_V(void)
{
//  asm("foo %1,%0" : "=m,r" (mout0) : "r,V" (min1));
}

// CHECK: @multi_lt
void multi_lt(void)
{
  register int out0 = 0;
  register int in1 = 1;
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|<r[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,r" (out0) : "r,<r" (in1));
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|r<[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,r" (out0) : "r,r<" (in1));
}

// CHECK: @multi_gt
void multi_gt(void)
{
  register int out0 = 0;
  register int in1 = 1;
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|>r[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,r" (out0) : "r,>r" (in1));
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|r>[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,r" (out0) : "r,r>" (in1));
}

// CHECK: @multi_r
void multi_r(void)
{
  register int out0 = 0;
  register int in1 = 1;
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|m[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,r" (out0) : "r,m" (in1));
}

// CHECK: @multi_i
void multi_i(void)
{
  register int out0 = 0;
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|i[[CLOBBERS]](i32 1)
  asm("foo %1,%0" : "=r,r" (out0) : "r,i" (1));
}

// CHECK: @multi_n
void multi_n(void)
{
  register int out0 = 0;
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|n[[CLOBBERS]](i32 1)
  asm("foo %1,%0" : "=r,r" (out0) : "r,n" (1));
}

// CHECK: @multi_E
void multi_E(void)
{
  register double out0 = 0.0;
  // CHECK: call double asm "foo $1,$0", "=r|r,r|E[[CLOBBERS]](double {{[0-9.eE+-]+}})
  asm("foo %1,%0" : "=r,r" (out0) : "r,E" (1.0e+01));
}

// CHECK: @multi_F
void multi_F(void)
{
  register double out0 = 0.0;
  // CHECK: call double asm "foo $1,$0", "=r|r,r|F[[CLOBBERS]](double {{[0-9.eE+-]+}})
  asm("foo %1,%0" : "=r,r" (out0) : "r,F" (1.0));
}

// CHECK: @multi_s
void multi_s(void)
{
  register int out0 = 0;
  //asm("foo %1,%0" : "=r,r" (out0) : "r,s" (multi_s));
}

// CHECK: @multi_g
void multi_g(void)
{
  register int out0 = 0;
  register int in1 = 1;
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|imr[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,r" (out0) : "r,g" (in1));
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|imr[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,r" (out0) : "r,g" (min1));
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|imr[[CLOBBERS]](i32 1)
  asm("foo %1,%0" : "=r,r" (out0) : "r,g" (1));
}

// CHECK: @multi_X
void multi_X(void)
{
  register int out0 = 0;
  register int in1 = 1;
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|X[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,r" (out0) : "r,X" (in1));
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|X[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,r" (out0) : "r,X" (min1));
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|X[[CLOBBERS]](i32 1)
  asm("foo %1,%0" : "=r,r" (out0) : "r,X" (1));
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|X[[CLOBBERS]](ptr {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,r" (out0) : "r,X" (marray));
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|X[[CLOBBERS]](double {{[0-9.eE+-]+}})
  asm("foo %1,%0" : "=r,r" (out0) : "r,X" (1.0e+01));
  // CHECK: call i32 asm "foo $1,$0", "=r|r,r|X[[CLOBBERS]](double {{[0-9.eE+-]+}})
  asm("foo %1,%0" : "=r,r" (out0) : "r,X" (1.0));
}

// CHECK: @multi_p
void multi_p(void)
{
  register int out0 = 0;
  // Constraint converted differently on different platforms moved to platform-specific.
  // : call i32 asm "foo $1,$0", "=r|r,r|im[[CLOBBERS]](ptr getelementptr inbounds ([2 x i32], ptr {{[a-zA-Z0-9@%]+}}, {{i[0-9]*}} 0, {{i[0-9]*}} 0))
  asm("foo %1,%0" : "=r,r" (out0) : "r,p" (marray));
}
