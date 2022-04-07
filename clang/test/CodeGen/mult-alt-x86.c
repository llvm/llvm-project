// RUN: %clang_cc1 -no-opaque-pointers -triple i686 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64 -emit-llvm %s -o - | FileCheck %s

int mout0;
int min1;
int marray[2];
double dout0;
double din1;

// CHECK: @single_R
void single_R(void)
{
  // CHECK: asm "foo $1,$0", "=R,R[[CLOBBERS:[a-zA-Z0-9@%{},~_ ]*\"]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=R" (mout0) : "R" (min1));
}

// CHECK: @single_q
void single_q(void)
{
  // CHECK: asm "foo $1,$0", "=q,q[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=q" (mout0) : "q" (min1));
}

// CHECK: @single_Q
void single_Q(void)
{
  // CHECK: asm "foo $1,$0", "=Q,Q[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=Q" (mout0) : "Q" (min1));
}

// CHECK: @single_a
void single_a(void)
{
  // CHECK: asm "foo $1,$0", "={ax},{ax}[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=a" (mout0) : "a" (min1));
}

// CHECK: @single_b
void single_b(void)
{
  // CHECK: asm "foo $1,$0", "={bx},{bx}[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=b" (mout0) : "b" (min1));
}

// CHECK: @single_c
void single_c(void)
{
  // CHECK: asm "foo $1,$0", "={cx},{cx}[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=c" (mout0) : "c" (min1));
}

// CHECK: @single_d
void single_d(void)
{
  // CHECK: asm "foo $1,$0", "={dx},{dx}[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=d" (mout0) : "d" (min1));
}

// CHECK: @single_S
void single_S(void)
{
  // CHECK: asm "foo $1,$0", "={si},{si}[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=S" (mout0) : "S" (min1));
}

// CHECK: @single_D
void single_D(void)
{
  // CHECK: asm "foo $1,$0", "={di},{di}[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=D" (mout0) : "D" (min1));
}

// CHECK: @single_A
void single_A(void)
{
  // CHECK: asm "foo $1,$0", "=A,A[[CLOBBERS]](i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=A" (mout0) : "A" (min1));
}

// CHECK: @single_f
void single_f(void)
{
//FIXME: I don't know how to do an 80387 floating point stack register operation, which I think is fp80.
}

// CHECK: @single_t
void single_t(void)
{
//FIXME: I don't know how to do an 80387 floating point stack register operation, which I think is fp80.
}

// CHECK: @single_u
void single_u(void)
{
//FIXME: I don't know how to do an 80387 floating point stack register operation, which I think is fp80.
}

// CHECK: @single_y
void single_y(void)
{
  // CHECK: call double asm "foo $1,$0", "=y,y[[CLOBBERS]](double {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=y" (dout0) : "y" (din1));
}

// CHECK: @single_x
void single_x(void)
{
  // CHECK: asm "foo $1,$0", "=x,x[[CLOBBERS]](double {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=x" (dout0) : "x" (din1));
}

// CHECK: @single_Y
void single_Y(void)
{
  // 'Y' constraint currently broken.
  //asm("foo %1,%0" : "=Y0" (mout0) : "Y0" (min1));
  //asm("foo %1,%0" : "=Yz" (mout0) : "Yz" (min1));
  //asm("foo %1,%0" : "=Yt" (mout0) : "Yt" (min1));
  //asm("foo %1,%0" : "=Yi" (mout0) : "Yi" (min1));
  //asm("foo %1,%0" : "=Ym" (mout0) : "Ym" (min1));
}

// CHECK: @single_I
void single_I(void)
{
  // CHECK: asm "foo $1,$0", "=*m,I[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=m" (mout0) : "I" (1));
}

// CHECK: @single_J
void single_J(void)
{
  // CHECK: asm "foo $1,$0", "=*m,J[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=m" (mout0) : "J" (1));
}

// CHECK: @single_K
void single_K(void)
{
  // CHECK: asm "foo $1,$0", "=*m,K[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=m" (mout0) : "K" (1));
}

// CHECK: @single_L
void single_L(void)
{
  // CHECK: asm "foo $1,$0", "=*m,L[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 255)
  asm("foo %1,%0" : "=m" (mout0) : "L" (0xff));
  // CHECK: asm "foo $1,$0", "=*m,L[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 65535)
  asm("foo %1,%0" : "=m" (mout0) : "L" (0xffff));
  // CHECK: asm "foo $1,$0", "=*m,L[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 -1)
  asm("foo %1,%0" : "=m" (mout0) : "L" (0xffffffff));
}

// CHECK: @single_M
void single_M(void)
{
  // CHECK: asm "foo $1,$0", "=*m,M[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=m" (mout0) : "M" (1));
}

// CHECK: @single_N
void single_N(void)
{
  // CHECK: asm "foo $1,$0", "=*m,N[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=m" (mout0) : "N" (1));
}

// CHECK: @single_G
void single_G(void)
{
  // CHECK: asm "foo $1,$0", "=*m,G[[CLOBBERS]](i32* elementtype(i32) @mout0, double {{1.[0]+e[+]*[0]+}})
  asm("foo %1,%0" : "=m" (mout0) : "G" (1.0));
}

// CHECK: @single_C
void single_C(void)
{
  // CHECK: asm "foo $1,$0", "=*m,C[[CLOBBERS]](i32* elementtype(i32) @mout0, double {{1.[0]+e[+]*[0]+}})
  asm("foo %1,%0" : "=m" (mout0) : "C" (1.0));
}

// CHECK: @single_e
void single_e(void)
{
  // CHECK: asm "foo $1,$0", "=*m,e[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=m" (mout0) : "e" (1));
}

// CHECK: @single_Z
void single_Z(void)
{
  // CHECK: asm "foo $1,$0", "=*m,Z[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=m" (mout0) : "Z" (1));
}

// CHECK: @multi_R
void multi_R(void)
{
  // CHECK: asm "foo $1,$0", "=*r|R|m,r|R|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,R,m" (mout0) : "r,R,m" (min1));
}

// CHECK: @multi_q
void multi_q(void)
{
  // CHECK: asm "foo $1,$0", "=*r|q|m,r|q|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,q,m" (mout0) : "r,q,m" (min1));
}

// CHECK: @multi_Q
void multi_Q(void)
{
  // CHECK: asm "foo $1,$0", "=*r|Q|m,r|Q|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,Q,m" (mout0) : "r,Q,m" (min1));
}

// CHECK: @multi_a
void multi_a(void)
{
  // CHECK: asm "foo $1,$0", "=*r|{ax}|m,r|{ax}|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,a,m" (mout0) : "r,a,m" (min1));
}

// CHECK: @multi_b
void multi_b(void)
{
  // CHECK: asm "foo $1,$0", "=*r|{bx}|m,r|{bx}|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,b,m" (mout0) : "r,b,m" (min1));
}

// CHECK: @multi_c
void multi_c(void)
{
  // CHECK: asm "foo $1,$0", "=*r|{cx}|m,r|{cx}|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,c,m" (mout0) : "r,c,m" (min1));
}

// CHECK: @multi_d
void multi_d(void)
{
  // CHECK: asm "foo $1,$0", "=*r|{dx}|m,r|{dx}|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,d,m" (mout0) : "r,d,m" (min1));
}

// CHECK: @multi_S
void multi_S(void)
{
  // CHECK: asm "foo $1,$0", "=*r|{si}|m,r|{si}|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,S,m" (mout0) : "r,S,m" (min1));
}

// CHECK: @multi_D
void multi_D(void)
{
  // CHECK: asm "foo $1,$0", "=*r|{di}|m,r|{di}|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,D,m" (mout0) : "r,D,m" (min1));
}

// CHECK: @multi_A
void multi_A(void)
{
  // CHECK: asm "foo $1,$0", "=*r|A|m,r|A|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,A,m" (mout0) : "r,A,m" (min1));
}

// CHECK: @multi_f
void multi_f(void)
{
//FIXME: I don't know how to do an 80387 floating point stack register operation, which I think is fp80.
}

// CHECK: @multi_t
void multi_t(void)
{
//FIXME: I don't know how to do an 80387 floating point stack register operation, which I think is fp80.
}

// CHECK: @multi_u
void multi_u(void)
{
//FIXME: I don't know how to do an 80387 floating point stack register operation, which I think is fp80.
}

// CHECK: @multi_y
void multi_y(void)
{
  // CHECK: asm "foo $1,$0", "=*r|y|m,r|y|m[[CLOBBERS]](double* elementtype(double) @dout0, double {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,y,m" (dout0) : "r,y,m" (din1));
}

// CHECK: @multi_x
void multi_x(void)
{
  // CHECK: asm "foo $1,$0", "=*r|x|m,r|x|m[[CLOBBERS]](double* elementtype(double) @dout0, double {{[a-zA-Z0-9@%]+}})
  asm("foo %1,%0" : "=r,x,m" (dout0) : "r,x,m" (din1));
}

// CHECK: @multi_Y
void multi_Y0(void)
{
  // Y constraint currently broken.
  //asm("foo %1,%0" : "=r,Y0,m" (mout0) : "r,Y0,m" (min1));
  //asm("foo %1,%0" : "=r,Yz,m" (mout0) : "r,Yz,m" (min1));
  //asm("foo %1,%0" : "=r,Yt,m" (mout0) : "r,Yt,m" (min1));
  //asm("foo %1,%0" : "=r,Yi,m" (mout0) : "r,Yi,m" (min1));
  //asm("foo %1,%0" : "=r,Ym,m" (mout0) : "r,Ym,m" (min1));
}

// CHECK: @multi_I
void multi_I(void)
{
  // CHECK: asm "foo $1,$0", "=*r|m|m,r|I|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=r,m,m" (mout0) : "r,I,m" (1));
}

// CHECK: @multi_J
void multi_J(void)
{
  // CHECK: asm "foo $1,$0", "=*r|m|m,r|J|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=r,m,m" (mout0) : "r,J,m" (1));
}

// CHECK: @multi_K
void multi_K(void)
{
  // CHECK: asm "foo $1,$0", "=*r|m|m,r|K|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=r,m,m" (mout0) : "r,K,m" (1));
}

// CHECK: @multi_L
void multi_L(void)
{
  // CHECK: asm "foo $1,$0", "=*r|m|m,r|L|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=r,m,m" (mout0) : "r,L,m" (1));
}

// CHECK: @multi_M
void multi_M(void)
{
  // CHECK: asm "foo $1,$0", "=*r|m|m,r|M|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=r,m,m" (mout0) : "r,M,m" (1));
}

// CHECK: @multi_N
void multi_N(void)
{
  // CHECK: asm "foo $1,$0", "=*r|m|m,r|N|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=r,m,m" (mout0) : "r,N,m" (1));
}

// CHECK: @multi_G
void multi_G(void)
{
  // CHECK: asm "foo $1,$0", "=*r|m|m,r|G|m[[CLOBBERS]](i32* elementtype(i32) @mout0, double {{1.[0]+e[+]*[0]+}})
  asm("foo %1,%0" : "=r,m,m" (mout0) : "r,G,m" (1.0));
}

// CHECK: @multi_C
void multi_C(void)
{
  // CHECK: asm "foo $1,$0", "=*r|m|m,r|C|m[[CLOBBERS]](i32* elementtype(i32) @mout0, double {{1.[0]+e[+]*[0]+}})
  asm("foo %1,%0" : "=r,m,m" (mout0) : "r,C,m" (1.0));
}

// CHECK: @multi_e
void multi_e(void)
{
  // CHECK: asm "foo $1,$0", "=*r|m|m,r|e|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=r,m,m" (mout0) : "r,e,m" (1));
}

// CHECK: @multi_Z
void multi_Z(void)
{
  // CHECK: asm "foo $1,$0", "=*r|m|m,r|Z|m[[CLOBBERS]](i32* elementtype(i32) @mout0, i32 1)
  asm("foo %1,%0" : "=r,m,m" (mout0) : "r,Z,m" (1));
}
