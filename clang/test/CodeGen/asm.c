// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// PR10415:
//
// CHECK:      module asm "foo1"
// CHECK-NEXT: module asm "foo2"
// CHECK-NEXT: module asm "foo3"
__asm__ ("foo1");
__asm__ ("foo2");
__asm__ ("foo3");

void t1(int len) {
  // CHECK-LABEL: @t1
  // CHECK:         call { i32, i32 } asm sideeffect "", "=&r,=&r,1,~{dirflag},~{fpsr},~{flags}"
  // CHECK-SAME:      (i32 [[T1:%[.a-z0-9]+]])
  __asm__ volatile ("" : "=&r" (len), "+&r" (len));
}

void t2(unsigned long long t)  {
  // CHECK-LABEL: @t2
  // CHECK:         call void asm sideeffect "", "=*m,*m,~{dirflag},~{fpsr},~{flags}"
  // CHECK-SAME:      (ptr elementtype(i64) [[T2:%[a-z0-9.]+]], ptr elementtype(i64) [[T2]])
  __asm__ volatile ("" : "+m" (t));
}

void t3(unsigned char *src, unsigned long long temp) {
  // CHECK-LABEL: @t3
  // CHECK:         call ptr asm sideeffect "", "=*m,=r,*m,1,~{dirflag},~{fpsr},~{flags}"
  // CHECK-SAME:      (ptr elementtype(i64) [[T3:%[a-z0-9.]+]], ptr elementtype(i64) [[T3]], ptr %{{.*}})
  __asm__ volatile ("" : "+m" (temp), "+r" (src));
}

void t4(void) {
  // CHECK-LABEL: @t4
  // CHECK:         call void asm sideeffect "", "*m,*m,~{dirflag},~{fpsr},~{flags}"
  // CHECK-SAME:      (ptr elementtype(i64) %{{.*}}, ptr elementtype(%struct.reg) %{{.*}})
  unsigned long long a;
  struct reg { unsigned long long a, b; } b;

  __asm__ volatile ("" : : "m" (a), "m" (b));
}

// PR3417
void t5(int i) {
  // CHECK-LABEL: @t5
  // CHECK:         call i32 asm "nop", "=r,0,~{dirflag},~{fpsr},~{flags}"(ptr @t5)
  asm ("nop" : "=r" (i) : "0" (t5));
}

// PR3641
void t6(void) {
  // CHECK-LABEL: @t6
  // CHECK:         call void asm sideeffect "", "i,~{dirflag},~{fpsr},~{flags}"(ptr @{{.*}})
  __asm__ volatile ("" : : "i" (t6));
}

void t7(int a) {
  // CHECK-LABEL: @t7
  // CHECK:         call i32 asm sideeffect "T7 NAMED: $1", "=r,i,0,~{dirflag},~{fpsr},~{flags}"
  // CHECK-SAME:      (i32 4, i32 %{{.*}})
  __asm__ volatile ("T7 NAMED: %[input]" : "+r" (a): [input] "i" (4));
}

void t8(void) {
  // CHECK-LABEL: @t8
  // CHECK:         call void asm sideeffect "T8 NAMED MODIFIER: ${0:c}", "i,~{dirflag},~{fpsr},~{flags}"(i32 4)
  __asm__ volatile ("T8 NAMED MODIFIER: %c[input]" : : [input] "i" (4));
}

// PR3682
unsigned t9(unsigned int a) {
  // CHECK-LABEL: @t9
  // CHECK:         call i32 asm "bswap $0 $1", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  asm ("bswap %0 %1" : "+r" (a));
  return a;
}

// PR3373
unsigned t10(signed char input) {
  // CHECK-LABEL: @t10
  // CHECK:         call i32 asm "xyz", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  unsigned  output;

  __asm__ ("xyz" : "=a" (output) : "0" (input));
  return output;
}

// PR3373
unsigned char t11(unsigned input) {
  // CHECK-LABEL: @t11
  // CHECK:         call i32 asm "xyz", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  unsigned char output;

  __asm__ ("xyz" : "=a" (output) : "0" (input));
  return output;
}

unsigned char t12(unsigned input) {
  // CHECK-LABEL: @t12
  // CHECK:         call i32 asm "xyz $1", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  unsigned char output;

  __asm__ ("xyz %1" : "=a" (output) : "0" (input));
  return output;
}

// bitfield destination of an asm.
struct S {
  int a : 4;
};

void t13(struct S *P) {
  // CHECK-LABEL: @t13
  // CHECK:         call i32 asm "abc $0", "=r,~{dirflag},~{fpsr},~{flags}"()
  __asm__ ("abc %0" : "=r" (P->a));
}

struct large {
  int x[1000];
};

unsigned long t14(int x, struct large *P) {
  // CHECK-LABEL: @t14
  // CHECK:         call i32 asm "xyz ", "=r,*m,0,~{dirflag},~{fpsr},~{flags}"
  // CHECK-SAME:      (ptr elementtype(%struct.large) %{{.*}}, i32 %{{.*}})
  __asm__ ("xyz " : "=r" (x) : "m" (*P), "0" (x));
  return x;
}

// PR4938
int t15(void) {
  // CHECK-LABEL: @t15
  // CHECK:         call i32 asm "nop;", "=%{cx},r,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  int a, b;

  asm ("nop;" :"=%c" (a) : "r" (b));
  return 0;
}

// PR6475
void t16(void) {
  // CHECK-LABEL: @t16
  // CHECK:         call void asm "nop", "=*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %{{.*}})
  int i;

  __asm__ ("nop": "=m" (i));
}

int t17(unsigned data) {
  // CHECK-LABEL: @t17
  // CHECK:         [[ASM_RES:%[a-z0-9.]+]] ={{.*}} call { i32, i32 }
  // CHECK-SAME:      asm "xyz", "={ax},={dx},{ax},~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  // CHECK-NEXT:    extractvalue { i32, i32 } [[ASM_RES]], 0
  // CHECK-NEXT:    extractvalue { i32, i32 } [[ASM_RES]], 1
  int a, b;

  asm ("xyz" : "=a" (a), "=d" (b) : "a" (data));
  return a + b;
}

// PR6780
int t18(unsigned data) {
  // CHECK-LABEL: @t18
  // CHECK:         call i32 asm "x$(abc$|def$|ghi$)z", "=r,r,~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  int a, b;

  asm ("x{abc|def|ghi}z" : "=r" (a) : "r" (data));
  return a + b;
}

// PR6845 - Mismatching source/dest fp types.
double t19(double x) {
  // CHECK-LABEL: @t19
  // CHECK:         fpext double {{.*}} to x86_fp80
  // CHECK-NEXT:    call x86_fp80 asm sideeffect "frndint", "={st},0,~{dirflag},~{fpsr},~{flags}"(x86_fp80 {{.*}})
  // CHECK:         fptrunc x86_fp80 {{.*}} to double
  register long double result;

  __asm __volatile ("frndint"  : "=t" (result) : "0" (x));
  return result;
}

float t20(long double x) {
  // CHECK-LABEL: @t20
  // CHECK:         call x86_fp80 asm sideeffect "frndint", "={st},0,~{dirflag},~{fpsr},~{flags}"(x86_fp80 {{.*}})
  // CHECK-NEXT:    fptrunc x86_fp80 {{.*}} to float
  register float result;

  __asm __volatile ("frndint"  : "=t" (result) : "0" (x));
  return result;
}

// accept 'l' constraint
unsigned char t21(unsigned char a, unsigned char b) {
  // CHECK-LABEL: @t21
  // CHECK:         call i32 asm "0:\0A1:\0A", "=l{ax},0,{cx},~{edx},~{cc},~{dirflag},~{fpsr},~{flags}"
  // CHECK-SAME:      (i32 {{.*}}, i32 {{.*}})
  unsigned int la = a;
  unsigned int lb = b;
  unsigned int bigres;
  unsigned char res;

  __asm__ ("0:\n1:\n"
           : [bigres] "=la"(bigres)
           : [la] "0"(la), [lb] "c"(lb)
           : "edx", "cc");
  res = bigres;
  return res;
}

// accept 'l' constraint
unsigned char t22(unsigned char a, unsigned char b) {
  // CHECK-LABEL: @t22
  // CHECK:         call i32 asm "0:\0A1:\0A", "=l{ax},0,{cx},~{edx},~{cc},~{dirflag},~{fpsr},~{flags}"
  // CHECK-SAME:      (i32 {{.*}}, i32 {{.*}})
  unsigned int la = a;
  unsigned int lb = b;
  unsigned char res;

  __asm__ ("0:\n1:\n"
           : [res] "=la" (res)
           : [la] "0" (la), [lb] "c" (lb)
           : "edx", "cc");
  return res;
}

void *t23(char c) {
  // CHECK-LABEL: @t23
  // CHECK:         [[C:%[a-z0-9.]+]] = zext i8 {{.*}} to i32
  // CHECK-NEXT:    call ptr asm "foobar", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i32 [[C]])
  void *addr;

  __asm__ ("foobar" : "=a" (addr) : "0" (c));
  return addr;
}

// PR10299 - fpsr, fpcr
void t24(void) {
  // CHECK-LABEL: @t24
  // CHECK:         call void asm sideeffect "finit", "~{st},~{st(1)},~{st(2)},~{st(3)},~{st(4)},~{st(5)},~{st(6)},~{st(7)},~{fpsr},~{fpcr},~{dirflag},~{fpsr},~{flags}"()
  __asm__ __volatile__ ("finit" : : :
                        "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)",
                        "st(6)", "st(7)", "fpsr", "fpcr");
}

// AVX registers
typedef long long __m256i __attribute__((__vector_size__(32)));

void t25(__m256i *p) {
  // CHECK-LABEL: @t25
  // CHECK:         call void asm sideeffect "vmovaps  $0, %ymm0", "*m,~{ymm0},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(<4 x i64>) {{.*}})
  __asm__ volatile ("vmovaps  %0, %%ymm0" : : "m" (*(__m256i*)p) : "ymm0");
}

// Check to make sure the inline asm non-standard dialect attribute _not_ is
// emitted.
void t26(void) {
  // CHECK-LABEL: @t26
  // CHECK:         call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"()
  // CHECK-NOT:     ia_nsdialect
  // CHECK:         ret void
  asm volatile ("nop");
}

// Check handling of '*' and '#' constraint modifiers.
void t27(void) {
  // CHECK-LABEL: @t27
  // CHECK:         call void asm sideeffect "/* $0 */", "i|r,~{dirflag},~{fpsr},~{flags}"(i32 1)
  asm volatile ("/* %0 */" : : "i#*X,*r" (1));
}

static unsigned t28_var[1];

void t28(void) {
  // CHECK-LABEL: @t28
  // CHECK:         call void asm sideeffect "movl %eax, $0", "*m,~{dirflag},~{fpsr},~{flags}"
  // CHECK-SAME:      (ptr elementtype([1 x i32]) @t28_var)
  asm volatile ("movl %%eax, %0" : : "m" (t28_var));
}

int t29(int cond) {
  // CHECK-LABEL: @t29
  // CHECK:         callbr void asm sideeffect "testl $0, $0; jne ${1:l};", "r,!i,!i,~{dirflag},~{fpsr},~{flags}"(i32 {{.*}})
  // CHECK-NEXT:            to label %asm.fallthrough [label %label_true, label %loop]
  asm goto ("testl %0, %0; jne %l1;" : : "r" (cond) : : label_true, loop);
  return 0;

loop:
  return 0;

label_true:
  return 1;
}

void *t30(void *ptr) {
  // CHECK-LABEL: @t30
  // CHECK:         call ptr asm "lea $1, $0", "=r,p,~{dirflag},~{fpsr},~{flags}"(ptr {{.*}})
  void *ret;

  asm ("lea %1, %0" : "=r" (ret) : "p" (ptr));
  return ret;
}

void t31(void) {
  // CHECK-LABEL: @t31
  // CHECK:         call void asm sideeffect "T31 CC NAMED MODIFIER: ${0:c}", "i,~{dirflag},~{fpsr},~{flags}"
  __asm__ volatile ("T31 CC NAMED MODIFIER: %cc[input]" : : [input] "i"  (4));
}

// TODO: Move the "rm" tests into a new testcase file once work to better
// support "rm" constraints is done.

void t32(int len) {
  // CHECK-LABEL: @t32
  // CHECK:         call void asm sideeffect "", "=*&rm,0,~{dirflag},~{fpsr},~{flags}"
  __asm__ volatile ("" : "+&&rm" (len));
}

void t33(int len) {
  // CHECK-LABEL: @t33
  // CHECK:         call void asm sideeffect "", "=*%rm,=*rm,0,1,~{dirflag},~{fpsr},~{flags}"
  __asm__ volatile ("" : "+%%rm" (len), "+rm" (len));
}

// PR3908
void t34(int r) {
  // CHECK-LABEL: @t34
  // CHECK:         call i32 asm "PR3908 $1 $3 $2 $0", "=r,mx,mr,x,0,~{dirflag},~{fpsr},~{flags}"
  // CHECK-SAME:      (i32 0, i32 0, double 0.000000e+00, i32 %{{.*}})
  __asm__ ("PR3908 %[lf] %[xx] %[li] %[r]"
           : [r] "+r" (r)
           : [lf] "mx" (0), [li] "mr" (0), [xx] "x" ((double)(0)));
}
