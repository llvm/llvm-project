// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// PR10415
__asm__ ("foo1");
__asm__ ("foo2");
__asm__ ("foo3");
// CHECK: module asm "foo1"
// CHECK-NEXT: module asm "foo2"
// CHECK-NEXT: module asm "foo3"

void t1(int len) {
  __asm__ volatile("" : "=&r"(len), "+&r"(len));
}

void t2(unsigned long long t)  {
  __asm__ volatile("" : "+m"(t));
}

void t3(unsigned char *src, unsigned long long temp) {
  __asm__ volatile("" : "+m"(temp), "+r"(src));
}

void t4(void) {
  unsigned long long a;
  struct reg { unsigned long long a, b; } b;

  __asm__ volatile ("":: "m"(a), "m"(b));
}

// PR3417
void t5(int i) {
  asm("nop" : "=r"(i) : "0"(t5));
}

// PR3641
void t6(void) {
  __asm__ volatile("" : : "i" (t6));
}

void t7(int a) {
  __asm__ volatile("T7 NAMED: %[input]" : "+r"(a): [input] "i" (4));
  // CHECK: @t7(i32
  // CHECK: T7 NAMED: $1
}

void t8(void) {
  __asm__ volatile("T8 NAMED MODIFIER: %c[input]" :: [input] "i" (4));
  // CHECK: @t8()
  // CHECK: T8 NAMED MODIFIER: ${0:c}
}

// PR3682
unsigned t9(unsigned int a) {
  asm("bswap %0 %1" : "+r" (a));
  return a;
}

// PR3908
void t10(int r) {
  __asm__("PR3908 %[lf] %[xx] %[li] %[r]" : [r] "+r" (r) : [lf] "mx" (0), [li] "mr" (0), [xx] "x" ((double)(0)));

// CHECK: @t10(
// CHECK:PR3908 $1 $3 $2 $0
}

// PR3373
unsigned t11(signed char input) {
  unsigned  output;
  __asm__("xyz"
          : "=a" (output)
          : "0" (input));
  return output;
}

// PR3373
unsigned char t12(unsigned input) {
  unsigned char output;
  __asm__("xyz"
          : "=a" (output)
          : "0" (input));
  return output;
}

unsigned char t13(unsigned input) {
  unsigned char output;
  __asm__("xyz %1"
          : "=a" (output)
          : "0" (input));
  return output;
}

struct large {
  int x[1000];
};

unsigned long t15(int x, struct large *P) {
  __asm__("xyz "
          : "=r" (x)
          : "m" (*P), "0" (x));
  return x;
}

// bitfield destination of an asm.
struct S {
  int a : 4;
};

void t14(struct S *P) {
  __asm__("abc %0" : "=r"(P->a) );
}

// PR4938
int t16(void) {
  int a,b;
  asm ( "nop;"
       :"=%c" (a)
       : "r" (b)
       );
  return 0;
}

// PR6475
void t17(void) {
  int i;
  __asm__ ( "nop": "=m"(i));

// CHECK: @t17()
// CHECK: call void asm "nop", "=*m,
}

// <rdar://problem/6841383>
int t18(unsigned data) {
  int a, b;

  asm("xyz" :"=a"(a), "=d"(b) : "a"(data));
  return a + b;
// CHECK: t18(i32
// CHECK: = call {{.*}}asm "xyz"
// CHECK-NEXT: extractvalue
// CHECK-NEXT: extractvalue
}

// PR6780
int t19(unsigned data) {
  int a, b;

  asm("x{abc|def|ghi}z" :"=r"(a): "r"(data));
  return a + b;
  // CHECK: t19(i32
  // CHECK: = call {{.*}}asm "x$(abc$|def$|ghi$)z"
}

// PR6845 - Mismatching source/dest fp types.
double t20(double x) {
  register long double result;
  __asm __volatile ("frndint"  : "=t" (result) : "0" (x));
  return result;

  // CHECK: @t20
  // CHECK: fpext double {{.*}} to x86_fp80
  // CHECK-NEXT: call x86_fp80 asm sideeffect "frndint"
  // CHECK: fptrunc x86_fp80 {{.*}} to double
}

float t21(long double x) {
  register float result;
  __asm __volatile ("frndint"  : "=t" (result) : "0" (x));
  return result;
  // CHECK: @t21
  // CHECK: call x86_fp80 asm sideeffect "frndint"
  // CHECK-NEXT: fptrunc x86_fp80 {{.*}} to float
}

// <rdar://problem/8348447> - accept 'l' constraint
unsigned char t22(unsigned char a, unsigned char b) {
  unsigned int la = a;
  unsigned int lb = b;
  unsigned int bigres;
  unsigned char res;
  __asm__ ("0:\n1:\n" : [bigres] "=la"(bigres) : [la] "0"(la), [lb] "c"(lb) :
                        "edx", "cc");
  res = bigres;
  return res;
}

// <rdar://problem/8348447> - accept 'l' constraint
unsigned char t23(unsigned char a, unsigned char b) {
  unsigned int la = a;
  unsigned int lb = b;
  unsigned char res;
  __asm__ ("0:\n1:\n" : [res] "=la"(res) : [la] "0"(la), [lb] "c"(lb) :
                        "edx", "cc");
  return res;
}

void *t24(char c) {
  void *addr;
  // CHECK: @t24
  // CHECK: zext i8 {{.*}} to i32
  // CHECK-NEXT: call i8* asm "foobar"
  __asm__ ("foobar" : "=a" (addr) : "0" (c));
  return addr;
}

// PR10299 - fpsr, fpcr
void t25(void)
{
  __asm__ __volatile__(					   \
		       "finit"				   \
		       :				   \
		       :				   \
		       :"st","st(1)","st(2)","st(3)",	   \
			"st(4)","st(5)","st(6)","st(7)",   \
			"fpsr","fpcr"			   \
							   );
}

// rdar://10510405 - AVX registers
typedef long long __m256i __attribute__((__vector_size__(32)));
void t26 (__m256i *p) {
  __asm__ volatile("vmovaps  %0, %%ymm0" :: "m" (*(__m256i*)p) : "ymm0");
}

// Check to make sure the inline asm non-standard dialect attribute _not_ is
// emitted.
void t27(void) {
  asm volatile("nop");
// CHECK: @t27
// CHECK: call void asm sideeffect "nop"
// CHECK-NOT: ia_nsdialect
// CHECK: ret void
}

// Check handling of '*' and '#' constraint modifiers.
void t28(void)
{
  asm volatile ("/* %0 */" : : "i#*X,*r" (1));
// CHECK: @t28
// CHECK: call void asm sideeffect "/* $0 */", "i|r,~{dirflag},~{fpsr},~{flags}"(i32 1)
}

static unsigned t29_var[1];

void t29(void) {
  asm volatile("movl %%eax, %0"
               :
               : "m"(t29_var));
  // CHECK: @t29
  // CHECK: call void asm sideeffect "movl %eax, $0", "*m,~{dirflag},~{fpsr},~{flags}"([1 x i32]* elementtype([1 x i32]) @t29_var)
}

void t30(int len) {
  __asm__ volatile(""
                   : "+&&rm"(len));
  // CHECK: @t30
  // CHECK: call void asm sideeffect "", "=*&rm,0,~{dirflag},~{fpsr},~{flags}"
}

void t31(int len) {
  __asm__ volatile(""
                   : "+%%rm"(len), "+rm"(len));
  // CHECK: @t31
  // CHECK: call void asm sideeffect "", "=*%rm,=*rm,0,1,~{dirflag},~{fpsr},~{flags}"
}

// CHECK: @t32
int t32(int cond)
{
  asm goto("testl %0, %0; jne %l1;" :: "r"(cond)::label_true, loop);
  // CHECK: callbr void asm sideeffect "testl $0, $0; jne ${1:l};", "r,i,i,~{dirflag},~{fpsr},~{flags}"(i32 %0, i8* blockaddress(@t32, %label_true), i8* blockaddress(@t32, %loop)) #1
  return 0;
loop:
  return 0;
label_true:
  return 1;
}
