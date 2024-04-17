// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s


// CHECK: cir.asm(x86_att, 
// CHECK:   out = [],
// CHECK:   in = [],
// CHECK:   in_out = [],
// CHECK:   {"" "~{dirflag},~{fpsr},~{flags}"}) side_effects
void empty1() {
  __asm__ volatile("" : : : );
}

// CHECK: cir.asm(x86_att, 
// CHECK:   out = [],
// CHECK:   in = [],
// CHECK:   in_out = [],
// CHECK:   {"xyz" "~{dirflag},~{fpsr},~{flags}"}) side_effects
void empty2() {
  __asm__ volatile("xyz" : : : );
}

// CHECK: cir.asm(x86_att, 
// CHECK:   out = [%0 : !cir.ptr<!s32i> (maybe_memory)],
// CHECK:   in = [],
// CHECK:   in_out = [%0 : !cir.ptr<!s32i> (maybe_memory)],
// CHECK:   {"" "=*m,*m,~{dirflag},~{fpsr},~{flags}"}) side_effects
void empty3(int x) {
  __asm__ volatile("" : "+m"(x));
}

// CHECK: cir.asm(x86_att, 
// CHECK:   out = [],
// CHECK:   in = [%0 : !cir.ptr<!s32i> (maybe_memory)],
// CHECK:   in_out = [],
// CHECK:   {"" "*m,~{dirflag},~{fpsr},~{flags}"}) side_effects
void empty4(int x) {
  __asm__ volatile("" : : "m"(x));
}

// CHECK: cir.asm(x86_att, 
// CHECK:   out = [%0 : !cir.ptr<!s32i> (maybe_memory)],
// CHECK:   in = [],
// CHECK:   in_out = [],
// CHECK:   {"" "=*m,~{dirflag},~{fpsr},~{flags}"}) side_effects
void empty5(int x) {
  __asm__ volatile("" : "=m"(x));
}

// CHECK: %3 = cir.asm(x86_att, 
// CHECK:   out = [],
// CHECK:   in = [],
// CHECK:   in_out = [%2 : !s32i],
// CHECK:   {"" "=&r,=&r,1,~{dirflag},~{fpsr},~{flags}"}) side_effects -> !ty_22anon2E022
void empty6(int x) {
  __asm__ volatile("" : "=&r"(x), "+&r"(x));
}

// CHECK: [[TMP0:%.*]] = cir.alloca !s32i, cir.ptr <!s32i>, ["a"] 
// CHECK: [[TMP1:%.*]] = cir.load %0 : cir.ptr <!u32i>, !u32i
// CHECK: [[TMP2:%.*]] = cir.asm(x86_att, 
// CHECK:       out = [],
// CHECK:       in = [%3 : !u32i],
// CHECK:       in_out = [],
// CHECK:       {"addl $$42, $1" "=r,r,~{dirflag},~{fpsr},~{flags}"}) -> !s32i
// CHECK: cir.store [[TMP2]], [[TMP0]] : !s32i, cir.ptr <!s32i> loc(#loc42)
unsigned add1(unsigned int x) {
  int a;
  __asm__("addl $42, %[val]"
      : "=r" (a)
      : [val] "r" (x)
      );
  
  return a;
}

// CHECK: [[TMP0:%.*]] = cir.alloca !u32i, cir.ptr <!u32i>, ["x", init] {alignment = 4 : i64}
// CHECK: cir.store %arg0, [[TMP0]] : !u32i, cir.ptr <!u32i>
// CHECK: [[TMP1:%.*]] = cir.load [[TMP0]] : cir.ptr <!u32i>, !u32i
// CHECK: [[TMP2:%.*]] = cir.asm(x86_att, 
// CHECK:       out = [],
// CHECK:       in = [],
// CHECK:       in_out = [%2 : !u32i],
// CHECK:       {"addl $$42, $0" "=r,0,~{dirflag},~{fpsr},~{flags}"}) -> !u32i
// CHECK: cir.store [[TMP2]], [[TMP0]] : !u32i, cir.ptr <!u32i>
unsigned add2(unsigned int x) {
  __asm__("addl $42, %[val]"
      : [val] "+r" (x)
      );
  return x;
}


// CHECK: [[TMP0:%.*]] = cir.alloca !u32i, cir.ptr <!u32i>, ["x", init]
// CHECK: [[TMP1:%.*]] = cir.load [[TMP0]] : cir.ptr <!u32i>, !u32i
// CHECK: [[TMP2:%.*]] = cir.asm(x86_att, 
// CHECK:       out = [],
// CHECK:       in = [],
// CHECK:       in_out = [%2 : !u32i],
// CHECK:       {"addl $$42, $0  \0A\09          subl $$1, $0    \0A\09          imul $$2, $0" "=r,0,~{dirflag},~{fpsr},~{flags}"}) -> !u32i
// CHECK: cir.store [[TMP2]], [[TMP0]]  : !u32i, cir.ptr <!u32i>
unsigned add3(unsigned int x) { // ((42 + x) - 1) * 2
  __asm__("addl $42, %[val]  \n\t\
          subl $1, %[val]    \n\t\
          imul $2, %[val]"
      : [val] "+r" (x)
      );  
  return x;
}

// CHECK: [[TMP0:%.*]] = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>, ["x", init] 
// CHECK: cir.store %arg0, [[TMP0]] : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK: [[TMP1:%.*]] = cir.load deref [[TMP0]] : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: cir.asm(x86_att, 
// CHECK:       out = [%1 : !cir.ptr<!s32i> (maybe_memory)],
// CHECK:       in = [],
// CHECK:       in_out = [],
// CHECK:       {"addl $$42, $0" "=*m,~{dirflag},~{fpsr},~{flags}"}) 
// CHECK-NEXT: cir.return
void add4(int *x) {    
  __asm__("addl $42, %[addr]" : [addr] "=m" (*x));
}


// CHECK: [[TMP0:%.*]] = cir.alloca !cir.float, cir.ptr <!cir.float>, ["x", init]
// CHECK: [[TMP1:%.*]] = cir.alloca !cir.float, cir.ptr <!cir.float>, ["y", init]
// CHECK: [[TMP2:%.*]] = cir.alloca !cir.float, cir.ptr <!cir.float>, ["r"]
// CHECK: cir.store %arg0, [[TMP0]] : !cir.float, cir.ptr <!cir.float>
// CHECK: cir.store %arg1, [[TMP1]] : !cir.float, cir.ptr <!cir.float>
// CHECK: [[TMP3:%.*]] = cir.load [[TMP0]] : cir.ptr <!cir.float>, !cir.float
// CHECK: [[TMP4:%.*]] = cir.load [[TMP1]] : cir.ptr <!cir.float>, !cir.float
// CHECK: [[TMP5:%.*]] = cir.asm(x86_att, 
// CHECK:       out = [],
// CHECK:       in = [%4 : !cir.float, %5 : !cir.float],
// CHECK:       in_out = [],
// CHECK:       {"flds $1; flds $2; faddp" "=&{st},imr,imr,~{dirflag},~{fpsr},~{flags}"}) -> !cir.float
// CHECK: cir.store [[TMP5]], [[TMP2]] : !cir.float, cir.ptr <!cir.float>
float add5(float x, float y) {
   float r;
  __asm__("flds %[x]; flds %[y]; faddp"
          : "=&t" (r)
          : [x] "g" (x), [y] "g" (y));
  return r;
}

/*
There are tests from clang/test/CodeGen/asm.c. No checks for now - we just make
sure no crashes happen
*/


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

void t5(int i) {
  asm("nop" : "=r"(i) : "0"(t5));
}

void t6(void) {
  __asm__ volatile("" : : "i" (t6));
}

void t7(int a) {
  __asm__ volatile("T7 NAMED: %[input]" : "+r"(a): [input] "i" (4));  
}

void t8(void) {
  __asm__ volatile("T8 NAMED MODIFIER: %c[input]" :: [input] "i" (4));  
}

unsigned t9(unsigned int a) {
  asm("bswap %0 %1" : "+r" (a));
  return a;
}

void t10(int r) {
  __asm__("PR3908 %[lf] %[xx] %[li] %[r]" : [r] "+r" (r) : [lf] "mx" (0), [li] "mr" (0), [xx] "x" ((double)(0)));
}

unsigned t11(signed char input) {
  unsigned  output;
  __asm__("xyz"
          : "=a" (output)
          : "0" (input));
  return output;
}

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

int t16(void) {
  int a,b;
  asm ( "nop;"
       :"=%c" (a)
       : "r" (b)
       );
  return 0;
}

void t17(void) {
  int i;
  __asm__ ( "nop": "=m"(i));
}

int t18(unsigned data) {
  int a, b;

  asm("xyz" :"=a"(a), "=d"(b) : "a"(data));
  return a + b;
}

int t19(unsigned data) {
  int a, b;

  asm("x{abc|def|ghi}z" :"=r"(a): "r"(data));
  return a + b;
}

// skip t20 and t21: long double is not supported

// accept 'l' constraint
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

// accept 'l' constraint
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
  __asm__ ("foobar" : "=a" (addr) : "0" (c));
  return addr;
}

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

//t26 skipped - no vector type support

// Check to make sure the inline asm non-standard dialect attribute _not_ is
// emitted.
void t27(void) {
  asm volatile("nop");
}

// Check handling of '*' and '#' constraint modifiers.
void t28(void)
{
  asm volatile ("/* %0 */" : : "i#*X,*r" (1));
}

static unsigned t29_var[1];

void t29(void) {
  asm volatile("movl %%eax, %0"
               :
               : "m"(t29_var));
}

void t30(int len) {
  __asm__ volatile(""
                   : "+&&rm"(len));
}

void t31(int len) {
  __asm__ volatile(""
                   : "+%%rm"(len), "+rm"(len));
}

//t32 skipped: no goto

void *t33(void *ptr)
{
  void *ret;
  asm ("lea %1, %0" : "=r" (ret) : "p" (ptr));
  return ret;  
}
