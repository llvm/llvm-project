// RUN: %clang_cc1 -triple i386-pc-linux-gnu -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify -std=c++11 %s

int a, b, c, d, e, f, g, h, i, j, k, l;

void test1(void) {
  __asm__ volatile goto (""
            :: [a] "r" (a), [b] "r" (b), [c] "r" (c), [d] "r" (d),
               [e] "r" (e), [f] "r" (f), [g] "r" (g), [h] "r" (h),
               [i] "r" (i), [j] "r" (j), [k] "r" (k), [l] "r" (l)
            ::lab1,lab2);
lab1: return;
lab2: return;
}

void test2(void) {
  __asm__ volatile goto (""
            :: [a] "r,m" (a), [b] "r,m" (b), [c] "r,m" (c), [d] "r,m" (d),
               [e] "r,m" (e), [f] "r,m" (f), [g] "r,m" (g), [h] "r,m" (h),
               [i] "r,m" (i), [j] "r,m" (j), [k] "r,m" (k), [l] "r,m" (l)
            :: lab);
  lab: return;
}

int test3(int x) {
  __asm__ volatile goto ("decl %0; jnz %l[a]"
                         : "=r" (x) : "m" (x) : "memory" : a);
a:
  return -x;
}

int test4(int x) {
  int y;
  if (x > 42)
    __asm__ volatile goto ("decl %0; jnz %l[a]"
                           : "=r" (x), "=r" (y) : "m" (x) : "memory" : a);
  else
    __asm__ volatile goto ("decl %0; jnz %l[b]"
                           : "=r" (x), "=r" (y) : "m" (x) : "memory" : b);
  x = y + 42;
a:
  return -x;
b:
  return +x;
}

int test5(void) {
  int x,cond,*e;
  // expected-error@+1 {{expected ')'}}
  asm ("mov %[e], %[e]" : : [e] "rm" (*e)::a)
  // expected-error@+1 {{expected identifier}}
  asm goto ("decl %0;" :: "m"(x) : "memory" : );
  // expected-error@+1  {{expected ':'}}
  asm goto ("decl %0;" :: "m"(x) : "memory" );
  // expected-error@+1 {{use of undeclared label 'x'}}
  asm goto ("decl %0;" :: "m"(x) : "memory" :x);
  // expected-error@+1 {{use of undeclared label 'b'}}
  asm goto ("decl %0;" :: "m"(x) : "memory" :b);
  // expected-error@+1 {{invalid operand number in inline asm string}}
  asm goto ("testl %0, %0; jne %l3;" :: "r"(cond)::label_true, loop);
  // expected-error@+1 {{unknown symbolic operand name in inline assembly string}}
  asm goto ("decl %0; jnz %l[b]" :: "m"(x) : "memory" : a);
label_true:
loop:
a:
  return 0;
}

int test6(int y) {
  int x,cond,*e;
  // expected-error@+1 {{expected ')'}}
  asm ("mov %[e], %[e]" : "=r" (y) : [e] "rm" (*e), "r" (y) :: a)
  // expected-error@+1 {{expected identifier}}
  asm goto ("decl %0;" : "=r" (y) : "m" (x), "r" (y) : "memory" :);
  // expected-error@+1  {{expected ':'}}
  asm goto ("decl %0;" : "=r" (y) : "m" (x), "r" (y) : "memory");
  // expected-error@+1 {{use of undeclared label 'x'}}
  asm goto ("decl %0;" : "=r" (y) : "m" (x), "r" (y) : "memory" : x);
  // expected-error@+1 {{use of undeclared label 'b'}}
  asm goto ("decl %0;" : "=r" (y) : "m" (x), "r" (y) : "memory" : b);
  // expected-error@+1 {{invalid operand number in inline asm string}}
  asm goto ("testl %0, %0; jne %l5;" : "=r" (y) : "r" (cond), "r" (y) :: label_true, loop);
  // expected-error@+1 {{unknown symbolic operand name in inline assembly string}}
  asm goto ("decl %0; jnz %l[b]" : "=r" (y) : "m" (x), "r" (y) : "memory" : a);
label_true:
loop:
a:
  return 0;
}
