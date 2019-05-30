// RUN: %clang_cc1 -fsyntax-only -verify %s

#if !__has_extension(gnu_asm)
#error Extension 'gnu_asm' should be available by default
#endif

void f1() {
  // PR7673: Some versions of GCC support an empty clobbers section.
  asm ("ret" : : :);
}

void f2() {
  asm("foo" : "=r" (a)); // expected-error {{use of undeclared identifier 'a'}}
  asm("foo" : : "r" (b)); // expected-error {{use of undeclared identifier 'b'}} 

  asm const (""); // expected-warning {{ignored const qualifier on asm}}
  asm volatile ("");
  asm restrict (""); // expected-warning {{ignored restrict qualifier on asm}}
  // FIXME: Once GCC supports _Atomic, check whether it allows this.
  asm _Atomic (""); // expected-warning {{ignored _Atomic qualifier on asm}}
}


int a, b, c, d, e, f, g, h, i, j, k, l;

void
fgoto1 (void)
{
  __asm__ volatile goto (""
            :: [a] "r" (a), [b] "r" (b), [c] "r" (c), [d] "r" (d),
               [e] "r" (e), [f] "r" (f), [g] "r" (g), [h] "r" (h),
               [i] "r" (i), [j] "r" (j), [k] "r" (k), [l] "r" (l)
            ::lab1,lab2);
lab1: return;
lab2: return;
}

void
fgoto2 (void)
{
  __asm__ volatile goto (""
            :: [a] "r,m" (a), [b] "r,m" (b), [c] "r,m" (c), [d] "r,m" (d),
               [e] "r,m" (e), [f] "r,m" (f), [g] "r,m" (g), [h] "r,m" (h),
               [i] "r,m" (i), [j] "r,m" (j), [k] "r,m" (k), [l] "r,m" (l)
            :: lab);
  lab: return;
}

int zoo ()
{
  int x,cond,*e;
  // expected-error@+1 {{expected ')'}}
  asm ("mov %[e], %[e]" : : [e] "rm" (*e)::a)
  // expected-error@+1 {{'asm goto' cannot have output constraints}}
  asm goto ("decl %0; jnz %l[a]" :"=r"(x): "m"(x) : "memory" : a);
  // expected-error@+1 {{expected identifie}}
  asm goto ("decl %0;" :: "m"(x) : "memory" : );
  // expected-error@+1 {{expected ':'}}
  asm goto ("decl %0;" :: "m"(x) : "memory" );
  // expected-error@+1 {{use of undeclared label 'x'}}
  asm goto ("decl %0;" :: "m"(x) : "memory" :x);
  // expected-error@+1 {{use of undeclared label 'b'}}
  asm goto ("decl %0;" :: "m"(x) : "memory" :b);
  // expected-error@+1 {{invalid operand number in inline asm string}}
  asm goto ("testl %0, %0; jne %l3;" :: "r"(cond)::label_true, loop);
  // expected-error@+1 {{unknown symbolic operand name in inline assembly string}}
  asm goto ("decl %0; jnz %l[b]" :: "m"(x) : "memory" : a);
a:
label_true:
loop:
  return 0;
}

// rdar://5952468
__asm ; // expected-error {{expected '(' after 'asm'}}

// <rdar://problem/10465079> - Don't crash on wide string literals in 'asm'.
int foo asm (L"bar"); // expected-error {{cannot use wide string literal in 'asm'}}

asm() // expected-error {{expected string literal in 'asm'}}
// expected-error@-1 {{expected ';' after top-level asm block}}

asm(; // expected-error {{expected string literal in 'asm'}}

asm("") // expected-error {{expected ';' after top-level asm block}}

// Unterminated asm strings at the end of the file were causing us to crash, so
// this needs to be last. rdar://15624081
// expected-warning@+3 {{missing terminating '"' character}}
// expected-error@+2 {{expected string literal in 'asm'}}
// expected-error@+1 {{expected ';' after top-level asm block}}
asm("
