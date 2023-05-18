// RUN: %clang_cc1 %s -triple i386-pc-linux-gnu -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -verify -fsyntax-only

struct S {
  ~S();
  int f(int);
private:
  int k;
};
void test1(int n) {
// expected-error@+1 {{cannot jump from this goto statement to its label}}
  goto DirectJump;
// expected-note@+1 {{jump bypasses variable with a non-trivial destructor}}
  S s1;

DirectJump:
// expected-error@+1 {{cannot jump from this asm goto statement to one of its possible targets}}
  asm goto("jmp %l0;" ::::Later);
// expected-note@+1 {{jump bypasses variable with a non-trivial destructor}}
  S s2;
// expected-note@+1 {{possible target of asm goto statement}}
Later:
  return;
}

struct T { ~T(); };
void test2(int a) {
  if (a) {
FOO:
// expected-note@+2 {{jump exits scope of variable with non-trivial destructor}}
// expected-note@+1 {{jump exits scope of variable with non-trivial destructor}}
    T t;
    void *p = &&BAR;
// expected-error@+1 {{cannot jump from this asm goto statement to one of its possible targets}}
    asm goto("jmp %l0;" ::::BAR);
// expected-error@+1 {{cannot jump from this indirect goto statement to one of its possible targets}}
    goto *p;
    p = &&FOO;
    goto *p;
    return;
  }
// expected-note@+2 {{possible target of asm goto statement}}
// expected-note@+1 {{possible target of indirect goto statement}}
BAR:
  return;
}

int test3(int n)
{
  // expected-error@+2 {{cannot jump from this asm goto statement to one of its possible targets}}
  // expected-error@+1 {{cannot jump from this asm goto statement to one of its possible targets}}
  asm volatile goto("testl %0, %0; jne %l1;" :: "r"(n)::label_true, loop);
  // expected-note@+3 {{jump bypasses initialization of variable length array}}
  // expected-note@+2 {{possible target of asm goto statement}}
  // expected-note@+1 {{jump enters a statement expression}}
  return ({int a[n];label_true: 2;});
  // expected-note@+1 {{jump bypasses initialization of variable length array}}
  int b[n];
// expected-note@+1 {{possible target of asm goto statement}}
loop:
  return 0;
}

void test4cleanup(int*);
// No errors expected.
void test4(void) {
  asm goto(""::::l0);
l0:;
  int x __attribute__((cleanup(test4cleanup)));
  asm goto(""::::l1);
l1:;
}

void statement_expressions() {
  ({
    __label__ label;
    asm goto("" : : : : label);
    label:;
  });

  ({
    __label__ label;
    asm goto("" : : : : label);
    label:;
  });
}
