// RUN: %clang_cc1 %s -triple i386-pc-linux-gnu -verify -fsyntax-only

struct NonTrivial {
  ~NonTrivial();
  int f(int);
private:
  int k;
};
void JumpDiagnostics(int n) {
// expected-error@+1 {{cannot jump from this goto statement to its label}}
  goto DirectJump;
// expected-note@+1 {{jump bypasses variable with a non-trivial destructor}}
  NonTrivial tnp1;

DirectJump:
// expected-error@+1 {{cannot jump from this asm goto statement to one of its possible targets}}
  asm goto("jmp %l0;" ::::Later);
// expected-note@+1 {{jump bypasses variable with a non-trivial destructor}}
  NonTrivial tnp2;
// expected-note@+1 {{possible target of asm goto statement}}
Later:
  return;
}

struct S { ~S(); };
void foo(int a) {
  if (a) {
FOO:
// expected-note@+2 {{jump exits scope of variable with non-trivial destructor}}
// expected-note@+1 {{jump exits scope of variable with non-trivial destructor}}
    S s;
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
