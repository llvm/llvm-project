// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp -fopenmp-version=45 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -DOMP51 -verify=expected,omp50,omp51 -fopenmp -fopenmp-version=51 -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-simd -fopenmp-version=45 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-simd -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -DOMP51 -verify=expected,omp50,omp51 -fopenmp-simd -fopenmp-version=51 -ferror-limit 100 %s -Wuninitialized

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp atomic read
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

int foo(void) {
L1:
  foo();
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
    foo();
    goto L1;
  }
  goto L2;
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
    foo();
  L2:
    foo();
  }

  return 0;
}

struct S {
  int a;
};

int readint(void) {
  int a = 0, b = 0;
// Test for atomic read
#pragma omp atomic read
  // expected-error@+2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;
#pragma omp atomic read
  // expected-error@+2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected built-in assignment operator}}
  foo();
#pragma omp atomic read
  // expected-error@+2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected built-in assignment operator}}
  a += b;
#pragma omp atomic read
  // expected-error@+2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected lvalue expression}}
  a = 0;
#pragma omp atomic read
  a = b;
  // expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'read' clause}}
#pragma omp atomic read read
  a = b;

  return 0;
}

int readS(void) {
  struct S a, b;
  // expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'read' clause}} expected-error@+1 {{unexpected OpenMP clause 'allocate' in directive '#pragma omp atomic'}}
#pragma omp atomic read read allocate(a)
  // expected-error@+2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected expression of scalar type}}
  a = b;

  return a.a;
}

int writeint(void) {
  int a = 0, b = 0;
// Test for atomic write
#pragma omp atomic write
  // expected-error@+2 {{the statement for 'atomic write' must be an expression statement of form 'x = expr;', where x is a lvalue expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;
#pragma omp atomic write
  // expected-error@+2 {{the statement for 'atomic write' must be an expression statement of form 'x = expr;', where x is a lvalue expression with scalar type}}
  // expected-note@+1 {{expected built-in assignment operator}}
  foo();
#pragma omp atomic write
  // expected-error@+2 {{the statement for 'atomic write' must be an expression statement of form 'x = expr;', where x is a lvalue expression with scalar type}}
  // expected-note@+1 {{expected built-in assignment operator}}
  a += b;
#pragma omp atomic write
  a = 0;
#pragma omp atomic write
  a = b;
  // expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'write' clause}}
#pragma omp atomic write write
  a = b;

  return 0;
}

int writeS(void) {
  struct S a, b;
  // expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'write' clause}}
#pragma omp atomic write write
  // expected-error@+2 {{the statement for 'atomic write' must be an expression statement of form 'x = expr;', where x is a lvalue expression with scalar type}}
  // expected-note@+1 {{expected expression of scalar type}}
  a = b;

  return a.a;
}

int updateint(void) {
  int a = 0, b = 0;
// Test for atomic update
#pragma omp atomic update
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected built-in binary or unary operator}}
  foo();
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected built-in binary operator}}
  a = b;
#pragma omp atomic update
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected one of '+', '*', '-', '/', '&', '^', '|', '<<', or '>>' built-in operations}}
  a = b || a;
#pragma omp atomic update
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected one of '+', '*', '-', '/', '&', '^', '|', '<<', or '>>' built-in operations}}
  a = a && b;
#pragma omp atomic update
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  a = (float)a + b;
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  a = 2 * b;
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  a = b + *&a;
#pragma omp atomic update
  *&a = *&a +  2;
#pragma omp atomic update
  a++;
#pragma omp atomic
  ++a;
#pragma omp atomic update
  a--;
#pragma omp atomic
  --a;
#pragma omp atomic update
  a += b;
#pragma omp atomic
  a %= b;
#pragma omp atomic update
  a *= b;
#pragma omp atomic
  a -= b;
#pragma omp atomic update
  a /= b;
#pragma omp atomic
  a &= b;
#pragma omp atomic update
  a ^= b;
#pragma omp atomic
  a |= b;
#pragma omp atomic update
  a <<= b;
#pragma omp atomic
  a >>= b;
#pragma omp atomic update
  a = b + a;
#pragma omp atomic
  a = a * b;
#pragma omp atomic update
  a = b - a;
#pragma omp atomic
  a = a / b;
#pragma omp atomic update
  a = b & a;
#pragma omp atomic
  a = a ^ b;
#pragma omp atomic update
  a = b | a;
#pragma omp atomic
  a = a << b;
#pragma omp atomic
  a = b >> a;
  // expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'update' clause}}
#pragma omp atomic update update
  a /= b;

  return 0;
}

int captureint(void) {
  int a = 0, b = 0, c = 0;
// Test for atomic capture
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected compound statement}}
  ;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  foo();
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected built-in binary or unary operator}}
  a = b;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  a = b || a;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected one of '+', '*', '-', '/', '&', '^', '|', '<<', or '>>' built-in operations}}
  b = a = a && b;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  a = (float)a + b;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  a = 2 * b;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  a = b + *&a;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected exactly two expression statements}}
  { a = b; }
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected exactly two expression statements}}
  {}
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of the first expression}}
  {a = b;a = b;}
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of the first expression}}
  {a = b; a = b || a;}
#pragma omp atomic capture
  {b = a; a = a && b;}
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  b = a = (float)a + b;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  b = a = 2 * b;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  b = a = b + *&a;
#pragma omp atomic capture
  c = *&a = *&a +  2;
#pragma omp atomic capture
  c = a++;
#pragma omp atomic capture
  c = ++a;
#pragma omp atomic capture
  c = a--;
#pragma omp atomic capture
  c = --a;
#pragma omp atomic capture
  c = a += b;
#pragma omp atomic capture
  c = a %= b;
#pragma omp atomic capture
  c = a *= b;
#pragma omp atomic capture
  c = a -= b;
#pragma omp atomic capture
  c = a /= b;
#pragma omp atomic capture
  c = a &= b;
#pragma omp atomic capture
  c = a ^= b;
#pragma omp atomic capture
  c = a |= b;
#pragma omp atomic capture
  c = a <<= b;
#pragma omp atomic capture
  c = a >>= b;
#pragma omp atomic capture
  c = a = b + a;
#pragma omp atomic capture
  c = a = a * b;
#pragma omp atomic capture
  c = a = b - a;
#pragma omp atomic capture
  c = a = a / b;
#pragma omp atomic capture
  c = a = b & a;
#pragma omp atomic capture
  c = a = a ^ b;
#pragma omp atomic capture
  c = a = b | a;
#pragma omp atomic capture
  c = a = a << b;
#pragma omp atomic capture
  c = a = b >> a;
#pragma omp atomic capture
  { c = *&a; *&a = *&a +  2;}
#pragma omp atomic capture
  { *&a = *&a +  2; c = *&a;}
#pragma omp atomic capture
  {c = a; a++;}
#pragma omp atomic capture
  {c = a; (a)++;}
#pragma omp atomic capture
  {++a;c = a;}
#pragma omp atomic capture
  {c = a;a--;}
#pragma omp atomic capture
  {--a;c = a;}
#pragma omp atomic capture
  {c = a; a += b;}
#pragma omp atomic capture
  {c = a; (a) += b;}
#pragma omp atomic capture
  {a %= b; c = a;}
#pragma omp atomic capture
  {c = a; a *= b;}
#pragma omp atomic capture
  {a -= b;c = a;}
#pragma omp atomic capture
  {c = a; a /= b;}
#pragma omp atomic capture
  {a &= b; c = a;}
#pragma omp atomic capture
  {c = a; a ^= b;}
#pragma omp atomic capture
  {a |= b; c = a;}
#pragma omp atomic capture
  {c = a; a <<= b;}
#pragma omp atomic capture
  {a >>= b; c = a;}
#pragma omp atomic capture
  {c = a; a = b + a;}
#pragma omp atomic capture
  {a = a * b; c = a;}
#pragma omp atomic capture
  {c = a; a = b - a;}
#pragma omp atomic capture
  {a = a / b; c = a;}
#pragma omp atomic capture
  {c = a; a = b & a;}
#pragma omp atomic capture
  {a = a ^ b; c = a;}
#pragma omp atomic capture
  {c = a; a = b | a;}
#pragma omp atomic capture
  {a = a << b; c = a;}
#pragma omp atomic capture
  {c = a; a = b >> a;}
#pragma omp atomic capture
  {c = a; a = foo();}
  // expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'capture' clause}}
#pragma omp atomic capture capture
  b = a /= b;

  return 0;
}

void hint(void) {
  int a = 0;
#pragma omp atomic hint // omp45-error {{unexpected OpenMP clause 'hint' in directive '#pragma omp atomic'}} expected-error {{expected '(' after 'hint'}}
  a += 1;
#pragma omp atomic hint( // omp45-error {{unexpected OpenMP clause 'hint' in directive '#pragma omp atomic'}} expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  a += 1;
#pragma omp atomic hint(+ // omp45-error {{unexpected OpenMP clause 'hint' in directive '#pragma omp atomic'}} expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  a += 1;
#pragma omp atomic hint(a // omp45-error {{unexpected OpenMP clause 'hint' in directive '#pragma omp atomic'}} expected-error {{expected ')'}} expected-note {{to match this '('}} omp50-error {{integer constant expression}}
  a += 1;
#pragma omp atomic hint(a) // omp45-error {{unexpected OpenMP clause 'hint' in directive '#pragma omp atomic'}} omp50-error {{integer constant expression}}
  a += 1;
#pragma omp atomic hint(1) hint(1) // omp45-error 2 {{unexpected OpenMP clause 'hint' in directive '#pragma omp atomic'}} expected-error {{directive '#pragma omp atomic' cannot contain more than one 'hint' clause}}
  a += 1;
}

#ifdef OMP51
extern void bbar(void);
extern int ffoo(void);

void compare(void) {
  int x = 0;
  int d = 0;
  int e = 0;
// omp51-error@+3 {{the statement for 'atomic compare' must be a compound statement of form '{x = expr ordop x ? expr : x;}', '{x = x ordop expr? expr : x;}', '{x = x == e ? d : x;}', '{x = e == x ? d : x;}', or 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected compound statement}}
#pragma omp atomic compare
  {}
// omp51-error@+3 {{the statement for 'atomic compare' must be a compound statement of form '{x = expr ordop x ? expr : x;}', '{x = x ordop expr? expr : x;}', '{x = x == e ? d : x;}', '{x = e == x ? d : x;}', or 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected exactly one expression statement}}
#pragma omp atomic compare
  {
    x = d;
    x = e;
  }
// omp51-error@+3 {{the statement for 'atomic compare' must be a compound statement of form '{x = expr ordop x ? expr : x;}', '{x = x ordop expr? expr : x;}', '{x = x == e ? d : x;}', '{x = e == x ? d : x;}', or 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected assignment statement}}
#pragma omp atomic compare
  { x += d; }
// omp51-error@+3 {{the statement for 'atomic compare' must be a compound statement of form '{x = expr ordop x ? expr : x;}', '{x = x ordop expr? expr : x;}', '{x = x == e ? d : x;}', '{x = e == x ? d : x;}', or 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected assignment statement}}
#pragma omp atomic compare
  { bbar(); }
// omp51-error@+3 {{the statement for 'atomic compare' must be a compound statement of form '{x = expr ordop x ? expr : x;}', '{x = x ordop expr? expr : x;}', '{x = x == e ? d : x;}', '{x = e == x ? d : x;}', or 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected conditional operator}}
#pragma omp atomic compare
  { x = d; }
// omp51-error@+3 {{the statement for 'atomic compare' must be a compound statement of form '{x = expr ordop x ? expr : x;}', '{x = x ordop expr? expr : x;}', '{x = x == e ? d : x;}', '{x = e == x ? d : x;}', or 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect binary operator in conditional expression}}
#pragma omp atomic compare
  { x = ffoo() ? e : x; }
// omp51-error@+3 {{the statement for 'atomic compare' must be a compound statement of form '{x = expr ordop x ? expr : x;}', '{x = x ordop expr? expr : x;}', '{x = x == e ? d : x;}', '{x = e == x ? d : x;}', or 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect '<', '>' or '==' as order operator}}
#pragma omp atomic compare
  { x = x >= e ? e : x; }
// omp51-error@+3 {{the statement for 'atomic compare' must be a compound statement of form '{x = expr ordop x ? expr : x;}', '{x = x ordop expr? expr : x;}', '{x = x == e ? d : x;}', '{x = e == x ? d : x;}', or 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect comparison in a form of 'x == e', 'e == x', 'x ordop expr', or 'expr ordop x'}}
#pragma omp atomic compare
  { x = d > e ? e : x; }
// omp51-error@+3 {{the statement for 'atomic compare' must be a compound statement of form '{x = expr ordop x ? expr : x;}', '{x = x ordop expr? expr : x;}', '{x = x == e ? d : x;}', '{x = e == x ? d : x;}', or 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect result value to be at false expression}}
#pragma omp atomic compare
  { x = d > x ? e : d; }
// omp51-error@+4 {{the statement for 'atomic compare' must be a compound statement of form '{x = expr ordop x ? expr : x;}', '{x = x ordop expr? expr : x;}', '{x = x == e ? d : x;}', '{x = e == x ? d : x;}', or 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+3 {{expect binary operator in conditional expression}}
#pragma omp atomic compare
  {
    if (foo())
      x = d;
  }
// omp51-error@+4 {{the statement for 'atomic compare' must be a compound statement of form '{x = expr ordop x ? expr : x;}', '{x = x ordop expr? expr : x;}', '{x = x == e ? d : x;}', '{x = e == x ? d : x;}', or 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+3 {{expect '<', '>' or '==' as order operator}}
#pragma omp atomic compare
  {
    if (x >= d)
      x = d;
  }
// omp51-error@+4 {{the statement for 'atomic compare' must be a compound statement of form '{x = expr ordop x ? expr : x;}', '{x = x ordop expr? expr : x;}', '{x = x == e ? d : x;}', '{x = e == x ? d : x;}', or 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+3 {{expect comparison in a form of 'x == e', 'e == x', 'x ordop expr', or 'expr ordop x'}}
#pragma omp atomic compare
  {
    if (e > d)
      x = d;
  }
// omp51-error@+3 {{the statement for 'atomic compare' must be a compound statement of form '{x = expr ordop x ? expr : x;}', '{x = x ordop expr? expr : x;}', '{x = x == e ? d : x;}', '{x = e == x ? d : x;}', or 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected exactly one expression statement}}
#pragma omp atomic compare
  {
    if (x > d)
      x = e;
    d = e;
  }
// omp51-error@+7 {{the statement for 'atomic compare' must be a compound statement of form '{x = expr ordop x ? expr : x;}', '{x = x ordop expr? expr : x;}', '{x = x == e ? d : x;}', '{x = e == x ? d : x;}', or 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+6 {{unexpected 'else' statement}}
#pragma omp atomic compare
  {
    if (x > e)
      x = e;
    else
      d = e;
  }
}

void compare_capture(void) {
  int x = 0;
  int d = 0;
  int e = 0;
  int v = 0;
  int r = 0;
  float dr = 0.0;
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected compound statement}}
#pragma omp atomic compare capture
  if (x == e) {}
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected exactly one expression statement}}
#pragma omp atomic compare capture
  if (x == e) {
    x = d;
    v = x;
  }
// omp51-error@+4 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+3 {{expected assignment statement}}
#pragma omp atomic compare capture
  if (x == e) {
    bbar();
  }
// omp51-error@+4 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+3 {{expected assignment statement}}
#pragma omp atomic compare capture
  if (x == e) {
    x += d;
  }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect binary operator in conditional expression}}
#pragma omp atomic compare capture
  if (ffoo()) {
    x = d;
  }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect '==' operator}}
#pragma omp atomic compare capture
  if (x > e) {
    x = d;
  }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect comparison in a form of 'x == e', 'e == x', 'x ordop expr', or 'expr ordop x'}}
#pragma omp atomic compare capture
  if (d == e) {
    x = d;
  }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect 'else' statement}}
#pragma omp atomic compare capture
  if (x == e) {
    x = d;
  }
// omp51-error@+5 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+4 {{expected compound statement}}
#pragma omp atomic compare capture
  if (x == e) {
    x = d;
  } else {
  }
// omp51-error@+5 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+4 {{expected exactly one expression statement}}
#pragma omp atomic compare capture
  if (x == e) {
    x = d;
  } else {
    v = x;
    d = e;
  }
// omp51-error@+6 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+5 {{expected assignment statement}}
#pragma omp atomic compare capture
  if (x == e) {
    x = d;
  } else {
    bbar();
  }
// omp51-error@+6 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+5 {{expected assignment statement}}
#pragma omp atomic compare capture
  if (x == e) {
    x = d;
  } else {
    v += x;
  }
// omp51-error@+6 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+5 {{expect an assignment statement 'v = x'}}
#pragma omp atomic compare capture
  if (x == e) {
    x = d;
  } else {
    v = d;
  }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected compound statement}}
#pragma omp atomic compare capture
  {}
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect a compound statement}}
#pragma omp atomic compare capture
  x = x > e ? e : x;
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect a 'if' statement}}
#pragma omp atomic compare capture
  { x = x > e ? e : x; }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect a form 'r = x == e; if (r) ...'}}
#pragma omp atomic compare capture
  { r = x == e; if (x == d) { x = e; } }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected assignment statement}}
#pragma omp atomic compare capture
  { r = x == e; if (r) { bbar(); } }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected assignment statement}}
#pragma omp atomic compare capture
  { r = x == e; if (r) { x += d; } }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected compound statement}}
#pragma omp atomic compare capture
  { r = x == e; if (r) {} }
// omp51-error@+5 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+4 {{expected exactly one expression statement}}
#pragma omp atomic compare capture
  {
    r = x == e;
    if (r) {
      x = d;
      v = x;
    }
  }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect '==' operator}}
#pragma omp atomic compare capture
  { r = x > e; if (r) { x = d; } }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect comparison in a form of 'x == e', 'e == x', 'x ordop expr', or 'expr ordop x'}}
#pragma omp atomic compare capture
  { r = d == e; if (r) { x = d; } }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected compound statement}}
#pragma omp atomic compare capture
  { r = x == e; if (r) { x = d; } else {} }
// omp51-error@+7 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+6 {{expected exactly one expression statement}}
#pragma omp atomic compare capture
  {
    r = x == e;
    if (r) {
      x = d;
    } else {
      v = x;
      d = e;
    }
  }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected assignment statement}}
#pragma omp atomic compare capture
  { r = x == e; if (r) { x = d; } else { bbar(); } }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected assignment statement}}
#pragma omp atomic compare capture
  { r = x == e; if (r) { x = d; } else { v += x; } }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect an assignment statement 'v = x'}}
#pragma omp atomic compare capture
  { r = x == e; if (r) { x = d; } else { v = d; } }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected assignment statement}}
#pragma omp atomic compare capture
  { v += x; if (x == e) { x = d; } }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expected assignment statement}}
#pragma omp atomic compare capture
  { if (x == e) { x = d; } v += x; }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect an assignment statement 'v = x'}}
#pragma omp atomic compare capture
  { v = d; if (x == e) { x = d; } }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect an assignment statement 'v = x'}}
#pragma omp atomic compare capture
  { if (x == e) { x = d; } v = d; }
// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect a 'if' statement}}
#pragma omp atomic compare capture
  { v = x; bbar(); }

// omp51-error@+3 {{the statement for 'atomic compare capture' must be a compound statement of form '{v = x; cond-up-stmt}', ''{cond-up-stmt v = x;}', '{if(x == e) {x = d;} else {v = x;}}', '{r = x == e; if(r) {x = d;}}', or '{r = x == e; if(r) {x = d;} else {v = x;}}', where 'cond-update-stmt' can have one of the following forms: 'if(expr ordop x) {x = expr;}', 'if(x ordop expr) {x = expr;}', 'if(x == e) {x = d;}', or 'if(e == x) {x = d;}' where 'x' is an lvalue expression with scalar type, 'expr', 'e', and 'd' are expressions with scalar type, and 'ordop' is one of '<' or '>'.}}
// omp51-note@+2 {{expect integer value}}
#pragma omp atomic compare capture
  { dr = x == e; if (dr) { x = d; } }
}
#endif
