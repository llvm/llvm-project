// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

int foo() {
L1:
  foo();
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
    foo();
    goto L1; // expected-error {{use of undeclared label 'L1'}}
  }
  goto L2; // expected-error {{use of undeclared label 'L2'}}
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
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

int readint() {
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

int readS() {
  struct S a, b;
  // expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'read' clause}}
#pragma omp atomic read read
  // expected-error@+2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected expression of scalar type}}
  a = b;

  return a.a;
}

int writeint() {
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

int writeS() {
  struct S a, b;
  // expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'write' clause}}
#pragma omp atomic write write
  // expected-error@+2 {{the statement for 'atomic write' must be an expression statement of form 'x = expr;', where x is a lvalue expression with scalar type}}
  // expected-note@+1 {{expected expression of scalar type}}
  a = b;

  return a.a;
}

int updateint() {
  int a = 0, b = 0;
// Test for atomic update
#pragma omp atomic update
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected built-in binary or unary operator}}
  foo();
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected built-in binary operator}}
  a = b;
#pragma omp atomic update
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected one of '+', '*', '-', '/', '&', '^', '|', '<<', or '>>' built-in operations}}
  a = b || a;
#pragma omp atomic update
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected one of '+', '*', '-', '/', '&', '^', '|', '<<', or '>>' built-in operations}}
  a = a && b;
#pragma omp atomic update
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  a = (float)a + b;
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  a = 2 * b;
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
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

int captureint() {
  int a = 0, b = 0, c = 0;
// Test for atomic capture
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected compound statement}}
  ;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both l-value expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  foo();
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both l-value expressions with scalar type}}
  // expected-note@+1 {{expected built-in binary or unary operator}}
  a = b;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both l-value expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  a = b || a;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both l-value expressions with scalar type}}
  // expected-note@+1 {{expected one of '+', '*', '-', '/', '&', '^', '|', '<<', or '>>' built-in operations}}
  b = a = a && b;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both l-value expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  a = (float)a + b;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both l-value expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  a = 2 * b;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both l-value expressions with scalar type}}
  // expected-note@+1 {{expected assignment expression}}
  a = b + *&a;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected exactly two expression statements}}
  { a = b; }
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected exactly two expression statements}}
  {}
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of the first expression}}
  {a = b;a = b;}
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be a compound statement of form '{v = x; x binop= expr;}', '{x binop= expr; v = x;}', '{v = x; x = x binop expr;}', '{v = x; x = expr binop x;}', '{x = x binop expr; v = x;}', '{x = expr binop x; v = x;}' or '{v = x; x = expr;}', '{v = x; x++;}', '{v = x; ++x;}', '{++x; v = x;}', '{x++; v = x;}', '{v = x; x--;}', '{v = x; --x;}', '{--x; v = x;}', '{x--; v = x;}' where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of the first expression}}
  {a = b; a = b || a;}
#pragma omp atomic capture
  {b = a; a = a && b;}
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both l-value expressions with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  b = a = (float)a + b;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both l-value expressions with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  b = a = 2 * b;
#pragma omp atomic capture
  // expected-error@+2 {{the statement for 'atomic capture' must be an expression statement of form 'v = ++x;', 'v = --x;', 'v = x++;', 'v = x--;', 'v = x binop= expr;', 'v = x = x binop expr' or 'v = x = expr binop x', where x and v are both l-value expressions with scalar type}}
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

