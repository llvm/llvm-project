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
  S &operator=(int v) {
    a = v;
    return *this;
  }
  S &operator+=(const S &s) {
    a += s.a;
    return *this;
  }
};

template <class T>
T read() {
  T a = T(), b = T();
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
  // expected-error@+2 2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 2 {{expected built-in assignment operator}}
  a += b;
#pragma omp atomic read
  // expected-error@+2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected lvalue expression}}
  a = 0;
#pragma omp atomic read
  // expected-error@+2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected built-in assignment operator}}
  a = b;
  // expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'read' clause}}
#pragma omp atomic read read
  // expected-error@+2 {{the statement for 'atomic read' must be an expression statement of form 'v = x;', where v and x are both lvalue expressions with scalar type}}
  // expected-note@+1 {{expected built-in assignment operator}}
  a = b;

  return a;
}

int read() {
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

  // expected-note@+2 {{in instantiation of function template specialization 'read<S>' requested here}}
  // expected-note@+1 {{in instantiation of function template specialization 'read<int>' requested here}}
  return read<int>() + read<S>().a;
}

template <class T>
T write() {
  T a, b = 0;
// Test for atomic write
#pragma omp atomic write
  // expected-error@+2 {{the statement for 'atomic write' must be an expression statement of form 'x = expr;', where x is a lvalue expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;
// expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'write' clause}}
#pragma omp atomic write write
  a = b;
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

  return T();
}

int write() {
  int a, b = 0;
// Test for atomic write
#pragma omp atomic write
  // expected-error@+2 {{the statement for 'atomic write' must be an expression statement of form 'x = expr;', where x is a lvalue expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;
// expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'write' clause}}
#pragma omp atomic write write
  a = b;
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
  a = foo();

  // expected-note@+1 {{in instantiation of function template specialization 'write<int>' requested here}}
  return write<int>();
}

template <class T>
T update() {
  T a = 0, b = 0, c = 0;
// Test for atomic update
#pragma omp atomic update
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;
// expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'update' clause}}
#pragma omp atomic update update
  a += b;
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
  a = float(a) + b;
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  a = 2 * b;
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  a = b + *&a;
#pragma omp atomic
  *&a = b * *&a;
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

#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;

  return T();
}

int update() {
  int a, b = 0;
// Test for atomic update
#pragma omp atomic update
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;
// expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'update' clause}}
#pragma omp atomic update update
  a += b;
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
  a = float(a) + b;
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  a = 2 * b;
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected in right hand side of expression}}
  a = b + *&a;
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
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;

  return update<int>();
}

template <class T>
T capture() {
  T a = 0, b = 0, c = 0;
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

  return T();
}

int capture() {
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
  {++a;c = a;}
#pragma omp atomic capture
  {c = a;a--;}
#pragma omp atomic capture
  {--a;c = a;}
#pragma omp atomic capture
  {c = a; a += b;}
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

  // expected-note@+1 {{in instantiation of function template specialization 'capture<int>' requested here}}
  return capture<int>();
}

template <class T>
T seq_cst() {
  T a, b = 0;
// Test for atomic seq_cst
#pragma omp atomic seq_cst
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;
// expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'seq_cst' clause}}
#pragma omp atomic seq_cst seq_cst
  a += b;

#pragma omp atomic update seq_cst
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;

  return T();
}

int seq_cst() {
  int a, b = 0;
// Test for atomic seq_cst
#pragma omp atomic seq_cst
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;
// expected-error@+1 {{directive '#pragma omp atomic' cannot contain more than one 'seq_cst' clause}}
#pragma omp atomic seq_cst seq_cst
  a += b;

#pragma omp atomic update seq_cst
  // expected-error@+2 {{the statement for 'atomic update' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  ;

 return seq_cst<int>();
}

template <class T>
T mixed() {
  T a, b = T();
// expected-error@+2 2 {{directive '#pragma omp atomic' cannot contain more than one 'read', 'write', 'update' or 'capture' clause}}
// expected-note@+1 2 {{'read' clause used here}}
#pragma omp atomic read write
  a = b;
// expected-error@+2 2 {{directive '#pragma omp atomic' cannot contain more than one 'read', 'write', 'update' or 'capture' clause}}
// expected-note@+1 2 {{'write' clause used here}}
#pragma omp atomic write read
  a = b;
// expected-error@+2 2 {{directive '#pragma omp atomic' cannot contain more than one 'read', 'write', 'update' or 'capture' clause}}
// expected-note@+1 2 {{'update' clause used here}}
#pragma omp atomic update read
  a += b;
// expected-error@+2 2 {{directive '#pragma omp atomic' cannot contain more than one 'read', 'write', 'update' or 'capture' clause}}
// expected-note@+1 2 {{'capture' clause used here}}
#pragma omp atomic capture read
  a = ++b;
  return T();
}

int mixed() {
  int a, b = 0;
// expected-error@+2 {{directive '#pragma omp atomic' cannot contain more than one 'read', 'write', 'update' or 'capture' clause}}
// expected-note@+1 {{'read' clause used here}}
#pragma omp atomic read write
  a = b;
// expected-error@+2 {{directive '#pragma omp atomic' cannot contain more than one 'read', 'write', 'update' or 'capture' clause}}
// expected-note@+1 {{'write' clause used here}}
#pragma omp atomic write read
  a = b;
// expected-error@+2 {{directive '#pragma omp atomic' cannot contain more than one 'read', 'write', 'update' or 'capture' clause}}
// expected-note@+1 {{'write' clause used here}}
#pragma omp atomic write update
  a = b;
// expected-error@+2 {{directive '#pragma omp atomic' cannot contain more than one 'read', 'write', 'update' or 'capture' clause}}
// expected-note@+1 {{'write' clause used here}}
#pragma omp atomic write capture
  a = b;
  // expected-note@+1 {{in instantiation of function template specialization 'mixed<int>' requested here}}
  return mixed<int>();
}

