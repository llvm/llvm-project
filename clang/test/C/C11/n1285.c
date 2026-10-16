// RUN: %clang_cc1 -fsyntax-only -verify=expected,c -std=c99 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,c -std=c11 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cpp -std=c++11 -x c++ %s

/* WG14 N1285: Clang 21
 * Extending the lifetime of temporary objects (factored approach)
 *
 * This paper introduced the notion of an object with a temporary lifetime. Any
 * operation resulting in an rvalue of structure or union type which contains
 * an array results in an object with temporary lifetime.
 *
 * Even though this is a change for C11, we treat it as a DR and apply it
 * retroactively to earlier C language modes.
 */

// C11 6.2.4p8: A non-lvalue expression with structure or union type, where the
// structure or union contains a member with array type (including,
// recursively, members of all contained structures and unions) refers to an
// object with automatic storage duration and temporary lifetime. Its lifetime
// begins when the expression is evaluated and its initial value is the value
// of the expression. Its lifetime ends when the evaluation of the containing
// full expression or full declarator ends. Any attempt to modify an object
// with temporary lifetime results in undefined behavior.

struct X { int a[5]; };
struct X f(void);

union U { int a[10]; double d; };
union U g(void);

void sink(int *);

int func_return(void) {
  int *p = f().a; // expected-warning {{temporary whose address is used as value of local variable 'p' will be destroyed at the end of the full-expression}}
  p = f().a;      // expected-warning {{object backing the pointer 'p' will be destroyed at the end of the full-expression}}
  p = g().a;      // expected-warning {{object backing the pointer 'p' will be destroyed at the end of the full-expression}}
  sink(f().a);    // Ok
  return *f().a;  // Ok
}

int ternary(void) {
  int *p = (1 ? (struct X){ 0 } : f()).a; // expected-warning {{temporary whose address is used as value of local variable 'p' will be destroyed at the end of the full-expression}}
  int *r = (1 ? (union U){ 0 } : g()).a;  // expected-warning {{temporary whose address is used as value of local variable 'r' will be destroyed at the end of the full-expression}}
  p = (1 ? (struct X){ 0 } : f()).a;      // expected-warning {{object backing the pointer 'p' will be destroyed at the end of the full-expression}}
  sink((1 ? (struct X){ 0 } : f()).a);    // Ok

  // This intentionally gets one diagnostic in C and two in C++. In C, the
  // compound literal results in an lvalue, not an rvalue as it does in C++. So
  // only one branch results in a temporary in C but both branches do in C++.
  int *q = 1 ? (struct X){ 0 }.a : f().a; // expected-warning {{temporary whose address is used as value of local variable 'q' will be destroyed at the end of the full-expression}} \
                                             cpp-warning {{temporary whose address is used as value of local variable 'q' will be destroyed at the end of the full-expression}}
  q = 1 ? (struct X){ 0 }.a : f().a; // expected-warning {{object backing the pointer 'q' will be destroyed at the end of the full-expression}} \
                                        cpp-warning {{object backing the pointer 'q' will be destroyed at the end of the full-expression}}
  q = 1 ? (struct X){ 0 }.a : g().a; // expected-warning {{object backing the pointer 'q' will be destroyed at the end of the full-expression}} \
                                        cpp-warning {{object backing the pointer 'q' will be destroyed at the end of the full-expression}}
  sink(1 ? (struct X){ 0 }.a : f().a); // Ok
  return *(1 ? (struct X){ 0 }.a : f().a); // Ok
}

int comma(void) {
  struct X x;
  int *p = ((void)0, x).a; // c-warning {{temporary whose address is used as value of local variable 'p' will be destroyed at the end of the full-expression}}
  p = ((void)0, x).a;      // c-warning {{object backing the pointer 'p' will be destroyed at the end of the full-expression}}
  sink(((void)0, x).a);    // Ok
  return *(((void)0, x).a);// Ok
}

int cast(void) {
  struct X x;
  int *p = ((struct X)x).a;  // expected-warning {{temporary whose address is used as value of local variable 'p' will be destroyed at the end of the full-expression}}
  p = ((struct X)x).a;       // expected-warning {{object backing the pointer 'p' will be destroyed at the end of the full-expression}}
  sink(((struct X)x).a);     // Ok
  return *(((struct X)x).a); // Ok
}

int assign(void) {
  struct X x, s;
  int *p = (x = s).a;  // c-warning {{temporary whose address is used as value of local variable 'p' will be destroyed at the end of the full-expression}}
  p = (x = s).a;       // c-warning {{object backing the pointer 'p' will be destroyed at the end of the full-expression}}
  sink((x = s).a);     // Ok
  return *((x = s).a); // Ok
}
