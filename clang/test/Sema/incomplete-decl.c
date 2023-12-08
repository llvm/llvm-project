// RUN: %clang_cc1 -fsyntax-only -verify=c,expected %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify=cxx,expected %s



struct foo; // c-note 5 {{forward declaration of 'struct foo'}} \
               cxx-note 3 {{forward declaration of 'foo'}}

void b;  // expected-error {{variable has incomplete type 'void'}}
struct foo f; // c-error {{tentative definition has type 'struct foo' that is never completed}} \
                 cxx-error {{variable has incomplete type 'struct foo'}}

static void c; // expected-error {{variable has incomplete type 'void'}}
static struct foo g;  // c-warning {{tentative definition of variable with internal linkage has incomplete non-array type 'struct foo'}} \
                         c-error {{tentative definition has type 'struct foo' that is never completed}} \
                         cxx-error {{variable has incomplete type 'struct foo'}}

extern void d; // cxx-error {{variable has incomplete type 'void'}}
extern struct foo e;

int ary[]; // c-warning {{tentative array definition assumed to have one element}} \
              cxx-error {{definition of variable with array type needs an explicit size or an initializer}}
struct foo bary[]; // c-error {{array has incomplete element type 'struct foo'}} \
                      cxx-error {{definition of variable with array type needs an explicit size or an initializer}}

void func(void) {
  int ary[]; // expected-error {{definition of variable with array type needs an explicit size or an initializer}}
  void b; // expected-error {{variable has incomplete type 'void'}}
  struct foo f; // expected-error {{variable has incomplete type 'struct foo'}}
}

int h[]; // c-warning {{tentative array definition assumed to have one element}} \
            cxx-error {{definition of variable with array type needs an explicit size or an initializer}}
int (*i)[] = &h+1; // c-error {{arithmetic on a pointer to an incomplete type 'int[]'}}

struct bar j = {1}; // expected-error {{variable has incomplete type 'struct bar'}} \
                       c-note {{forward declaration of 'struct bar'}} \
                       cxx-note 2 {{forward declaration of 'bar'}}

struct bar k; // cxx-error {{variable has incomplete type 'struct bar'}}
struct bar { int a; };

struct x y; //c-note 2 {{forward declaration of 'struct x'}} \
              cxx-error {{variable has incomplete type 'struct x'}} \
              cxx-note {{forward declaration of 'x'}}
void foo() {
  (void)(1 ? y : y); // c-error 2 {{incomplete type 'struct x' where a complete type is required}}
}
struct x{
  int a;
};
