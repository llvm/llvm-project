// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

char *const_cast_test(const char *var)
{
  return const_cast<char*>(var);
}

struct A {
  virtual ~A() {}
};

struct B : public A {
};

struct B *dynamic_cast_test(struct A *a)
{
  return dynamic_cast<struct B*>(a);
}

char *reinterpret_cast_test()
{
  return reinterpret_cast<char*>(0xdeadbeef);
}

double static_cast_test(int i)
{
  return static_cast<double>(i);
}

char postfix_expr_test()
{
  return reinterpret_cast<char*>(0xdeadbeef)[0];
}

// This was being incorrectly tentatively parsed.
namespace test1 {
  template <class T> class A {}; // expected-note 2{{here}}
  void foo() { A<int>(*(A<int>*)0); } // expected-warning {{binding dereferenced null pointer to reference has undefined behavior}}
}

typedef char* c;
typedef A* a;
void test2(char x, struct B * b) {
  (void)const_cast<::c>(&x);
#if __cplusplus <= 199711L
  // expected-error@-2 {{found '<::' after a const_cast which forms the digraph '<:' (aka '[') and a ':', did you mean '< ::'?}}
#endif

  (void)dynamic_cast<::a>(b);
#if __cplusplus <= 199711L
  // expected-error@-2 {{found '<::' after a dynamic_cast which forms the digraph '<:' (aka '[') and a ':', did you mean '< ::'?}}
#endif

  (void)reinterpret_cast<::c>(x);
#if __cplusplus <= 199711L
  // expected-error@-2 {{found '<::' after a reinterpret_cast which forms the digraph '<:' (aka '[') and a ':', did you mean '< ::'?}}
#endif

  (void)static_cast<::c>(&x);
#if __cplusplus <= 199711L
  // expected-error@-2 {{found '<::' after a static_cast which forms the digraph '<:' (aka '[') and a ':', did you mean '< ::'?}}
#endif

  // Do not do digraph correction.
  (void)static_cast<: :c>(&x); //\
       expected-error {{expected '<' after 'static_cast'}} \
       expected-error {{expected expression}}\
       expected-error {{expected ']'}}\
       expected-note {{to match this '['}}
  (void)static_cast<: // expected-error {{expected '<' after 'static_cast'}} \
                         expected-note {{to match this '['}}
  :c>(&x); // expected-error {{expected expression}} \
              expected-error {{expected ']'}}
#define LC <:
#define C :
  test1::A LC:B> c; // expected-error {{class template 'test1::A' requires template arguments}} expected-error 2{{}}
  (void)static_cast LC:c>(&x); // expected-error {{expected '<' after 'static_cast'}} expected-error 2{{}} expected-note{{}}
  test1::A<:C B> d; // expected-error {{class template 'test1::A' requires template arguments}} expected-error 2{{}}
  (void)static_cast<:C c>(&x); // expected-error {{expected '<' after 'static_cast'}} expected-error 2{{}} expected-note{{}}

#define LCC <::
  test1::A LCC B> e;
#if __cplusplus <= 199711L
  // expected-error@-2 {{found '<::' after a template name which forms the digraph '<:' (aka '[') and a ':', did you mean '< ::'?}}
#endif

  (void)static_cast LCC c>(&x);
#if __cplusplus <= 199711L
  // expected-error@-2 {{found '<::' after a static_cast which forms the digraph '<:' (aka '[') and a ':', did you mean '< ::'?}}
#endif
}

                               // This note comes from "::D[:F> A5;"
template <class T> class D {}; // expected-note{{template is declared here}}
template <class T> void E() {};
class F {};

void test3() {
  ::D<::F> A1;
#if __cplusplus <= 199711L
  // expected-error@-2 {{found '<::' after a template name which forms the digraph '<:' (aka '[') and a ':', did you mean '< ::'?}}
#endif

  D<::F> A2;
#if __cplusplus <= 199711L
  // expected-error@-2 {{found '<::' after a template name which forms the digraph '<:' (aka '[') and a ':', did you mean '< ::'?}}
#endif

  ::E<::F>();
#if __cplusplus <= 199711L
  // expected-error@-2 {{found '<::' after a template name which forms the digraph '<:' (aka '[') and a ':', did you mean '< ::'?}}
#endif

  E<::F>();
#if __cplusplus <= 199711L
  // expected-error@-2 {{found '<::' after a template name which forms the digraph '<:' (aka '[') and a ':', did you mean '< ::'?}}
#endif

  ::D< ::F> A3;
  D< ::F> A4;
  ::E< ::F>();
  E< ::F>();

  // Make sure that parser doesn't expand '[:' to '< ::'
  ::D[:F> A5; // expected-error {{class template '::D' requires template arguments}} \
              // expected-error {{expected expression}} \
              // expected-error {{expected unqualified-id}}
}

// Ensure that a C-style cast doesn't turn off colon protection.
void PR19748() {
  struct A {};
  int A = 0, b;
  int test1 = true ? (int)A : b;

  struct f {};
  extern B f(), (*p)();
  (true ? (B(*)())f : p)();
}

void PR19751(int n) {
  struct T { void operator++(int); };
  (T())++; // ok, not an ill-formed cast to function type
  (T())++n; // expected-error {{C-style cast from 'int' to 'T ()' is not allowed}}
}

// PR13619. Must be at end of file.
int n = reinterpret_cast // expected-error {{expected '<'}} expected-error {{expected ';'}}
