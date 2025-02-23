// RUN: %clang_cc1 -fsyntax-only -std=c++20 -Wno-all -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions -fexperimental-bounds-safety-attributes -verify %s

#include <ptrcheck.h>

typedef unsigned size_t;
namespace std {
  template <typename CharT>
  struct basic_string {
    const CharT *data() const noexcept;
    CharT *data() noexcept;
    const CharT *c_str() const noexcept;
    size_t size() const noexcept;
    size_t length() const noexcept;
  };

  typedef basic_string<char> string;

  template <typename CharT>
  struct basic_string_view {
    basic_string_view(basic_string<CharT> str);
    const CharT *data() const noexcept;
    size_t size() const noexcept;
    size_t length() const noexcept;
  };

  typedef basic_string_view<char> string_view;
} // namespace std

void nt_parm(const char * __null_terminated);
const char * __null_terminated get_nt(const char *, size_t);

void basics(const char * cstr, size_t cstr_len, std::string cxxstr) {
  const char * __null_terminated p = "hello";   // safe init

  nt_parm(p);
  nt_parm(get_nt(cstr, cstr_len));

  const char * __null_terminated p2 = cxxstr.c_str(); // safe init
  const char * __null_terminated p3;

  p3 = cxxstr.c_str();         // safe assignment
  p3 = "hello";                // safe assignment
  p3 = p;                      // safe assignment
  p3 = get_nt(cstr, cstr_len); // safe assignment

  // expected-error@+1 {{initializing 'const char * __terminated_by(0)' (aka 'const char *') with an expression of incompatible type 'const char *' is an unsafe operation}}
  const char * __null_terminated p4 = cstr;  // warn

  // expected-error@+1 {{passing 'const char *' to parameter of incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  nt_parm(cstr);                             // warn
  // expected-error@+1 {{assigning to 'const char * __terminated_by(0)' (aka 'const char *') from incompatible type 'const char *' is an unsafe operation}}
  p4 = cstr;                                 // warn

  std::string_view view{cxxstr};

  // expected-error@+1 {{assigning to 'const char * __terminated_by(0)' (aka 'const char *') from incompatible type 'const char *' is an unsafe operation}}
  p4 = view.data();                          // warn
  // expected-error@+1 {{passing 'const char *' to parameter of incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  nt_parm(view.data());                      // warn

  const char * __null_terminated p5 = 0;                 // nullptr is ok
  // expected-error@+1 {{initializing 'const char * __terminated_by(0)' (aka 'const char *') with an expression of incompatible type 'const char *' is an unsafe operation}}
  const char * __null_terminated p6 = (const char *)1;   // other integer literal is unsafe

  // (non-0)-terminated pointer is NOT compatible with 'std::string::c_str':
  // expected-error@+1 {{initializing 'const char * __terminated_by(42)' (aka 'const char *') with an expression of incompatible type 'const char *' is an unsafe operation}}
  const char * __terminated_by(42) p7 = cxxstr.c_str();
}

void test_explicit_cast(char * p, const char * q) {
  // expected-error@+1 {{casting 'char *' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  const char * __null_terminated nt = (const char * __null_terminated) p;
  const char * __null_terminated nt2 = reinterpret_cast<const char * __null_terminated> (p); // FP for now

    // expected-error@+1 {{casting 'char *' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  nt = (const char * __null_terminated) p;
  nt2 = reinterpret_cast<const char * __null_terminated> (p); // FP for now
  // expected-error@+1 {{casting 'char *' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  nt2 = static_cast<const char * __null_terminated> (p);      // FP for now

  // expected-error@+1 {{casting 'const char *' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  const char * __null_terminated nt3 = (const char * __null_terminated) q;
  const char * __null_terminated nt4 = reinterpret_cast<const char * __null_terminated> (q); // FP for now

  // expected-error@+1 {{casting 'const char *' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  nt3 = (const char * __null_terminated) q;
  nt4 = reinterpret_cast<const char * __null_terminated> (q); // FP for now
  // expected-error@+1 {{casting 'const char *' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  nt4 = static_cast<const char * __null_terminated> (q);

  // OK cases for C-style casts
  char * __null_terminated nt_p;
  const char * __null_terminated nt_p2;
  int * __null_terminated nt_p3;
  const int * __null_terminated nt_p4;

  const char * __null_terminated nt5 = (const char * __null_terminated) nt_p;
  const char * __null_terminated nt6 = (const char * __null_terminated) nt_p2;
  const char * __null_terminated nt7 = (const char * __null_terminated) nt_p3;
  const char * __null_terminated nt8 = (const char * __null_terminated) nt_p4;

  nt5 = (const char * __null_terminated) nt_p;
  nt6 = (const char * __null_terminated) nt_p2;
  nt7 = (const char * __null_terminated) nt_p3;
  nt8 = (const char * __null_terminated) nt_p4;


  char * __null_terminated nt9 = (char * __null_terminated) nt_p;
  char * __null_terminated nt10 = (char * __null_terminated) nt_p2;
  char * __null_terminated nt11 = (char * __null_terminated) nt_p3;
  char * __null_terminated nt12 = (char * __null_terminated) nt_p4;

  nt9 = (char * __null_terminated) nt_p;
  nt10 = (char * __null_terminated) nt_p2;
  nt11 = (char * __null_terminated) nt_p3;
  nt12 = (char * __null_terminated) nt_p4;
}

const char * __null_terminated test_return(const char * p, char * q, std::string &str) {
  if (p)
    return p; // expected-error {{returning 'const char *' from a function with incompatible result type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  if (q)
    return q; // expected-error {{returning 'char *' from a function with incompatible result type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  return str.c_str();
}

void test_array(char * cstr) {
  const char arr[__null_terminated 3] = {'h', 'i', '\0'};
  // expected-error@+1 {{array 'arr2' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: 0; got 'i')}}
  const char arr2[__null_terminated 2] = {'h', 'i'};
  const char * __null_terminated arr3[] = {"hello", "world"};
  // expected-error@+1 {{initializing 'const char * __terminated_by(0)' (aka 'const char *') with an expression of incompatible type 'char *' is an unsafe operation}}
  const char * __null_terminated arr4[] = {"hello", "world", cstr};

  // expected-error@+1 {{assigning to 'const char * __terminated_by(0)' (aka 'const char *') from incompatible type 'char *' is an unsafe operation}}
  arr3[0] = cstr;
}

struct T {
  int a;
  const char * __null_terminated p;
  struct TT {
    int a;
    const char * __null_terminated p;
  } tt;
};
void test_compound(char * cstr) {
  std::string str;
  T t = {42, "hello"};
  T t2 = {.a = 42};
  T t3 = {.p = str.c_str()};
  // expected-error@+1 {{initializing 'const char * __terminated_by(0)' (aka 'const char *') with an expression of incompatible type 'char *' is an unsafe operation}}
  T t4 = {42, "hello", {.p = cstr}};

  // expected-error@+1 {{initializing 'const char * __terminated_by(0)' (aka 'const char *') with an expression of incompatible type 'char *' is an unsafe operation}}
  t4 = (struct T){42, "hello", {.p = cstr}};
}

  // expected-error@+3 {{initializing 'const char * __terminated_by(0)' (aka 'const char *') with an expression of incompatible type 'char *' is an unsafe operation}}
class C {
  const char * __null_terminated p;
  const char * __null_terminated q = (char *) 1; // warn
  struct T t;
public:
  // expected-error@+1 2{{initializing 'const char * __terminated_by(0)' (aka 'const char *') with an expression of incompatible type 'char *' is an unsafe operation}}
  C(char * p): p(p), t({0, p}) {};
  C(const char * __null_terminated p, struct T t);
};

void f(const C &c);
C test_class(char * cstr) {
  // expected-error@+2 {{passing 'char *' to parameter of incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  // expected-error@+1 {{initializing 'const char * __terminated_by(0)' (aka 'const char *') with an expression of incompatible type 'char *' is an unsafe operation}}
  C c{cstr, {0, cstr}};
  // expected-error@+2 {{passing 'char *' to parameter of incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  // expected-error@+1 {{initializing 'const char * __terminated_by(0)' (aka 'const char *') with an expression of incompatible type 'char *' is an unsafe operation}}
  C c1(cstr, {0, cstr});
  // expected-error@+2 {{passing 'char *' to parameter of incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  // expected-error@+1 {{initializing 'const char * __terminated_by(0)' (aka 'const char *') with an expression of incompatible type 'char *' is an unsafe operation}}
  C *c2 = new C(cstr, {0, cstr});
  // expected-error@+2 {{passing 'char *' to parameter of incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  // expected-error@+1 {{initializing 'const char * __terminated_by(0)' (aka 'const char *') with an expression of incompatible type 'char *' is an unsafe operation}}
  f(C{cstr, {0, cstr}});

  C("hello", {0, "hello"});
  if (1-1)
    return {cstr, {}}; // expected-error {{passing 'char *' to parameter of incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  return {"hello", {}};
}


// Test input/output __null_terminated parameter.
// expected-note@+1 {{candidate function not viable:}}
void g(const char * __null_terminated *p);
void test_output(const char * __null_terminated p) {
  const char * __null_terminated local_nt = p;
  const char * const_local;
  char * local;

  g(&local_nt); // safe
  // expected-error@+1 {{passing 'const char **' to parameter of incompatible type 'const char * __terminated_by(0)*' (aka 'const char **') that adds '__terminated_by' attribute is not allowed}}
  g(&const_local);
  // expected-error@+1 {{no matching function for call to 'g'}}
  g(&local);
}
