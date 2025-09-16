// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=c,bs %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=c,bs %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -fexperimental-bounds-safety-attributes -Wunsafe-buffer-usage -verify=cxx,ubu %s

#include <ptrcheck.h>
#include <stddef.h>

#ifndef __cplusplus
typedef __CHAR16_TYPE__ char16_t;
typedef __CHAR32_TYPE__ char32_t;
#endif

struct Foo {
  int a[__null_terminated 2];
  int b[__terminated_by(42) 2];
};

struct Bar {
  int a[__null_terminated 2];
};

struct Baz {
  char a[__terminated_by('X') 3];
  char b[__null_terminated 3];
};

struct Qux {
  struct Foo foo;
  struct Bar bar;
  struct Baz baz;
};

struct Quux {
  const char *__null_terminated p;
};

void explicit_const_init(void) {
  // ok
  int a1[__null_terminated 3] = {1, 2, 0};

  // ubu-warning@+2{{array 'a2' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '3')}}
  // bs-error@+1{{array 'a2' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '3')}}
  int a2[__null_terminated 3] = {1, 2, 3};

  // ok
  int a3[__terminated_by(42) 3] = {1, 2, 42};

  // ubu-warning@+2{{array 'a4' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  // bs-error@+1{{array 'a4' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  int a4[__terminated_by(42) 3] = {1, 2, 0};

  // ok
  char s1[__null_terminated 3] = "HI";

  // ubu-warning@+2{{array 's2' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  // bs-error@+1{{array 's2' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  char s2[__terminated_by('X') 3] = "HI";

  // ok
  struct Foo foo1 = {{1, 0}, {2, 42}};

  // ubu-warning@+2{{array 'foo2.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '1')}}
  // bs-error@+1{{array 'foo2.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '1')}}
  struct Foo foo2 = {{1, 1}, {2, 42}};

  // ubu-warning@+4{{array 'foo3.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '1')}}
  // ubu-warning@+3{{array 'foo3.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '2')}}
  // bs-error@+2{{array 'foo3.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '1')}}
  // bs-error@+1{{array 'foo3.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '2')}}
  struct Foo foo3 = {{1, 1}, {2, 2}};

  // ok
  struct Qux qux1 = {{{1, 0}, {2, 42}}, {{1, 0}}, {{'Z', 'Y', 'X'}, "HI"}};

  // ubu-warning@+10{{array 'qux2.foo.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '1')}}
  // ubu-warning@+9{{array 'qux2.foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '2')}}
  // ubu-warning@+8{{array 'qux2.bar.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '3')}}
  // ubu-warning@+7{{array 'qux2.baz.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  // ubu-warning@+6{{array 'qux2.baz.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got ''X'')}}
  // bs-error@+5{{array 'qux2.foo.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '1')}}
  // bs-error@+4{{array 'qux2.foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '2')}}
  // bs-error@+3{{array 'qux2.bar.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '3')}}
  // bs-error@+2{{array 'qux2.baz.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  // bs-error@+1{{array 'qux2.baz.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got ''X'')}}
  struct Qux qux2 = {{{1, 1}, {2, 2}}, {{1, 3}}, {"HI", {'Z', 'Y', 'X'}}};
}

void explicit_const_init_excess(void) {
  // cxx-error@+2{{excess elements in array initializer}}
  // c-warning@+1{{excess elements in array initializer}}
  int a1[__null_terminated 3] = {1, 2, 0, 4};

  // cxx-error@+3{{excess elements in array initializer}}
  // c-warning@+2{{excess elements in array initializer}}
  // bs-error@+1{{array 'a2' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '3')}}
  int a2[__null_terminated 3] = {1, 2, 3, 4};

  // cxx-error@+2{{excess elements in array initializer}}
  // c-warning@+1{{excess elements in array initializer}}
  int a3[__terminated_by(42) 3] = {1, 2, 42, 4};

  // cxx-error@+3{{excess elements in array initializer}}
  // c-warning@+2{{excess elements in array initializer}}
  // bs-error@+1{{array 'a4' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  int a4[__terminated_by(42) 3] = {1, 2, 0, 42};

  // cxx-error@+1{{initializer-string for char array is too long}}
  char s1[__null_terminated 3] = "HI\0";

  // cxx-error@+3{{initializer-string for char array is too long}}
  // ubu-warning@+2{{array 's2' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got ''J'')}}
  // bs-error@+1{{array 's2' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got ''J'')}}
  char s2[__terminated_by('X') 3] = "HEJ";

  // cxx-error@+2{{initializer-string for char array is too long}}
  // c-warning@+1{{initializer-string for char array is too long}}
  char s3[__null_terminated 3] = "HI\0HI";

  // cxx-error@+4{{initializer-string for char array is too long}}
  // c-warning@+3{{initializer-string for char array is too long}}
  // ubu-warning@+2{{array 's4' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got ''C'')}}
  // bs-error@+1{{array 's4' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got ''C'')}}
  char s4[__terminated_by('X') 3] = "ABCD";

  // cxx-error@+2 2{{excess elements in array initializer}}
  // c-warning@+1 2{{excess elements in array initializer}}
  struct Foo foo1 = {{1, 0, 1}, {2, 42, 0}};

  // cxx-error@+3{{excess elements in array initializer}}
  // c-warning@+2{{excess elements in array initializer}}
  // bs-error@+1{{array 'foo2.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '1')}}
  struct Foo foo2 = {{1, 1, 0}, {2, 42}};

  // cxx-error@+4 2{{excess elements in array initializer}}
  // c-warning@+3 2{{excess elements in array initializer}}
  // bs-error@+2{{array 'foo3.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '1')}}
  // bs-error@+1{{array 'foo3.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '2')}}
  struct Foo foo3 = {{1, 1, 0}, {2, 2, 42}};

  // cxx-error@+1 2{{initializer-string for char array is too long}}
  struct Baz baz1 = {"ZYX", "HI\0"};

  // cxx-error@+2 2{{initializer-string for char array is too long}}
  // c-warning@+1 2{{initializer-string for char array is too long}}
  struct Baz baz2 = {"ZYXW", "HI\0HI"};

  // cxx-error@+6 2{{initializer-string for char array is too long}}
  // c-warning@+5 2{{initializer-string for char array is too long}}
  // ubu-warning@+4{{array 'baz3.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got ''C'')}}
  // ubu-warning@+3{{array 'baz3.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got ''X'')}}
  // bs-error@+2{{array 'baz3.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got ''C'')}}
  // bs-error@+1{{array 'baz3.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got ''X'')}}
  struct Baz baz3 = {"ABCD", "ZYXW"};

  // cxx-error@+3{{initializer-string for char array is too long}}
  // cxx-error@+2 4{{excess elements in array initializer}}
  // c-warning@+1 4{{excess elements in array initializer}}
  struct Qux qux1 = {{{1, 0, 1}, {2, 42, 0}}, {{1, 0, 1}}, {{'Z', 'Y', 'X', 'W'}, "HI\0"}};

  // cxx-error@+8{{initializer-string for char array is too long}}
  // cxx-error@+7 3{{excess elements in array initializer}}
  // c-warning@+6 3{{excess elements in array initializer}}
  // bs-error@+5{{array 'qux2.foo.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '1')}}
  // bs-error@+4{{array 'qux2.foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '2')}}
  // bs-error@+3{{array 'qux2.bar.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '3')}}
  // bs-error@+2{{array 'qux2.baz.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got ''A'')}}
  // bs-error@+1{{array 'qux2.baz.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got ''J'')}}
  struct Qux qux2 = {{{1, 1, 0}, {2, 2}}, {{1, 3, 0}}, {{'Z', 'Y', 'A', 'W'}, "HEJ"}};
}

void implicit_init(void) {
  // ok
  int a1[__null_terminated 3] = {};

  // ok
  int a2[__null_terminated 3] = {1, 2};

  // ubu-warning@+2{{array 'a3' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  // bs-error@+1{{array 'a3' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  int a3[__terminated_by(42) 3] = {};

  // ubu-warning@+2{{array 'a4' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  // bs-error@+1{{array 'a4' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  int a4[__terminated_by(42) 3] = {1, 2};

  // ok
  char s1[__null_terminated 3] = "";

  // ok
  char s2[__null_terminated 3] = "X";

  // ubu-warning@+2{{array 'foo1.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  // bs-error@+1{{array 'foo1.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  struct Foo foo1 = {};

  // ubu-warning@+2{{array 'foo2.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  // bs-error@+1{{array 'foo2.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  struct Foo foo2 = {{}};

  // ubu-warning@+2{{array 'foo3.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  // bs-error@+1{{array 'foo3.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  struct Foo foo3 = {{}, {}};

  // ok
  struct Foo foo4 = {{}, {1, 42}};

  // ok
  struct Foo foo5 = {{1}, {1, 42}};

  // ok
  struct Bar bar1 = {};

  // ok
  struct Bar bar2 = {{}};

  // ubu-warning@+2{{array 'baz1.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  // bs-error@+1{{array 'baz1.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  struct Baz baz1 = {};

  // ubu-warning@+2{{array 'baz2.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  // bs-error@+1{{array 'baz2.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  struct Baz baz2 = {{}};

  // ubu-warning@+2{{array 'baz3.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  // bs-error@+1{{array 'baz3.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  struct Baz baz3 = {{}, {}};

  // cxx-error@+1{{initializer-string for char array is too long}}
  struct Baz baz4 = {"ZYX"};

  // ubu-warning@+4{{array 'qux1.foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  // ubu-warning@+3{{array 'qux1.baz.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  // bs-error@+2{{array 'qux1.foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  // bs-error@+1{{array 'qux1.baz.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  struct Qux qux1 = {};

  // ubu-warning@+4{{array 'qux2.foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  // ubu-warning@+3{{array 'qux2.baz.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  // bs-error@+2{{array 'qux2.foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  // bs-error@+1{{array 'qux2.baz.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  struct Qux qux2 = {{}, {}, {}};

  // ubu-warning@+4{{array 'qux3.foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  // ubu-warning@+3{{array 'qux3.baz.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  // bs-error@+2{{array 'qux3.foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  // bs-error@+1{{array 'qux3.baz.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  struct Qux qux3 = {{{}, {}}, {{}}, {{}, {}}};

  // ok
  struct Qux qux4 = {{{}, {1, 42}}, {}, {{'Z', 'Y', 'X'}}};

  // ok
  struct Qux qux5 = {{{}, {1, 42}}, {}, {{'Z', 'Y', 'X'}, {}}};
}

void explicit_non_const_init(int i, char c) {
  // ubu-warning@+2{{terminator in array 'a1' must be a compile-time constant}}
  // bs-error@+1{{terminator in array 'a1' must be a compile-time constant}}
  int a1[__null_terminated 3] = {1, 2, i};

  // ok
  int a2[__null_terminated 3] = {i, i, 0};

  // ubu-warning@+10{{terminator in array 'qux1.foo.a' must be a compile-time constant}}
  // ubu-warning@+9{{terminator in array 'qux1.foo.b' must be a compile-time constant}}
  // ubu-warning@+8{{terminator in array 'qux1.bar.a' must be a compile-time constant}}
  // ubu-warning@+7{{terminator in array 'qux1.baz.a' must be a compile-time constant}}
  // ubu-warning@+6{{terminator in array 'qux1.baz.b' must be a compile-time constant}}
  // bs-error@+5{{terminator in array 'qux1.foo.a' must be a compile-time constant}}
  // bs-error@+4{{terminator in array 'qux1.foo.b' must be a compile-time constant}}
  // bs-error@+3{{terminator in array 'qux1.bar.a' must be a compile-time constant}}
  // bs-error@+2{{terminator in array 'qux1.baz.a' must be a compile-time constant}}
  // bs-error@+1{{terminator in array 'qux1.baz.b' must be a compile-time constant}}
  struct Qux qux1 = {{{0, i}, {0, i}}, {{0, i}}, {{'A', 'A', c}, {'A', 'A', c}}};

  // ok
  struct Qux qux2 = {{{i, 0}, {i, 42}}, {{i, 0}}, {{c, c, 'X'}, {c, c, '\0'}}};
}

void incomplete_array_init(void) {
  // ok
  int a1[__null_terminated] = {1, 2, 0};

  // ubu-warning@+2{{array 'a2' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '3')}}
  // bs-error@+1{{array 'a2' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '3')}}
  int a2[__null_terminated] = {1, 2, 3};

  // cxx-error@+1{{array initializer must be an initializer list}}
  int a3[__null_terminated] = (int[3]){1, 2, 0};

  // cxx-error@+2{{array initializer must be an initializer list}}
  // bs-error@+1{{array 'a4' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '3')}}
  int a4[__null_terminated] = (int[3]){1, 2, 3};

  // ok
  char s1[__null_terminated] = "HI";

  // ubu-warning@+2{{array 's2' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  // bs-error@+1{{array 's2' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0')}}
  char s2[__terminated_by('X')] = "HI";
}

void string_literal_init(void) {
  // ok
  char s1[__null_terminated 3] = "HI";

  // ok
  wchar_t s2[__null_terminated 3] = L"HI";

  // ok
  char16_t s3[__null_terminated 3] = u"HI";

  // ok
  char32_t s4[__null_terminated 3] = U"HI";

  // cxx-error@+1{{initializer-string for char array is too long}}
  char s5[__terminated_by('X') 3] = "ZYX";

  // cxx-error@+1{{initializer-string for char array is too long}}
  wchar_t s6[__terminated_by(L'X') 3] = L"ZYX";

  // cxx-error@+1{{initializer-string for char array is too long}}
  char16_t s7[__terminated_by(u'X') 3] = u"ZYX";

  // cxx-error@+1{{initializer-string for char array is too long}}
  char32_t s8[__terminated_by(U'X') 3] = U"ZYX";

  // cxx-error@+3{{initializer-string for char array is too long}}
  // ubu-warning@+2{{array 's9' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got ''C'')}}
  // bs-error@+1{{array 's9' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got ''C'')}}
  char s9[__terminated_by('X') 3] = "ABC";

  // cxx-error@+3{{initializer-string for char array is too long}}
  // ubu-warning@+2{{array 's10' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: 'L'X''; got 'L'C'')}}
  // bs-error@+1{{array 's10' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: 'L'X''; got 'L'C'')}}
  wchar_t s10[__terminated_by(L'X') 3] = L"ABC";

  // cxx-error@+3{{initializer-string for char array is too long}}
  // ubu-warning@+2{{array 's11' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: 'u'X''; got 'u'C'')}}
  // bs-error@+1{{array 's11' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: 'u'X''; got 'u'C'')}}
  char16_t s11[__terminated_by(u'X') 3] = u"ABC";

  // cxx-error@+3{{initializer-string for char array is too long}}
  // ubu-warning@+2{{array 's12' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: 'U'X''; got 'U'C'')}}
  // bs-error@+1{{array 's12' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: 'U'X''; got 'U'C'')}}
  char32_t s12[__terminated_by(U'X') 3] = U"ABC";

  // cxx-error@+1{{initializer-string for char array is too long}}
  wchar_t s13[__terminated_by(U'\U0010F00C') 3] = L"\U0010F00A\U0010F00B\U0010F00C";

  // cxx-error@+1{{initializer-string for char array is too long}}
  char16_t s14[__terminated_by(u'\u1122') 3] = u"\u1120\u1121\u1122";

  // cxx-error@+1{{initializer-string for char array is too long}}
  char32_t s15[__terminated_by(U'\U0010F00C') 3] = U"\U0010F00A\U0010F00B\U0010F00C";
}

void sign_test(void) {
  // ok
  signed int a1[__terminated_by(-1) 2] = {0, -1};

  // cxx-note@+2{{insert an explicit cast to silence this issue}}
  // cxx-error@+1{{constant expression evaluates to -1 which cannot be narrowed to type 'unsigned int'}}
  unsigned int a2[__terminated_by(-1) 2] = {0, -1};

  // cxx-note@+2{{insert an explicit cast to silence this issue}}
  // cxx-error@+1{{constant expression evaluates to -1 which cannot be narrowed to type 'unsigned long long'}}
  unsigned long long a3[__terminated_by(-1) 2] = {0, -1};
}

void array_of_pointers(void) {
  // ok
  const char *__null_terminated a1[__null_terminated 3] = {"foo", "bar", 0};

  // ubu-warning@+2{{terminator in array 'a2' must be a compile-time constant}}
  // bs-error@+1{{terminator in array 'a2' must be a compile-time constant}}
  const char *__null_terminated a2[__null_terminated 3] = {"foo", "bar", "baz"};

  // ok
  const char *__null_terminated a3[__null_terminated 3] = {"foo", 0, 0};

  // ok
  const char *__null_terminated a4[__null_terminated 3] = {};

  // ok
  const char *__null_terminated a5[__null_terminated 3] = {0};

  // ok
  const char *__null_terminated a6[__null_terminated 3] = {"foo"};

  // ok
  const char *__null_terminated a7[3] = {};

  // ok
  const char *__null_terminated a8[3] = {0};

  // ok
  const char *__null_terminated a9[3] = {"foo"};
}

void foo_as_param(struct Foo foo);
void bar_as_param(struct Bar bar);
void quux_as_param(struct Quux quux);

void as_param(void) {
  // ubu-warning@+2{{array 'foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  // bs-error@+1{{array 'foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  foo_as_param((struct Foo){});

  // ubu-warning@+4{{array 'foo.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '2')}}
  // ubu-warning@+3{{array 'foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '4')}}
  // bs-error@+2{{array 'foo.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '2')}}
  // bs-error@+1{{array 'foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '4')}}
  foo_as_param((struct Foo){{1, 2}, {3, 4}});

  // ok
  foo_as_param((struct Foo){{1, 0}, {3, 42}});

  // ok
  bar_as_param((struct Bar){});

  // ubu-warning@+2{{array 'bar.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '2')}}
  // bs-error@+1{{array 'bar.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '2')}}
  bar_as_param((struct Bar){{1, 2}});

  // ok
  bar_as_param((struct Bar){{1, 0}});
}

struct Foo foo_as_ret(void) {
  // ubu-warning@+2{{array '.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  // bs-error@+1{{array '.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  return (struct Foo){};

  // ubu-warning@+4{{array '.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '2')}}
  // ubu-warning@+3{{array '.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '4')}}
  // bs-error@+2{{array '.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '2')}}
  // bs-error@+1{{array '.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '4')}}
  return (struct Foo){{1, 2}, {3, 4}};

  // ok
  return (struct Foo){{1, 0}, {3, 42}};
}

struct Bar bar_as_ret(void) {
  // ok
  return (struct Bar){};

  // ubu-warning@+2{{array '.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '2')}}
  // bs-error@+1{{array '.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '0'; got '2')}}
  return (struct Bar){{1, 2}};

  // ok
  return (struct Bar){{1, 0}};
}

struct Foo copy_init(void) {
  struct Foo foo1 = {{1, 0}, {2, 42}};
  struct Bar bar1 = {};

  // cxx-error@+1 2{{initializer-string for char array is too long}}
  struct Baz baz1 = {"ZYX", "HI\0"};

  // ok
  struct Foo foo2 = foo1;

  // ok
  struct Qux qux1 = { .foo = foo1, .bar = bar1, .baz = baz1 };

  // ok
  struct Qux qux2 = { .foo = foo1, /* .bar = ... */ .baz = baz1 };

  // ubu-warning@+2{{array 'qux3.baz.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0'}}
  // bs-error@+1{{array 'qux3.baz.a' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: ''X''; got '0'}}
  struct Qux qux3 = { .foo = foo1, .bar = bar1 /* .baz = ... */ };

  // ok
  foo_as_param(foo1);

  // ok
  return foo1;
}

struct Quux copy_init2(void) {
  struct Quux quux1 = { "Hello world" };

  // ok
  struct Quux quux2 = quux1;

  // ok
  quux_as_param(quux1);

  // ok
  return quux1;
}
