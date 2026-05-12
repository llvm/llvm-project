// RUN: %clang_cc1 -fsyntax-only -fms-compatibility -triple x86_64-windows-msvc -verify %s

typedef long long LONG_PTR;
typedef long LONG;
#define FIELD_OFFSET(type, field) ((LONG_PTR)&(((type *)0)->field))

struct S {
  int x;
  int y;
};

template<class T, bool = __builtin_choose_expr(FIELD_OFFSET(T, y) > 0, true, false)>
char probe(int);

template<class>
long probe(...);

static_assert(sizeof(probe<S>(0)) == sizeof(char), "");
// expected-error@-1 {{static assertion failed due to requirement 'sizeof (probe<S>(0)) == sizeof(char)'}}
// expected-note@-2 {{expression evaluates to '4 == 1'}}
