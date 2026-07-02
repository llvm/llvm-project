// RUN: %clang_cc1 -fsyntax-only -verify -fms-compatibility -triple x86_64-windows-msvc %s

typedef long long LONG_PTR;
typedef long LONG;
#define FIELD_OFFSET(type, field) ((LONG_PTR)&(((type *)0)->field))

struct S {
  int x;
  int y;
};

template<class T, LONG_PTR = FIELD_OFFSET(S, y)>
char probe(int);

template<class>
long probe(...);

static_assert(sizeof(probe<int>(0)) == sizeof(char), "");
// expected-error@-1 {{static assertion failed due to requirement 'sizeof (probe<int>(0)) == sizeof(char)'}}
// expected-note@-2 {{expression evaluates to '4 == 1'}}
