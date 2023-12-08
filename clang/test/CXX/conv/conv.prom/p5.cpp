// RUN: %clang_cc1 -verify %s
// expected-no-diagnostics

// A prvalue for an integral bit-field can be converted to a prvalue of type
// int if int can represent all the values of the bit-field
struct X { long long e : 1; };
static_assert(sizeof(+X().e) == sizeof(int), "");
static_assert(sizeof(X().e + 1) == sizeof(int), "");
static_assert(sizeof(true ? X().e : 0) == sizeof(int), "");

enum E : long long { a = __LONG_LONG_MAX__ };
static_assert(sizeof(E{}) == sizeof(long long), "");

// If the bit-field has an enumerated type, it is treated as any other value of
// that [enumerated] type for promotion purposes.
struct Y { E e : 1; };
static_assert(sizeof(+Y().e) == sizeof(long long), "");
static_assert(sizeof(Y().e + 1) == sizeof(long long), "");
static_assert(sizeof(true ? Y().e : 0) == sizeof(long long), "");
