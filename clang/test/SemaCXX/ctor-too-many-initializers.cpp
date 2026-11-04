// RUN: %clang_cc1 -fsyntax-only -verify %s

// Note: This test is quite time consuming(~5s on a release-with-asserts), so if
// the limit increases at all we should consider just deleting this test.

#define F0(p) int p;
#define F1(p) F0(p##0) F0(p##1) F0(p##2) F0(p##3)
#define F2(p) F1(p##0) F1(p##1) F1(p##2) F1(p##3)
#define F3(p) F2(p##0) F2(p##1) F2(p##2) F2(p##3)
#define F4(p) F3(p##0) F3(p##1) F3(p##2) F3(p##3)
#define F5(p) F4(p##0) F4(p##1) F4(p##2) F4(p##3)
#define F6(p) F5(p##0) F5(p##1) F5(p##2) F5(p##3)
#define F7(p) F6(p##0) F6(p##1) F6(p##2) F6(p##3)
#define F8(p) F7(p##0) F7(p##1) F7(p##2) F7(p##3)
#define F9(p) F8(p##0) F8(p##1) F8(p##2) F8(p##3)

// expected-error@+1 {{constructor has too many base and member initializers (524288); the maximum is 524287}}
struct TooMany {
  F9(a)
  F9(b)
};

// expected-note@+1 {{in implicit copy constructor for 'TooMany' first required here}}
void f(TooMany &a) { TooMany b(a); }
