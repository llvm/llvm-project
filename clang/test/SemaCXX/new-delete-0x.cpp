// RUN: %clang_cc1 -fsyntax-only -verify %s -triple=i686-pc-linux-gnu -std=c++11

using size_t = decltype(sizeof(0));
struct noreturn_t {} constexpr noreturn = {};

void *operator new [[noreturn]] (size_t, noreturn_t);
void operator delete [[noreturn]] (void*, noreturn_t);

void good_news()
{
  auto p = new int[2][[]];
  auto q = new int[[]][2];
  auto r = new int*[[]][2][[]];
  auto s = new (int(*[[]])[2][[]]);
}

void bad_news(int *ip)
{
  // attribute-specifiers can go almost anywhere in a new-type-id...
  auto r = new int[[]{return 1;}()][2]; // expected-error {{expected ']'}}
  auto s = new int*[[]{return 1;}()][2]; // expected-error {{expected ']'}}
  // ... but not here:
  auto t = new (int(*)[[]]); // expected-error {{an attribute list cannot appear here}}
  auto u = new (int(*)[[]{return 1;}()][2]); // expected-error {{C++11 only allows consecutive left square brackets when introducing an attribute}} expected-error {{variably modified type}}
}

void good_deletes()
{
  delete [&]{ return (int*)0; }();
}

void bad_deletes()
{
  // 'delete []' is always array delete, per [expr.delete]p1.
  // FIXME: Give a better diagnostic.
  delete []{ return (int*)0; }(); // expected-error {{expected expression}}
}
