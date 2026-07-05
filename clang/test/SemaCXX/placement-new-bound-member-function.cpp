// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

struct S {
  int NoArgs();
  int OneArg(int);

  template <typename T>
  T TemplNoArgs(); // expected-note {{possible target for call}} // expected-note {{possible target for call}}

  template <typename T>
  T TemplOneArg(T); // expected-note {{possible target for call}} // expected-note {{possible target for call}}

  void* operator new(__SIZE_TYPE__, int);
};

S* GetS();

int test() {
  S s, *ps = GetS();
  int (S::*pNoArgs)() = &S::NoArgs;
  int (S::*pOneArg)(int) = &S::OneArg;
  int (S::*pTemplNoArgs)() = &S::TemplNoArgs<int>;
  int (S::*pTemplOneArg)(int) = &S::TemplOneArg<int>;

  new (s.NoArgs()) S;
  new (s.OneArg(1)) S;
  new (ps->NoArgs()) S;
  new (ps->OneArg(1)) S;
  new ((s.*pNoArgs)()) S;
  new ((s.*pOneArg)(1)) S;
  new ((ps->*pNoArgs)()) S;
  new ((ps->*pOneArg)(1)) S;

  new (s.TemplNoArgs<int>()) S;
  new (s.TemplOneArg<int>(1)) S;
  new (ps->TemplNoArgs<int>()) S;
  new (ps->TemplOneArg<int>(1)) S;
  new ((s.*pTemplNoArgs)()) S;
  new ((s.*pTemplOneArg)(1)) S;
  new ((ps->*pTemplNoArgs)()) S;
  new ((ps->*pTemplOneArg)(1)) S;

  new (s.NoArgs) S; // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
  new (s.OneArg) S; // expected-error {{reference to non-static member function must be called}}
  new (ps->NoArgs) S; // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
  new (ps->OneArg) S; // expected-error {{reference to non-static member function must be called}}
  new (s.*pNoArgs) S; // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
  new (s.*pOneArg) S; // expected-error {{reference to non-static member function must be called}}
  new (ps->*pNoArgs) S; // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
  new (ps->*pOneArg) S; // expected-error {{reference to non-static member function must be called}}
  new ((s.*pNoArgs)) S; // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
  new ((s.*pOneArg)) S; // expected-error {{reference to non-static member function must be called}}
  new ((ps->*pNoArgs)) S; // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
  new ((ps->*pOneArg)) S; // expected-error {{reference to non-static member function must be called}}

  new (s.TemplNoArgs<int>) S; // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
  new (s.TemplOneArg<int>) S; // expected-error {{reference to non-static member function must be called}}
  new (ps->TemplNoArgs<int>) S; // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
  new (ps->TemplOneArg<int>) S; // expected-error {{reference to non-static member function must be called}}
  new (s.*pTemplNoArgs) S; // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
  new (s.*pTemplOneArg) S; // expected-error {{reference to non-static member function must be called}}
  new (ps->*pTemplNoArgs) S; // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
  new (ps->*pTemplOneArg) S; // expected-error {{reference to non-static member function must be called}}
  new ((s.*pTemplNoArgs)) S; // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
  new ((s.*pTemplOneArg)) S; // expected-error {{reference to non-static member function must be called}}
  new ((ps->*pTemplNoArgs)) S; // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
  new ((ps->*pTemplOneArg)) S; // expected-error {{reference to non-static member function must be called}}
}
