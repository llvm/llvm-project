// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.WebKitRefCntblBaseVirtualDtor -verify %s

struct RefCntblBase {
  void ref() {}
  void deref() {}
};

template<class T>
struct DerivedClassTmpl1 : T { };
// expected-warning@-1{{Struct 'RefCntblBase' is used as a base of struct 'DerivedClassTmpl1<RefCntblBase>' but doesn't have virtual destructor}}

DerivedClassTmpl1<RefCntblBase> a;



template<class T>
struct DerivedClassTmpl2 : T { };
// expected-warning@-1{{Struct 'RefCntblBase' is used as a base of struct 'DerivedClassTmpl2<RefCntblBase>' but doesn't have virtual destructor}}

template<class T> int foo(T) { DerivedClassTmpl2<T> f; return 42; }
int b = foo(RefCntblBase{});



template<class T>
struct DerivedClassTmpl3 : T { };
// expected-warning@-1{{Struct 'RefCntblBase' is used as a base of struct 'DerivedClassTmpl3<RefCntblBase>' but doesn't have virtual destructor}}

typedef DerivedClassTmpl3<RefCntblBase> Foo;
Foo c;
