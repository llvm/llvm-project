// RUN: %clang_cc1 %s -fsyntax-only -Wno-unused-value -Wmicrosoft -verify -fms-extensions
// expected-no-diagnostics

// Wide character predefined identifiers
#define _STR2WSTR(str) L##str
#define STR2WSTR(str) _STR2WSTR(str)
void abcdefghi12(void) {
 const wchar_t (*ss)[12] = &STR2WSTR(__FUNCTION__);
 static int arr[sizeof(STR2WSTR(__FUNCTION__))==12*sizeof(wchar_t) ? 1 : -1];
 const wchar_t (*ss2)[31] = &STR2WSTR(__FUNCSIG__);
 static int arr2[sizeof(STR2WSTR(__FUNCSIG__))==31*sizeof(wchar_t) ? 1 : -1];
}

namespace PR13206 {
void foo(const wchar_t *);

template<class T> class A {
public:
 void method() {
  foo(L__FUNCTION__);
 }
};

void bar() {
 A<int> x;
 x.method();
}
}
