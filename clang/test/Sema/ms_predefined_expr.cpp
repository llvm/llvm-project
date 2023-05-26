// RUN: %clang_cc1 %s -fsyntax-only -Wmicrosoft -verify -fms-extensions

void f() {
 const char a[] = __FUNCTION__; // expected-warning{{initializing an array from a '__FUNCTION__' predefined identifier is a Microsoft extension}}
 const char b[] = __FUNCDNAME__; // expected-warning{{initializing an array from a '__FUNCDNAME__' predefined identifier is a Microsoft extension}}
 const char c[] = __FUNCSIG__; // expected-warning{{initializing an array from a '__FUNCSIG__' predefined identifier is a Microsoft extension}}
 const char d[] = __func__; // expected-warning{{initializing an array from a '__func__' predefined identifier is a Microsoft extension}}
 const char e[] = __PRETTY_FUNCTION__; // expected-warning{{initializing an array from a '__PRETTY_FUNCTION__' predefined identifier is a Microsoft extension}}
}
