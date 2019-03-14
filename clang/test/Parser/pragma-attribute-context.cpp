// RUN: %clang_cc1 -triple x86_64-apple-darwin9.0.0 -verify -std=c++11 %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9.0.0 -xobjective-c++ -verify -std=c++11 %s

#define BEGIN_PRAGMA _Pragma("clang attribute push (__attribute__((availability(macos, introduced=1000))), apply_to=function)")
#define END_PRAGMA _Pragma("clang attribute pop")

extern "C" {
BEGIN_PRAGMA
int f(); // expected-note{{'f' has been marked as being introduced in macOS 1000 here}}
END_PRAGMA
}

namespace my_ns {
BEGIN_PRAGMA
int g(); // expected-note{{'g' has been marked as being introduced in macOS 1000 here}}
END_PRAGMA
namespace nested {
BEGIN_PRAGMA
int h(); // expected-note{{'h' has been marked as being introduced in macOS 1000 here}}
END_PRAGMA
}
}

int a = f(); // expected-warning{{'f' is only available on macOS 1000 or newer}} expected-note{{annotate 'a'}}
int b = my_ns::g(); // expected-warning{{'g' is only available on macOS 1000 or newer}} expected-note{{annotate 'b'}}
int c = my_ns::nested::h(); // expected-warning{{'h' is only available on macOS 1000 or newer}} expected-note{{annotate 'c'}}

struct InStruct {
  // FIXME: This asserts in Objective-C++!
  // FIXME: This is a horrible diagnostic!
#ifndef __OBJC__
  BEGIN_PRAGMA // expected-error {{expected member name or ';' after declaration specifiers}}
#endif
};
