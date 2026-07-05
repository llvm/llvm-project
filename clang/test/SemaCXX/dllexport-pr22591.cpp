// RUN: %clang_cc1 -triple i686-windows-gnu  -fms-extensions -verify -std=c++03 %s
// RUN: %clang_cc1 -triple i686-windows-gnu  -fms-extensions -verify -std=c++11 %s
// RUN: %clang_cc1 -triple i686-windows-msvc -fms-extensions -verify -std=c++03 -DERROR %s
// RUN: %clang_cc1 -triple i686-windows-msvc -fms-extensions -verify -std=c++11 %s

#ifndef ERROR
// expected-no-diagnostics
#endif

struct NonCopyable {
private:
#ifdef ERROR
  // expected-note@+2{{declared private here}}
#endif
  NonCopyable();
};

#ifdef ERROR
// expected-error@+4{{field of type 'NonCopyable' has private default constructor}}
// expected-note@+3{{implicit default constructor for 'S' first required here}}
// expected-note@+2{{due to 'S' being dllexported; try compiling in C++11 mode}}
#endif
struct __declspec(dllexport) S {
  NonCopyable member;
};
