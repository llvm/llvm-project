// RUN: %clang_cc1 -triple arm64-apple-ios -std=c++2b -Wno-string-plus-int -fptrauth-calls -fptrauth-intrinsics -verify -fsyntax-only %s

struct S {
  static constexpr auto options = "strip";
};

constexpr const char* const_options(int i) {
  const char* local_const = "isa-pointer";
  constexpr auto local_constexpr = ",";
  static constexpr auto static_const = "strip";
  static constexpr auto static_constexpr = "sign-and-strip";

  switch (i) {
    case 0:
      return "";
    case 1:
      return "authenticates-null-values";
    case 2:
      return local_const;
    case 3:
      return local_constexpr;
    case 4:
      return static_const;
    case 5:
      return static_constexpr;
    case 6:
      return "some characters";
    case 7:
      return S::options;
    case 8:
      return const_options(3)+1;
    default:
      #ifdef __EXCEPTIONS
      throw "invalid index";
      #else
      __builtin_trap();
      #endif
  }
}

void test_func() {
  int * __ptrauth(1,1,1,const_options(0)) zero;
  int * __ptrauth(1,1,1,const_options(1)) one;
  int * __ptrauth(1,1,1,const_options(2)) two;
  int * __ptrauth(1,1,1,const_options(3)) three;
  // expected-error@-1 {{'__ptrauth' options parameter contains an empty option}}
  // expected-error@-2 {{'__ptrauth' options parameter has a trailing comma}}
  // expected-note@-3 {{options parameter evaluated to ','}}
  int * __ptrauth(1,1,1,const_options(4)) four;
  int * __ptrauth(1,1,1,const_options(5)) five;
  int * __ptrauth(1,1,1,const_options(6)) six;
  // expected-error@-1 {{missing comma after 'some' option in '__ptrauth' qualifier}}
  int * __ptrauth(1,1,1,const_options(7)) seven;
  int * __ptrauth(1,1,1,const_options(8)) eight;
  int * __ptrauth(1,1,1,S::options) struct_access;
  int * __ptrauth(1,1,1,2 * 3) ice;
  // expected-error@-1 {{'__ptrauth' options parameter must be a string of comma separated flags}}
  int * __ptrauth(1,1,1,4 + "wat,strip") arithmetic_string;
  int * __ptrauth(1,1,1,5 + "wat,strip") arithmetic_string2;
  // expected-error@-1 {{unknown '__ptrauth' authentication option 'trip'}}

  // Handle evaluation failing
  int * __ptrauth(1,1,1,const_options(50)) fifty;
  // expected-error@-1 {{'__ptrauth' options parameter must be a string of comma separated flags}}
}
