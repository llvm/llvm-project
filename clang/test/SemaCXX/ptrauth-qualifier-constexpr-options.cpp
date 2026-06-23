// RUN: %clang_cc1 -triple arm64-apple-ios -std=c++2b -Wno-string-plus-int -fptrauth-calls -fptrauth-intrinsics -verify -fsyntax-only %s

struct S {
  static constexpr auto options = "strip";
};

struct string_view {
  int S;
  const char* D;
  constexpr string_view() : S(0), D(0){}
  constexpr string_view(const char* Str) : S(__builtin_strlen(Str)), D(Str) {}
  constexpr string_view(int Size, const char* Str) : S(Size), D(Str) {}
  constexpr int size() const {
      return S;
  }
  constexpr const char* data() const {
      return D;
  }
};
template <class StringType> constexpr const char* const_options(int i) {
  const char* local_const = "isa-pointer";
  constexpr auto local_constexpr = "strip,";
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
      return const_options<StringType>(3)+1;
    default:
      #ifdef __EXCEPTIONS
      throw "invalid index";
      #else
      __builtin_trap();
      #endif
  }
}

// When we support dependent pointer auth qualifiers this can become a template
// function rather than manually duplicating it.
void test_func_charptr() {
  using StringType = const char*;
  int * __ptrauth(1,1,1,const_options<StringType>(0)) zero;
  int * __ptrauth(1,1,1,const_options<StringType>(1)) one;
  int * __ptrauth(1,1,1,const_options<StringType>(2)) two;
  int * __ptrauth(1,1,1,const_options<StringType>(3)) three;
  // expected-error@-1 {{unexpected trailing comma in '__ptrauth' options argument}}
  // expected-note@-2 {{options parameter evaluated to 'strip,'}}
  int * __ptrauth(1,1,1,const_options<StringType>(4)) four;
  int * __ptrauth(1,1,1,const_options<StringType>(5)) five;
  int * __ptrauth(1,1,1,const_options<StringType>(6)) six;
  // expected-error@-1 {{expected a comma after 'some' in '__ptrauth' options argument}}
  // expected-note@-2 {{options parameter evaluated to 'some characters'}}
  int * __ptrauth(1,1,1,const_options<StringType>(7)) seven;
  int * __ptrauth(1,1,1,const_options<StringType>(8)) eight;
  // expected-error@-1 {{unknown '__ptrauth' authentication option 'trip'}}
  // expected-note@-2 {{options parameter evaluated to 'trip,'}}
  int * __ptrauth(1,1,1,2 * 3) ice;
  // expected-error@-1 {{the expression in '__ptrauth' options must be a string literal or an object with 'data()' and 'size()' member functions}}
  int * __ptrauth(1,1,1,4 + "wat,strip") arithmetic_string;
  int * __ptrauth(1,1,1,5 + "wat,strip") arithmetic_string2;
  // expected-error@-1 {{unknown '__ptrauth' authentication option 'trip'}}
  // expected-note@-2 {{options parameter evaluated to 'trip'}}

  // Handle evaluation failing
  int * __ptrauth(1,1,1,const_options<StringType>(50)) fifty;
  // expected-error@-1 {{the expression in '__ptrauth' options must be a string literal or an object with 'data()' and 'size()' member functions}}
}

void test_func_string_view() {
  using StringType = string_view;
  int * __ptrauth(1,1,1,const_options<StringType>(0)) zero;
  int * __ptrauth(1,1,1,const_options<StringType>(1)) one;
  int * __ptrauth(1,1,1,const_options<StringType>(2)) two;
  int * __ptrauth(1,1,1,const_options<StringType>(3)) three;
  // expected-error@-1 {{unexpected trailing comma in '__ptrauth' options argument}}
  // expected-note@-2 {{options parameter evaluated to 'strip,'}}
  int * __ptrauth(1,1,1,const_options<StringType>(4)) four;
  int * __ptrauth(1,1,1,const_options<StringType>(5)) five;
  int * __ptrauth(1,1,1,const_options<StringType>(6)) six;
  // expected-error@-1 {{expected a comma after 'some' in '__ptrauth' options argument}}
  // expected-note@-2 {{options parameter evaluated to 'some characters'}}
  int * __ptrauth(1,1,1,const_options<StringType>(7)) seven;
  int * __ptrauth(1,1,1,const_options<StringType>(8)) eight;
  // expected-error@-1 {{unknown '__ptrauth' authentication option 'trip'}}
  // expected-note@-2 {{options parameter evaluated to 'trip,'}}
  int * __ptrauth(1,1,1,2 * 3) ice;
  // expected-error@-1 {{the expression in '__ptrauth' options must be a string literal or an object with 'data()' and 'size()' member functions}}
  int * __ptrauth(1,1,1,4 + "wat,strip") arithmetic_string;
  int * __ptrauth(1,1,1,5 + "wat,strip") arithmetic_string2;
  // expected-error@-1 {{unknown '__ptrauth' authentication option 'trip'}}
  // expected-note@-2 {{options parameter evaluated to 'trip'}}

  // Handle evaluation failing
  int * __ptrauth(1,1,1,const_options<StringType>(50)) fifty;
  // expected-error@-1 {{the expression in '__ptrauth' options must be a string literal or an object with 'data()' and 'size()' member functions}}
}
