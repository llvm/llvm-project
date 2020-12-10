// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c++11

// PR11179
template <short T> class Type1 {};
template <short T> void Function1(Type1<T>& x) {} // expected-note{{candidate function [with T = -42] not viable: expects an lvalue for 1st argument}}

template <unsigned short T> class Type2 {};
template <unsigned short T> void Function2(Type2<T>& x) {} // expected-note{{candidate function [with T = 42] not viable: expects an lvalue for 1st argument}}

enum class boolTy : bool {
  b = 0,
};

template <boolTy T> struct Type3Helper;
template <> struct Type3Helper<boolTy::b> { typedef boolTy Ty; };
template <boolTy T, typename Type3Helper<T>::Ty U> struct Type3 {};

// PR14386
enum class charTy : char {
  c = 0,
};

template <charTy T> struct Type4Helper;
template <> struct Type4Helper<charTy::c> { typedef charTy Ty; };
template <charTy T, typename Type4Helper<T>::Ty U> struct Type4 {};

enum class scharTy : signed char {
  c = 0,
};

template <scharTy T> struct Type5Helper;
template <> struct Type5Helper<scharTy::c> { typedef scharTy Ty; };
template <scharTy T, typename Type5Helper<T>::Ty U> struct Type5 {};

enum class ucharTy : unsigned char {
  c = 0,
};

template <ucharTy T> struct Type6Helper;
template <> struct Type6Helper<ucharTy::c> { typedef ucharTy Ty; };
template <ucharTy T, typename Type6Helper<T>::Ty U> struct Type6 {};

enum class wcharTy : wchar_t {
  c = 0,
};

template <wcharTy T> struct Type7Helper;
template <> struct Type7Helper<wcharTy::c> { typedef wcharTy Ty; };
template <wcharTy T, typename Type7Helper<T>::Ty U> struct Type7 {};

enum class char16Ty : char16_t {
  c = 0,
};

template <char16Ty T> struct Type8Helper;
template <> struct Type8Helper<char16Ty::c> { typedef char16Ty Ty; };
template <char16Ty T, typename Type8Helper<T>::Ty U> struct Type8 {};

enum class char32Ty : char16_t {
  c = 0,
};

template <char32Ty T> struct Type9Helper;
template <> struct Type9Helper<char32Ty::c> { typedef char32Ty Ty; };
template <char32Ty T, typename Type9Helper<T>::Ty U> struct Type9 {};

void Function() {
  Function1(Type1<-42>()); // expected-error{{no matching function for call to 'Function1'}}
  Function2(Type2<42>()); // expected-error{{no matching function for call to 'Function2'}}

  struct Type3<boolTy::b, "3"> t3; // expected-error{{value of type 'const char [2]' is not implicitly convertible to 'typename Type3Helper<(boolTy)false>::Ty' (aka 'boolTy')}}

  struct Type4<charTy::c, "4"> t4; // expected-error{{value of type 'const char [2]' is not implicitly convertible to 'typename Type4Helper<(charTy)'\x00'>::Ty' (aka 'charTy')}}
  struct Type5<scharTy::c, "5"> t5; // expected-error{{value of type 'const char [2]' is not implicitly convertible to 'typename Type5Helper<(scharTy)'\x00'>::Ty' (aka 'scharTy')}}
  struct Type6<ucharTy::c, "6"> t6; // expected-error{{value of type 'const char [2]' is not implicitly convertible to 'typename Type6Helper<(ucharTy)'\x00'>::Ty' (aka 'ucharTy')}}
  struct Type7<wcharTy::c, "7"> t7; // expected-error{{value of type 'const char [2]' is not implicitly convertible to 'typename Type7Helper<(wcharTy)L'\x00'>::Ty' (aka 'wcharTy')}}
  struct Type8<char16Ty::c, "8"> t8; // expected-error{{value of type 'const char [2]' is not implicitly convertible to 'typename Type8Helper<(char16Ty)u'\x00'>::Ty' (aka 'char16Ty')}}
  struct Type9<char32Ty::c, "9"> t9; // expected-error{{value of type 'const char [2]' is not implicitly convertible to 'typename Type9Helper<(char32Ty)u'\x00'>::Ty' (aka 'char32Ty')}}
}
