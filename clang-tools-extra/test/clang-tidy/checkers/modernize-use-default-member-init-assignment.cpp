// RUN: %check_clang_tidy %s modernize-use-default-member-init %t -- \
// RUN: -config="{CheckOptions: [{key: modernize-use-default-member-init.UseAssignment, value: 1}]}"

struct S {
};

struct PositiveValueChar {
  PositiveValueChar() : c0(), c1()/*, c2(), c3()*/ {}
  // CHECK-FIXES: PositiveValueChar()  {}
  const char c0;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use default member initializer for 'c0' [modernize-use-default-member-init]
  // CHECK-FIXES: const char c0 = '\0';
  wchar_t c1;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use default member initializer for 'c1'
  // CHECK-FIXES: wchar_t c1 = L'\0';
  // FIXME: char16_t c2;
  // C HECK-MESSAGES: :[[@LINE-1]]:12: warning: use default member initializer for 'c2'
  // C HECK-FIXES: char16_t c2 = u'\0';
  // FIXME: char32_t c3;
  // C HECK-MESSAGES: :[[@LINE-1]]:12: warning: use default member initializer for 'c3'
  // C HECK-FIXES: char32_t c3 = U'\0';
};

struct PositiveChar {
  PositiveChar() : d('a') {}
  // CHECK-FIXES: PositiveChar()  {}
  char d;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use default member initializer for 'd'
  // CHECK-FIXES: char d = 'a';
};

struct PositiveValueInt {
  PositiveValueInt() : i() {}
  // CHECK-FIXES: PositiveValueInt()  {}
  const int i;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use default member initializer for 'i'
  // CHECK-FIXES: const int i = 0;
};

struct PositiveInt {
  PositiveInt() : j(1) {}
  // CHECK-FIXES: PositiveInt()  {}
  int j;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use default member initializer for 'j'
  // CHECK-FIXES: int j = 1;
};

struct PositiveUnaryMinusInt {
  PositiveUnaryMinusInt() : j(-1) {}
  // CHECK-FIXES: PositiveUnaryMinusInt()  {}
  int j;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use default member initializer for 'j'
  // CHECK-FIXES: int j = -1;
};

struct PositiveUnaryPlusInt {
  PositiveUnaryPlusInt() : j(+1) {}
  // CHECK-FIXES: PositiveUnaryPlusInt()  {}
  int j;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use default member initializer for 'j'
  // CHECK-FIXES: int j = +1;
};

struct PositiveValueComplexInt {
  PositiveValueComplexInt() : i() {}
  // CHECK-FIXES: PositiveValueComplexInt()  {}
  _Complex int i;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use default member initializer for 'i'
  // CHECK-FIXES: _Complex int i = 0;
};

struct PositiveValueFloat {
  PositiveValueFloat() : f() {}
  // CHECK-FIXES: PositiveValueFloat()  {}
  float f;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use default member initializer for 'f'
  // CHECK-FIXES: float f = 0.0f;
};

struct PositiveValueDouble {
  PositiveValueDouble() : d() {}
  // CHECK-FIXES: PositiveValueDouble()  {}
  double d;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use default member initializer for 'd'
  // CHECK-FIXES: double d = 0.0;
};

struct PositiveDouble {
  PositiveDouble() : f(2.5463e43) {}
  // CHECK-FIXES: PositiveDouble()  {}
  double f;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use default member initializer for 'f'
  // CHECK-FIXES: double f = 2.5463e43;
};

struct PositiveValueComplexFloat {
  PositiveValueComplexFloat() : f() {}
  // CHECK-FIXES: PositiveValueComplexFloat()  {}
  _Complex float f;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use default member initializer for 'f'
  // CHECK-FIXES: _Complex float f = 0.0f;
};

struct PositiveValueComplexDouble {
  PositiveValueComplexDouble() : f() {}
  // CHECK-FIXES: PositiveValueComplexDouble()  {}
  _Complex double f;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use default member initializer for 'f'
  // CHECK-FIXES: _Complex double f = 0.0;
};

struct PositiveUnaryMinusDouble {
  PositiveUnaryMinusDouble() : f(-2.5463e43) {}
  // CHECK-FIXES: PositiveUnaryMinusDouble()  {}
  double f;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use default member initializer for 'f'
  // CHECK-FIXES: double f = -2.5463e43;
};

struct PositiveUnaryPlusDouble {
  PositiveUnaryPlusDouble() : f(+2.5463e43) {}
  // CHECK-FIXES: PositiveUnaryPlusDouble()  {}
  double f;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use default member initializer for 'f'
  // CHECK-FIXES: double f = +2.5463e43;
};

struct PositiveValueBool {
  PositiveValueBool() : b() {}
  // CHECK-FIXES: PositiveValueBool()  {}
  bool b;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use default member initializer for 'b'
  // CHECK-FIXES: bool b = false;
};

struct PositiveBool {
  PositiveBool() : a(true) {}
  // CHECK-FIXES: PositiveBool()  {}
  bool a;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use default member initializer for 'a'
  // CHECK-FIXES: bool a = true;
};

struct PositiveValuePointer {
  PositiveValuePointer() : p() {}
  // CHECK-FIXES: PositiveValuePointer()  {}
  int *p;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use default member initializer for 'p'
  // CHECK-FIXES: int *p = nullptr;
};

struct PositiveNullPointer {
  PositiveNullPointer() : q(nullptr) {}
  // CHECK-FIXES: PositiveNullPointer()  {}
  int *q;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use default member initializer for 'q'
  // CHECK-FIXES: int *q = nullptr;
};

enum Enum { Foo };
struct PositiveEnum {
  PositiveEnum() : e(Foo) {}
  // CHECK-FIXES: PositiveEnum()  {}
  Enum e;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use default member initializer for 'e'
  // CHECK-FIXES: Enum e = Foo;
};

struct PositiveValueEnum {
  PositiveValueEnum() : e() {}
  // CHECK-FIXES: PositiveValueEnum()  {}
  Enum e;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use default member initializer for 'e'
  // CHECK-FIXES: Enum e{};
};

struct PositiveString {
  PositiveString() : s("foo") {}
  // CHECK-FIXES: PositiveString()  {}
  const char *s;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use default member initializer for 's'
  // CHECK-FIXES: const char *s = "foo";
};

template <typename T>
struct NegativeTemplate {
    NegativeTemplate() : t() {}
    T t;
};

NegativeTemplate<int> nti;
NegativeTemplate<double> ntd;
