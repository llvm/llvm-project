// RUN: %check_clang_tidy %s modernize-use-default-member-init %t -- -- -std=c++11

struct S {
};

struct PositiveValueChar {
  PositiveValueChar() : c0(), c1()/*, c2(), c3()*/ {}
  // CHECK-FIXES: PositiveValueChar()  {}
  const char c0;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use default member initializer for 'c0' [modernize-use-default-member-init]
  // CHECK-FIXES: const char c0{};
  wchar_t c1;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use default member initializer for 'c1'
  // CHECK-FIXES: wchar_t c1{};
  // FIXME: char16_t c2;
  // C HECK-MESSAGES: :[[@LINE-1]]:12: warning: use default member initializer for 'c2'
  // C HECK-FIXES: char16_t c2{};
  // FIXME: char32_t c3;
  // C HECK-MESSAGES: :[[@LINE-1]]:12: warning: use default member initializer for 'c3'
  // C HECK-FIXES: char32_t c3{};
};

struct PositiveChar {
  PositiveChar() : d('a') {}
  // CHECK-FIXES: PositiveChar()  {}
  char d;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use default member initializer for 'd'
  // CHECK-FIXES: char d{'a'};
};

struct PositiveValueInt {
  PositiveValueInt() : i() {}
  // CHECK-FIXES: PositiveValueInt()  {}
  const int i;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use default member initializer for 'i'
  // CHECK-FIXES: const int i{};
};

struct PositiveInt {
  PositiveInt() : j(1) {}
  // CHECK-FIXES: PositiveInt()  {}
  int j;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use default member initializer for 'j'
  // CHECK-FIXES: int j{1};
};

struct PositiveUnaryMinusInt {
  PositiveUnaryMinusInt() : j(-1) {}
  // CHECK-FIXES: PositiveUnaryMinusInt()  {}
  int j;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use default member initializer for 'j'
  // CHECK-FIXES: int j{-1};
};

struct PositiveUnaryPlusInt {
  PositiveUnaryPlusInt() : j(+1) {}
  // CHECK-FIXES: PositiveUnaryPlusInt()  {}
  int j;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use default member initializer for 'j'
  // CHECK-FIXES: int j{+1};
};

struct PositiveValueComplexInt {
  PositiveValueComplexInt() : i() {}
  // CHECK-FIXES: PositiveValueComplexInt()  {}
  _Complex int i;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use default member initializer for 'i'
  // CHECK-FIXES: _Complex int i{};
};

struct PositiveValueFloat {
  PositiveValueFloat() : f() {}
  // CHECK-FIXES: PositiveValueFloat()  {}
  float f;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use default member initializer for 'f'
  // CHECK-FIXES: float f{};
};

struct PositiveValueDouble {
  PositiveValueDouble() : d() {}
  // CHECK-FIXES: PositiveValueDouble()  {}
  double d;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use default member initializer for 'd'
  // CHECK-FIXES: double d{};
};

struct PositiveDouble {
  PositiveDouble() : f(2.5463e43) {}
  // CHECK-FIXES: PositiveDouble()  {}
  double f;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use default member initializer for 'f'
  // CHECK-FIXES: double f{2.5463e43};
};

struct PositiveValueComplexFloat {
  PositiveValueComplexFloat() : f() {}
  // CHECK-FIXES: PositiveValueComplexFloat()  {}
  _Complex float f;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use default member initializer for 'f'
  // CHECK-FIXES: _Complex float f{};
};

struct PositiveValueComplexDouble {
  PositiveValueComplexDouble() : f() {}
  // CHECK-FIXES: PositiveValueComplexDouble()  {}
  _Complex double f;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use default member initializer for 'f'
  // CHECK-FIXES: _Complex double f{};
};

struct PositiveUnaryMinusDouble {
  PositiveUnaryMinusDouble() : f(-2.5463e43) {}
  // CHECK-FIXES: PositiveUnaryMinusDouble()  {}
  double f;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use default member initializer for 'f'
  // CHECK-FIXES: double f{-2.5463e43};
};

struct PositiveUnaryPlusDouble {
  PositiveUnaryPlusDouble() : f(+2.5463e43) {}
  // CHECK-FIXES: PositiveUnaryPlusDouble()  {}
  double f;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use default member initializer for 'f'
  // CHECK-FIXES: double f{+2.5463e43};
};

struct PositiveValueBool {
  PositiveValueBool() : b() {}
  // CHECK-FIXES: PositiveValueBool()  {}
  bool b;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use default member initializer for 'b'
  // CHECK-FIXES: bool b{};
};

struct PositiveBool {
  PositiveBool() : a(true) {}
  // CHECK-FIXES: PositiveBool()  {}
  bool a;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use default member initializer for 'a'
  // CHECK-FIXES: bool a{true};
};

struct PositiveValuePointer {
  PositiveValuePointer() : p() {}
  // CHECK-FIXES: PositiveValuePointer()  {}
  int *p;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use default member initializer for 'p'
  // CHECK-FIXES: int *p{};
};

struct PositiveNullPointer {
  PositiveNullPointer() : q(nullptr) {}
  // CHECK-FIXES: PositiveNullPointer()  {}
  int *q;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use default member initializer for 'q'
  // CHECK-FIXES: int *q{nullptr};
};

enum Enum { Foo, Bar };
struct PositiveEnum {
  PositiveEnum() : e(Foo) {}
  // CHECK-FIXES: PositiveEnum()  {}
  Enum e;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use default member initializer for 'e'
  // CHECK-FIXES: Enum e{Foo};
};

struct PositiveString {
  PositiveString() : s("foo") {}
  // CHECK-FIXES: PositiveString()  {}
  const char *s;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use default member initializer for 's'
  // CHECK-FIXES: const char *s{"foo"};
};

struct PositiveStruct {
  PositiveStruct() : s(7) {}
  // CHECK-FIXES: PositiveStruct()  {}
  struct {
    int s;
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use default member initializer for 's'
    // CHECK-FIXES: int s{7};
  };
};

template <typename T>
struct NegativeTemplate {
    NegativeTemplate() : t() {}
    T t;
};

NegativeTemplate<int> nti;
NegativeTemplate<double> ntd;

struct NegativeDefaultMember {
  NegativeDefaultMember() {}
  int i = 2;
};

struct NegativeClass : S {
  NegativeClass() : s() {}
  S s;
};

struct NegativeBase : S {
  NegativeBase() : S() {}
};

struct NegativeDefaultOtherMember{
  NegativeDefaultOtherMember() : i(3) {}
  int i = 4;
};

struct NegativeUnion {
  NegativeUnion() : d(5.0) {}
  union {
    int i;
    double d;
  };
};

struct NegativeBitField
{
  NegativeBitField() : i(6) {}
  int i : 5;
};

struct NegativeNotDefaultInt
{
  NegativeNotDefaultInt(int) : i(7) {}
  int i;
};

struct NegativeDefaultArg
{
  NegativeDefaultArg(int i = 4) : i(i) {}
  int i;
};

struct ExistingChar {
  ExistingChar(short) : e1(), e2(), e3(), e4() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: member initializer for 'e1' is redundant [modernize-use-default-member-init]
  // CHECK-MESSAGES: :[[@LINE-2]]:31: warning: member initializer for 'e2' is redundant
  // CHECK-MESSAGES: :[[@LINE-3]]:37: warning: member initializer for 'e3' is redundant
  // CHECK-FIXES: ExistingChar(short) :  e4() {}
  ExistingChar(int) : e1(0), e2(0), e3(0), e4(0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: member initializer for 'e1' is redundant
  // CHECK-MESSAGES: :[[@LINE-2]]:30: warning: member initializer for 'e2' is redundant
  // CHECK-MESSAGES: :[[@LINE-3]]:37: warning: member initializer for 'e3' is redundant
  // CHECK-FIXES: ExistingChar(int) :  e4(0) {}
  ExistingChar(long) : e1('\0'), e2('\0'), e3('\0'), e4('\0') {}
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: member initializer for 'e1' is redundant
  // CHECK-MESSAGES: :[[@LINE-2]]:34: warning: member initializer for 'e2' is redundant
  // CHECK-MESSAGES: :[[@LINE-3]]:44: warning: member initializer for 'e3' is redundant
  // CHECK-FIXES: ExistingChar(long) :  e4('\0') {}
  ExistingChar(char) : e1('a'), e2('a'), e3('a'), e4('a') {}
  // CHECK-MESSAGES: :[[@LINE-1]]:51: warning: member initializer for 'e4' is redundant
  // CHECK-FIXES: ExistingChar(char) : e1('a'), e2('a'), e3('a') {}
  char e1{};
  char e2 = 0;
  char e3 = '\0';
  char e4 = 'a';
};

struct ExistingInt {
  ExistingInt(short) : e1(), e2(), e3(), e4(), e5(), e6() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: member initializer for 'e1' is redundant [modernize-use-default-member-init]
  // CHECK-MESSAGES: :[[@LINE-2]]:30: warning: member initializer for 'e2' is redundant
  // CHECK-FIXES: ExistingInt(short) :  e3(), e4(), e5(), e6() {}
  ExistingInt(int) : e1(0), e2(0), e3(0), e4(0), e5(0), e6(0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: member initializer for 'e1' is redundant
  // CHECK-MESSAGES: :[[@LINE-2]]:29: warning: member initializer for 'e2' is redundant
  // CHECK-FIXES: ExistingInt(int) :  e3(0), e4(0), e5(0), e6(0) {}
  ExistingInt(long) : e1(5), e2(5), e3(5), e4(5), e5(5), e6(5) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: member initializer for 'e3' is redundant
  // CHECK-MESSAGES: :[[@LINE-2]]:44: warning: member initializer for 'e4' is redundant
  // CHECK-MESSAGES: :[[@LINE-3]]:58: warning: member initializer for 'e6' is redundant
  // CHECK-FIXES: ExistingInt(long) : e1(5), e2(5),  e5(5) {}
  ExistingInt(char) : e1(-5), e2(-5), e3(-5), e4(-5), e5(-5), e6(-5) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:55: warning: member initializer for 'e5' is redundant
  // CHECK-FIXES: ExistingInt(char) : e1(-5), e2(-5), e3(-5), e4(-5),  e6(-5) {}
  int e1{};
  int e2 = 0;
  int e3 = {5};
  int e4 = 5;
  int e5 = -5;
  int e6 = +5;
};

struct ExistingDouble {
  ExistingDouble(short) : e1(), e2(), e3(), e4(), e5() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: member initializer for 'e1' is redundant
  // CHECK-MESSAGES: :[[@LINE-2]]:33: warning: member initializer for 'e2' is redundant
  // CHECK-FIXES: ExistingDouble(short) :  e3(), e4(), e5() {}
  ExistingDouble(int) : e1(0.0), e2(0.0), e3(0.0), e4(0.0), e5(0.0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: member initializer for 'e1' is redundant
  // CHECK-MESSAGES: :[[@LINE-2]]:34: warning: member initializer for 'e2' is redundant
  // CHECK-FIXES: ExistingDouble(int) :  e3(0.0), e4(0.0), e5(0.0) {}
  ExistingDouble(long) : e1(5.0), e2(5.0), e3(5.0), e4(5.0), e5(5.0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: member initializer for 'e3' is redundant
  // CHECK-MESSAGES: :[[@LINE-2]]:62: warning: member initializer for 'e5' is redundant
  // CHECK-FIXES: ExistingDouble(long) : e1(5.0), e2(5.0),  e4(5.0) {}
  ExistingDouble(char) : e1(-5.0), e2(-5.0), e3(-5.0), e4(-5.0), e5(-5.0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:56: warning: member initializer for 'e4' is redundant
  // CHECK-FIXES: ExistingDouble(char) : e1(-5.0), e2(-5.0), e3(-5.0),  e5(-5.0) {}
  double e1{};
  double e2 = 0.0;
  double e3 = 5.0;
  double e4 = -5.0;
  double e5 = +5.0;
};

struct ExistingBool {
  ExistingBool(short) : e1(), e2(), e3() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: member initializer for 'e1' is redundant
  // CHECK-MESSAGES: :[[@LINE-2]]:31: warning: member initializer for 'e2' is redundant
  // CHECK-FIXES: ExistingBool(short) :  e3() {}
  ExistingBool(int) : e1(false), e2(false), e3(false) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: member initializer for 'e1' is redundant
  // CHECK-MESSAGES: :[[@LINE-2]]:34: warning: member initializer for 'e2' is redundant
  // CHECK-FIXES: ExistingBool(int) :  e3(false) {}
  ExistingBool(long) : e1(true), e2(true), e3(true) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: member initializer for 'e3' is redundant
  // CHECK-FIXES: ExistingBool(long) : e1(true), e2(true) {}
  bool e1{};
  bool e2 = false;
  bool e3 = true;
};

struct ExistingEnum {
  ExistingEnum(short) : e1(Foo), e2(Foo) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: member initializer for 'e1' is redundant
  // CHECK-FIXES: ExistingEnum(short) :  e2(Foo) {}
  ExistingEnum(int) : e1(Bar), e2(Bar) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: member initializer for 'e2' is redundant
  // CHECK-FIXES: ExistingEnum(int) : e1(Bar) {}
  Enum e1 = Foo;
  Enum e2{Bar};
};

struct ExistingPointer {
  ExistingPointer(short) : e1(), e2(), e3(), e4() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: member initializer for 'e1' is redundant
  // CHECK-MESSAGES: :[[@LINE-2]]:34: warning: member initializer for 'e2' is redundant
  // CHECK-MESSAGES: :[[@LINE-3]]:40: warning: member initializer for 'e3' is redundant
  // CHECK-FIXES: ExistingPointer(short) :  e4() {}
  ExistingPointer(int) : e1(0), e2(0), e3(0), e4(&e1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: member initializer for 'e1' is redundant
  // CHECK-MESSAGES: :[[@LINE-2]]:33: warning: member initializer for 'e2' is redundant
  // CHECK-MESSAGES: :[[@LINE-3]]:40: warning: member initializer for 'e3' is redundant
  // CHECK-FIXES: ExistingPointer(int) :  e4(&e1) {}
  ExistingPointer(long) : e1(nullptr), e2(nullptr), e3(nullptr), e4(&e2) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: member initializer for 'e1' is redundant
  // CHECK-MESSAGES: :[[@LINE-2]]:40: warning: member initializer for 'e2' is redundant
  // CHECK-MESSAGES: :[[@LINE-3]]:53: warning: member initializer for 'e3' is redundant
  // CHECK-FIXES: ExistingPointer(long) :  e4(&e2) {}
  int *e1{};
  int *e2 = 0;
  int *e3 = nullptr;
  int **e4 = &e1;
};

struct ExistingString {
  ExistingString(short) : e1(), e2(), e3(), e4() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: member initializer for 'e1' is redundant [modernize-use-default-member-init]
  // CHECK-MESSAGES: :[[@LINE-2]]:33: warning: member initializer for 'e2' is redundant
  // CHECK-FIXES: ExistingString(short) :  e3(), e4() {}
  ExistingString(int) : e1(0), e2(0), e3(0), e4(0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: member initializer for 'e1' is redundant
  // CHECK-MESSAGES: :[[@LINE-2]]:32: warning: member initializer for 'e2' is redundant
  // CHECK-FIXES: ExistingString(int) :  e3(0), e4(0) {}
  ExistingString(long) : e1(nullptr), e2(nullptr), e3(nullptr), e4(nullptr) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: member initializer for 'e1' is redundant
  // CHECK-MESSAGES: :[[@LINE-2]]:39: warning: member initializer for 'e2' is redundant
  // CHECK-FIXES: ExistingString(long) :  e3(nullptr), e4(nullptr) {}
  ExistingString(char) : e1("foo"), e2("foo"), e3("foo"), e4("foo") {}
  // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: member initializer for 'e3' is redundant
  // CHECK-FIXES: ExistingString(char) : e1("foo"), e2("foo"),  e4("foo") {}
  const char *e1{};
  const char *e2 = nullptr;
  const char *e3 = "foo";
  const char *e4 = "bar";
};

template <typename T>
struct NegativeTemplateExisting {
  NegativeTemplateExisting(int) : t(0) {}
  T t{};
};

NegativeTemplateExisting<int> ntei(0);
NegativeTemplateExisting<double> nted(0);

// This resulted in a warning by default.
#define MACRO() \
  struct MacroS { \
    void *P; \
    MacroS() : P(nullptr) {} \
  };

MACRO();
