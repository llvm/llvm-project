// RUN: %check_clang_tidy \
// RUN: -std=c++17-or-later %s modernize-use-string-view %t -- \
// RUN: -- -isystem %clang_tidy_headers

#include <string>

namespace std {
template <class T, class U> struct is_same { static constexpr bool value = false; };
template <class T> struct is_same<T, T> { static constexpr bool value = true; };
template <class T, class U> constexpr bool is_same_v = is_same<T, U>::value;
} // namespace std


// ==========================================================
// Positive tests
// ==========================================================

std::string simpleLiteral() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view simpleLiteral() {
  return "simpleLiteral";
}

std::string simpleLiteralConcat() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view simpleLiteralConcat() {
  return "hello" " " "world";
}

std::wstring simpleLiteralW() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::wstring_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::wstring_view simpleLiteralW() {
  return L"wide literal";
}

std::string simpleRLiteral() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view simpleRLiteral() {
  return R"(simpleLiteral)";
}

[[nodiscard]] std::string Attributed() {
// CHECK-MESSAGES:[[@LINE-1]]:15: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: {{\[\[nodiscard\]\]}} std::string_view Attributed() {
  return "attributed";
}

const std::string Const() {
// CHECK-MESSAGES:[[@LINE-1]]:7: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: const std::string_view Const() {
  return "Const";
}

auto Trailing() -> std::string {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// TODO: support fixes for trailing types
  return "Trailing";
}

std::string initList() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view initList() {
  return {"list"};
}

std::string ctorReturn(int i) {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view ctorReturn(int i) {
  if( i == 42) {
    do {
      return std::string();
// CHECK-FIXES: return {};
    } while(true);
  }
  else {
    for(int k = 0; k < 1000; k++)
      if ( i > k ) {
        return std::string();
// CHECK-FIXES: return {};
      }
      else {
        return "else";
      }
  }
  return std::string{};
// CHECK-FIXES: return {};
}

std::string ctorWithInitListReturn() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view ctorWithInitListReturn() {
// CHECK-FIXES: return {};
  return std::string{};
}

std::string emptyReturn() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view emptyReturn() {
  return {};
}

std::string switchCaseTest(int i) {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view switchCaseTest(int i) {
  switch (i) {
  case 1:
    return "case1";
  case 2:
    return "case2";
  case 3:
    return {};
  default:
    return "default";
  }
}

std::string switchCaseTestWithOnlyEmptyStrings(int i) {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view switchCaseTestWithOnlyEmptyStrings(int i) {
  switch (i) {
  case 1:
// CHECK-FIXES: return {};
    return std::string();
  case 2:
// CHECK-FIXES: return {};
    return std::string{};
  default:
    return {};
  }
}

std::string ifElseTest(bool flag) {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view ifElseTest(bool flag) {
  if (flag)
    return "true";
  return "false";
}

std::string ternary(bool flag) {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view ternary(bool flag) {
  return flag ? "true" : "false";
}

std::string nested(int x) {
  // TODO: support for nested ternary
  return x < 0 ? "neg" : (x == 0 ? "zero" : "pos");
}

class A {
  std::string classMethodInt() { return "internal"; }
// CHECK-MESSAGES:[[@LINE-1]]:3: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view classMethodInt() { return "internal"; }

  std::string classMethodExt();
// CHECK-FIXES: std::string_view classMethodExt();
};

std::string A::classMethodExt() { return "external"; }
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view A::classMethodExt() { return "external"; }

#define MACRO "MACRO LITERAL"
std::string macro() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view macro() {
  return MACRO;
}

#define my_string std::string
my_string macro_type() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view macro_type() {
  return "MACTO LITERAL";
}

#define my_definition std::string function_inside_macro()
my_definition {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
  return "literal";
}

std::string multiDecl();
std::string multiDecl();
std::string multiDecl() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view multiDecl() {
  return "multiDecl";
}

using MyTrivialString = std::string;
MyTrivialString aliasedTrivial() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view aliasedTrivial() {
  return "aliased";
}

template <typename T>
using MyString = std::basic_string<T>;

MyString<char> aliasedChar() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view aliasedChar() {
  return "aliasedChar";
}

MyString<wchar_t> aliasedWChar() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::wstring_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::wstring_view aliasedWChar() {
  return L"aliasedWChar";
}

// ==========================================================
// Negative tests
// ==========================================================

std::string toString() {
  return "ignored by default options";
}

std::string ToString() {
  return "ignored by default options";
}

std::string to_string() {
  return "ignored by default options";
}

std::string localVariable() {
  std::string s = "local variable";
  // TODO: extract and return literal
  return s;
}

std::string dynamicCalculation() {
  std::string s1 = "hello ";
  return s1 + "world";
}

std::string mixedReturns(bool flag) {
  if (flag) {
    return "safe static literal";
  }
  std::string s = "unsafe dynamic";
  return s;
}

std::string stringTernary(bool flag) {
  // TODO: extract literals
  return flag ? std::string("true") : std::string("false");
}

std::string_view alreadyGood() {
  return "alreadyGood";
}

std::wstring_view alreadyGoodW() {
  return L"alreadyGood";
}

std::string simpleLiteralS() {
  // TODO: replace ""s to literal and return string_view
  using namespace std::literals::string_literals;
  return "simpleLiteral"s;
}

std::string_view alreadyGoodSV() {
  using namespace std::literals::string_view_literals;
  return "alreadyGood"sv;
}

std::string returnArgCopy(std::string s) {
  // Must not be converted to string_view because of use-after-free on stack
  return s;
}

std::string returnIndirection(const char* ptr) {
  // Can be unsafe or intentional, like converting string_view into string
  return ptr;
}

std::string localBuffer() {
  char buf[] = "local buffer";
  // Must not be converted to string_view because of use-after-free on stack
  return buf;
}

std::string returnConstVar() {
  // TODO: seems safe
  constexpr auto kVar = "const string";
  return kVar;
}

std::string passStringView(std::string_view sv) {
  // TODO: Can be unsafe or intentional, like converting string_view into string
  return std::string(sv);
}

std::string explicitConstruction() {
  // Cannot be std::string_view: returning address of local temporary object
  // TODO: extract and return literal
  return std::string("explicitConstruction");
}

std::string explicitConstructionWithInitList() {
  // Cannot be std::string_view: returning address of local temporary object
  // TODO: extract and return literal
  return std::string{"explicitConstruction"};
}

std::string explicitConstructionEmpty() {
  return std::string("");
}

std::string explicitConstructionWithInitListEmpty() {
  return std::string{""};
}

std::string switchCaseTestWithExplicitEmptyString(int i) {
  switch (i) {
  case 1:
    return "case1";
  case 2:
    return "case2";
  case 3:
    return {};
  default:
    return std::string("");
  }
}

std::string switchCaseTestWithExplicitNonEmptyString(int i) {
  switch (i) {
  case 1:
    return "case1";
  case 2:
    return "case2";
  case 3:
    return {};
  default:
    return std::string("default");
  }
}

struct B {
  virtual ~B() = default;
  virtual std::string virtMethod1() { return "B::virtual1"; }
  virtual std::string virtMethod2();
};

 std::string B::virtMethod2() { return "B::virtual2"; }

struct C: public B {
  std::string virtMethod1() override { return "C::virtual"; }
  std::string virtMethod2() override;
};

std::string C::virtMethod2() { return "C::virtual"; }

std::string lambda() {
  // TODO: extract and return literal from lambda
  return []() {
    return "lambda";
  }();
}

struct TemplateString {
  static constexpr char* val = "TEMPLATE";
  template<typename T>
  // TODO: extract and return literal
  std::string templateFunc() { return T::val; }
  std::string templateFuncCall() {
    return templateFunc<TemplateString>();
  }
};

template <class T>
std::basic_string<T> templateStringConditional() {
  if constexpr(std::is_same_v<T, wchar_t>) {
    return L"TEMPLATE";
  } else {
    return "TEMPLATE";
  }
}

template <class T>
std::basic_string<T> templateStringMixedConditional() {
  if constexpr(std::is_same_v<T, wchar_t>) {
    return L"TEMPLATE";
  } else {
    std::string s = "haha";
    return s;
  }
}

void UseTemplateStringConditional() {
  templateStringConditional<char>();
  templateStringConditional<wchar_t>();

  templateStringMixedConditional<char>();
  templateStringMixedConditional<wchar_t>();
}

std::string& Ref() {
  static std::string s = "Ref";
  return s;
}

const std::string& ConstRef() {
  static std::string s = "ConstRef";
  return s;
}

auto autoReturn() {
  // Deduced to const char*
  return "autoReturn";
}

template <class T>
std::basic_string<T> templateString() {
// Intentionally skip templates
  return L"TEMPLATE";
}
std::wstring returnTemplateString() {
  return templateString<wchar_t>();
}

template <typename R> R f() {
// Intentionally skip templates
  return "str";
}
template std::string f();

struct stringOperator {
  operator std::string() const {
    return "conversion";
  }
};

std::string safeFunctionWithLambda() {
  auto lambda = []() -> std::string {
    std::string local = "unsafe";
    return local;
  };
  // TODO: fix hasDescendant(returnStmt(hasReturnValue... which is too strict
  return "safe literal";
}

using Handle = std::wstring;

#ifdef HANDLE_SUPPORTED
Handle handle_or_string();
#else
std::string handle_or_string();
#endif

#ifdef HANDLE_SUPPORTED
Handle
#else
std::string
#endif
my_function() {
  return handle_or_string();
}


namespace TemplatedFunctions {
template <typename MojoType>
extern std::string GetErrorString(const MojoType& mojo_type);

class A{};
class B{};

template <>
std::string GetErrorString(const A& a) {
  return "Can be string_view";
}

template <>
std::string GetErrorString(const B& a) {
  std::string s("Cannot be string_view");
  return s;
}
} // namespace TemplatedFunctions
