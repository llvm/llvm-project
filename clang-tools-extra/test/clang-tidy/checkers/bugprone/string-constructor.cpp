// RUN: %check_clang_tidy %s bugprone-string-constructor %t

namespace std {
template <typename T>
class allocator {};
template <typename T>
class char_traits {};
template <typename C, typename T = std::char_traits<C>, typename A = std::allocator<C> >
struct basic_string {
  basic_string();
  basic_string(const C*, unsigned int size);
  basic_string(const C *, const A &allocator = A());
  basic_string(unsigned int size, C c);
};
typedef basic_string<char> string;
typedef basic_string<wchar_t> wstring;

template <typename C, typename T = std::char_traits<C>>
struct basic_string_view {
  basic_string_view();
  basic_string_view(const C *, unsigned int size);
  basic_string_view(const C *);
};
typedef basic_string_view<char> string_view;
typedef basic_string_view<wchar_t> wstring_view;
}

const char* kText = "";
const char kText2[] = "";
extern const char kText3[];

void Test() {
  std::string str('x', 4);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: string constructor parameters are probably swapped; expecting string(count, character) [bugprone-string-constructor]
  // CHECK-FIXES: std::string str(4, 'x');
  std::wstring wstr(L'x', 4);
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: string constructor parameters are probably swapped
  // CHECK-FIXES: std::wstring wstr(4, L'x');
  std::string s0(0, 'x');
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: constructor creating an empty string
  std::string s1(-4, 'x');
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: negative value used as length parameter
  std::string s2(0x1000000, 'x');
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: suspicious large length parameter

  std::string q0("test", 0);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: constructor creating an empty string
  std::string q1(kText, -4);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: negative value used as length parameter
  std::string q2("test", 200);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: length is bigger than string literal size
  std::string q3(kText, 200);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: length is bigger than string literal size
  std::string q4(kText2, 200);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: length is bigger than string literal size
  std::string q5(kText3,  0x1000000);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: suspicious large length parameter
  std::string q6(nullptr);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: constructing string from nullptr is undefined behaviour
  std::string q7 = 0;
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: constructing string from nullptr is undefined behaviour
}

void TestView() {
  std::string_view q0("test", 0);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: constructor creating an empty string
  std::string_view q1(kText, -4);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: negative value used as length parameter
  std::string_view q2("test", 200);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: length is bigger than string literal size
  std::string_view q3(kText, 200);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: length is bigger than string literal size
  std::string_view q4(kText2, 200);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: length is bigger than string literal size
  std::string_view q5(kText3, 0x1000000);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: suspicious large length parameter
  std::string_view q6(nullptr);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: constructing string from nullptr is undefined behaviour
  std::string_view q7 = 0;
  // CHECK-MESSAGES: [[@LINE-1]]:25: warning: constructing string from nullptr is undefined behaviour
}

std::string StringFromZero() {
  return 0;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: constructing string from nullptr is undefined behaviour
}

std::string_view StringViewFromZero() {
  return 0;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: constructing string from nullptr is undefined behaviour
}

void Valid() {
  std::string empty();
  std::string str(4, 'x');
  std::wstring wstr(4, L'x');
  std::string s1("test", 4);
  std::string s2("test", 3);
  std::string s3("test");

  std::string_view emptyv();
  std::string_view sv1("test", 4);
  std::string_view sv2("test", 3);
  std::string_view sv3("test");
}

namespace instantiation_dependent_exprs {
template<typename T>
struct S {
  bool x;
  std::string f() { return x ? "a" : "b"; }
  std::string_view g() { return x ? "a" : "b"; }
};
}
