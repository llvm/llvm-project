// RUN: %check_clang_tidy -std=c++11,c++14 %s readability-redundant-string-init %t \
// RUN:   -config="{CheckOptions: \
// RUN:             [{key: readability-redundant-string-init.StringNames, \
// RUN:               value: '::std::basic_string;our::TestString'}] \
// RUN:             }"
// FIXME: Fix the checker to work in C++17 mode.

namespace std {
template <typename T>
class allocator {};
template <typename T>
class char_traits {};
template <typename C, typename T = std::char_traits<C>, typename A = std::allocator<C>>
struct basic_string {
  basic_string();
  basic_string(const basic_string&);
  basic_string(const C *, const A &a = A());
  ~basic_string();
};
typedef basic_string<char> string;
typedef basic_string<wchar_t> wstring;
}

void f() {
  std::string a = "";
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant string initialization [readability-redundant-string-init]
  // CHECK-FIXES: std::string a;
  std::string b("");
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant string initialization
  // CHECK-FIXES: std::string b;
  std::string c = R"()";
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant string initialization
  // CHECK-FIXES: std::string c;
  std::string d(R"()");
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant string initialization
  // CHECK-FIXES: std::string d;

  std::string u = "u";
  std::string w("w");
  std::string x = R"(x)";
  std::string y(R"(y)");
  std::string z;
}

void g() {
  std::wstring a = L"";
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: redundant string initialization
  // CHECK-FIXES: std::wstring a;
  std::wstring b(L"");
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: redundant string initialization
  // CHECK-FIXES: std::wstring b;
  std::wstring c = LR"()";
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: redundant string initialization
  // CHECK-FIXES: std::wstring c;
  std::wstring d(LR"()");
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: redundant string initialization
  // CHECK-FIXES: std::wstring d;

  std::wstring u = L"u";
  std::wstring w(L"w");
  std::wstring x = LR"(x)";
  std::wstring y(LR"(y)");
  std::wstring z;
}

template <typename T>
void templ() {
  std::string s = "";
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant string initialization
  // CHECK-FIXES: std::string s;
}

#define M(x) x
#define N { std::string s = ""; }
// CHECK-FIXES: #define N { std::string s = ""; }

void h() {
  templ<int>();
  templ<double>();

  M({ std::string s = ""; })
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: redundant string initialization
  // CHECK-FIXES: M({ std::string s; })

  N
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: redundant string initialization
  // CHECK-FIXES: N
  N
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: redundant string initialization
  // CHECK-FIXES: N
}

typedef std::string MyString;
#define STRING MyString
#define DECL_STRING(name, val) STRING name = val

void i() {
  MyString a = "";
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: redundant string initialization
  // CHECK-FIXES: MyString a;
  STRING b = "";
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: redundant string initialization
  // CHECK-FIXES: STRING b;
  MyString c = "" "" "";
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: redundant string initialization
  // CHECK-FIXES: MyString c;
  STRING d = "" "" "";
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: redundant string initialization
  // CHECK-FIXES: STRING d;
  DECL_STRING(e, "");
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant string initialization

  MyString f = "u";
  STRING g = "u";
  DECL_STRING(h, "u");
}

#define EMPTY_STR ""
void j() {
  std::string a(EMPTY_STR);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant string initialization
  // CHECK-FIXES: std::string a;
  std::string b = (EMPTY_STR);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant string initialization
  // CHECK-FIXES: std::string b;

  std::string c(EMPTY_STR "u" EMPTY_STR);
}

void k() {
  std::string a = "", b = "", c = "";
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant string initialization
  // CHECK-MESSAGES: [[@LINE-2]]:23: warning: redundant string initialization
  // CHECK-MESSAGES: [[@LINE-3]]:31: warning: redundant string initialization
  // CHECK-FIXES: std::string a, b, c;

  std::string d = "u", e = "u", f = "u";

  std::string g = "u", h = "", i = "uuu", j = "", k;
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: redundant string initialization
  // CHECK-MESSAGES: [[@LINE-2]]:43: warning: redundant string initialization
  // CHECK-FIXES: std::string g = "u", h, i = "uuu", j, k;
}

// These cases should not generate warnings.
extern void Param1(std::string param = "");
extern void Param2(const std::string& param = "");
void Param3(std::string param = "") {}
void Param4(STRING param = "") {}

namespace our {
struct TestString {
  TestString();
  TestString(const TestString &);
  TestString(const char *);
  ~TestString();
};
}

void ourTestStringTests() {
  our::TestString a = "";
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: redundant string initialization
  // CHECK-FIXES: our::TestString a;
  our::TestString b("");
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: redundant string initialization
  // CHECK-FIXES: our::TestString b;
  our::TestString c = R"()";
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: redundant string initialization
  // CHECK-FIXES: our::TestString c;
  our::TestString d(R"()");
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: redundant string initialization
  // CHECK-FIXES: our::TestString d;

  our::TestString u = "u";
  our::TestString w("w");
  our::TestString x = R"(x)";
  our::TestString y(R"(y)");
  our::TestString z;
}

namespace their {
using TestString = our::TestString;
}

// their::TestString is the same type so should warn / fix
void theirTestStringTests() {
  their::TestString a = "";
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: redundant string initialization
  // CHECK-FIXES: their::TestString a;
  their::TestString b("");
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: redundant string initialization
  // CHECK-FIXES: their::TestString b;
}

namespace other {
// Identical declarations to above but different type
struct TestString {
  TestString();
  TestString(const TestString &);
  TestString(const char *);
  ~TestString();
};

// Identical declarations to above but different type
template <typename T>
class allocator {};
template <typename T>
class char_traits {};
template <typename C, typename T = std::char_traits<C>, typename A = std::allocator<C>>
struct basic_string {
  basic_string();
  basic_string(const basic_string &);
  basic_string(const C *, const A &a = A());
  ~basic_string();
};
typedef basic_string<char> string;
typedef basic_string<wchar_t> wstring;
}

// other::TestString, other::string, other::wstring are unrelated to the types
// being checked. No warnings / fixes should be produced for these types.
void otherTestStringTests() {
  other::TestString a = "";
  other::TestString b("");
  other::string c = "";
  other::string d("");
  other::wstring e = L"";
  other::wstring f(L"");
}
