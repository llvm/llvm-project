// RUN: %check_clang_tidy %s readability-redundant-string-init %t \
// RUN:   -config="{CheckOptions: \
// RUN:             {readability-redundant-string-init.StringNames: \
// RUN:                '::std::basic_string;::std::basic_string_view;our::TestString'} \
// RUN:             }"

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

template <typename C, typename T = std::char_traits<C>, typename A = std::allocator<C>>
struct basic_string_view {
  using size_type = decltype(sizeof(0));

  basic_string_view();
  basic_string_view(const basic_string_view &);
  basic_string_view(const C *, size_type);
  basic_string_view(const C *);
  template <class It, class End>
  basic_string_view(It, End);
};
typedef basic_string_view<char> string_view;
typedef basic_string_view<wchar_t> wstring_view;
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
  std::string e{""};
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant string initialization
  // CHECK-FIXES: std::string e;
  std::string f = {""};
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant string initialization
  // CHECK-FIXES: std::string f;

  std::string u = "u";
  std::string w("w");
  std::string x = R"(x)";
  std::string y(R"(y)");
  std::string z;
}

void fview() {
  std::string_view a = "";
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: redundant string initialization [readability-redundant-string-init]
  // CHECK-FIXES: std::string_view a;
  std::string_view b("");
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: redundant string initialization
  // CHECK-FIXES: std::string_view b;
  std::string_view c = R"()";
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: redundant string initialization
  // CHECK-FIXES: std::string_view c;
  std::string_view d(R"()");
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: redundant string initialization
  // CHECK-FIXES: std::string_view d;
  std::string_view e{""};
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: redundant string initialization
  // CHECK-FIXES: std::string_view e;
  std::string_view f = {""};
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: redundant string initialization
  // CHECK-FIXES: std::string_view f;

  std::string_view u = "u";
  std::string_view w("w");
  std::string_view x = R"(x)";
  std::string_view y(R"(y)");
  std::string_view z;
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

void gview() {
  std::wstring_view a = L"";
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: redundant string initialization [readability-redundant-string-init]
  // CHECK-FIXES: std::wstring_view a;
  std::wstring_view b(L"");
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: redundant string initialization
  // CHECK-FIXES: std::wstring_view b;
  std::wstring_view c = L"";
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: redundant string initialization
  // CHECK-FIXES: std::wstring_view c;
  std::wstring_view d(L"");
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: redundant string initialization
  // CHECK-FIXES: std::wstring_view d;
  std::wstring_view e{L""};
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: redundant string initialization
  // CHECK-FIXES: std::wstring_view e;
  std::wstring_view f = {L""};
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: redundant string initialization
  // CHECK-FIXES: std::wstring_view f;

  std::wstring_view u = L"u";
  std::wstring_view w(L"w");
  std::wstring_view x = LR"(x)";
  std::wstring_view y(LR"(y)");
  std::wstring_view z;
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

class Foo {
  std::string A = "";
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant string initialization
  // CHECK-FIXES:  std::string A;
  std::string B;
  std::string C;
  std::string D;
  std::string E = "NotEmpty";

public:
  // Check redundant constructor where Field has a redundant initializer.
  Foo() : A("") {}
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant string initialization
  // CHECK-FIXES:  Foo()  {}

  // Check redundant constructor where Field has no initializer.
  Foo(char) : B("") {}
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant string initialization
  // CHECK-FIXES:  Foo(char)  {}

  // Check redundant constructor where Field has a valid initializer.
  Foo(long) : E("") {}
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: redundant string initialization
  // CHECK-FIXES:  Foo(long) : E() {}

  // Check how it handles removing 1 initializer, and defaulting the other.
  Foo(int) : B(""), E("") {}
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: redundant string initialization
  // CHECK-MESSAGES: [[@LINE-2]]:21: warning: redundant string initialization
  // CHECK-FIXES:  Foo(int) :  E() {}

  Foo(short) : B{""} {}
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: redundant string initialization
  // CHECK-FIXES:  Foo(short)  {}

  Foo(float) : A{""}, B{""} {}
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: redundant string initialization
  // CHECK-MESSAGES: [[@LINE-2]]:23: warning: redundant string initialization
  // CHECK-FIXES:  Foo(float)  {}

  // Check how it handles removing some redundant initializers while leaving
  // valid initializers intact.
  Foo(std::string Arg) : A(Arg), B(""), C("NonEmpty"), D(R"()"), E("") {}
  // CHECK-MESSAGES: [[@LINE-1]]:34: warning: redundant string initialization
  // CHECK-MESSAGES: [[@LINE-2]]:56: warning: redundant string initialization
  // CHECK-MESSAGES: [[@LINE-3]]:66: warning: redundant string initialization
  // CHECK-FIXES:  Foo(std::string Arg) : A(Arg),  C("NonEmpty"),  E() {}
};
