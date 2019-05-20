// RUN: %check_clang_tidy -std=c++11,c++14 %s readability-redundant-string-init %t
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
  // MSVC headers define two constructors instead of using optional arguments.
  basic_string(const C *);
  basic_string(const C *, const A &);
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
