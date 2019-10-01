// RUN: %check_clang_tidy %s bugprone-string-integer-assignment %t

namespace std {
template<typename T>
struct basic_string {
  basic_string& operator=(T);
  basic_string& operator=(basic_string);
  basic_string& operator+=(T);
  basic_string& operator+=(basic_string);
  const T &operator[](int i) const;
  T &operator[](int i);
};

typedef basic_string<char> string;
typedef basic_string<wchar_t> wstring;

int tolower(int i);
int toupper(int i);
}

int tolower(int i);
int toupper(int i);

typedef int MyArcaneChar;

constexpr char kCharConstant = 'a';

int main() {
  std::string s;
  std::wstring ws;
  int x = 5;
  const char c = 'c';

  s = 6;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: an integer is interpreted as a character code when assigning {{.*}} [bugprone-string-integer-assignment]
// CHECK-FIXES: {{^}}  s = '6';{{$}}
  s = 66;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: an integer is interpreted as a chara
// CHECK-FIXES: {{^}}  s = "66";{{$}}
  s = x;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: an integer is interpreted as a chara
// CHECK-FIXES: {{^}}  s = std::to_string(x);{{$}}
  s = 'c';
  s = static_cast<char>(6);

// +=
  ws += 6;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: an integer is interpreted as a chara
// CHECK-FIXES: {{^}}  ws += L'6';{{$}}
  ws += 66;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: an integer is interpreted as a chara
// CHECK-FIXES: {{^}}  ws += L"66";{{$}}
  ws += x;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: an integer is interpreted as a chara
// CHECK-FIXES: {{^}}  ws += std::to_wstring(x);{{$}}
  ws += L'c';
  ws += (wchar_t)6;

  std::basic_string<MyArcaneChar> as;
  as = 6;
  as = static_cast<MyArcaneChar>(6);
  as = 'a';

  s += toupper(x);
  s += tolower(x);
  s += (std::tolower(x));

  s += c & s[1];
  s += c ^ s[1];
  s += c | s[1];

  s[x] += 1;
  s += s[x];
  as += as[x];

  // Likely character expressions.
  s += x & 0xff;
  s += 0xff & x;
  s += x % 26;
  s += 26 % x;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: an integer is interpreted as a chara
  // CHECK-FIXES: {{^}}  s += std::to_string(26 % x);{{$}}
  s += c | 0x80;
  s += c | 0x8000;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: an integer is interpreted as a chara
  // CHECK-FIXES: {{^}}  s += std::to_string(c | 0x8000);{{$}}
  as += c | 0x8000;

  s += 'a' + (x % 26);
  s += kCharConstant + (x % 26);
  s += 'a' + (s[x] & 0xf);
  s += (x % 10) + 'b';

  s += x > 255 ? c : x;
  s += x > 255 ? 12 : x;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: an integer is interpreted as a chara
  // CHECK-FIXES: {{^}}  s += std::to_string(x > 255 ? 12 : x);{{$}}
}

namespace instantiation_dependent_exprs {
template<typename T>
struct S {
  static constexpr T t = 0x8000;
  std::string s;
  void f(char c) { s += c | static_cast<int>(t); }
};

template S<int>;
}
