// RUN: %check_clang_tidy %s bugprone-string-integer-assignment %t

namespace std {
template<typename T>
struct basic_string {
  basic_string& operator=(T);
  basic_string& operator=(basic_string);
  basic_string& operator+=(T);
  basic_string& operator+=(basic_string);
};

typedef basic_string<char> string;
typedef basic_string<wchar_t> wstring;

int tolower(int i);
int toupper(int i);
}

int tolower(int i);
int toupper(int i);

typedef int MyArcaneChar;

int main() {
  std::string s;
  std::wstring ws;
  int x = 5;

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
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: an integer is interpreted as a chara
// CHECK-FIXES: {{^}}  as = 6;{{$}}

  s += toupper(x);
  s += tolower(x);
  s += std::tolower(x);
}
