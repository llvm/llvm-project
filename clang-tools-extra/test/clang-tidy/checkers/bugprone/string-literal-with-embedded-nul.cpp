// RUN: %check_clang_tidy %s bugprone-string-literal-with-embedded-nul %t

namespace std {
template <typename T>
class allocator {};
template <typename T>
class char_traits {};
template <typename C, typename T, typename A>
struct basic_string {
  typedef basic_string<C, T, A> _Type;
  basic_string();
  basic_string(const C *p, const A &a = A());

  _Type& operator+=(const C* s);
  _Type& operator=(const C* s);
};

typedef basic_string<char, std::char_traits<char>, std::allocator<char>> string;
typedef basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t>> wstring;
}

bool operator==(const std::string&, const char*);
bool operator==(const char*, const std::string&);


const char Valid[] = "This is valid \x12.";
const char Strange[] = "This is strange \0x12 and must be fixed";
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: suspicious embedded NUL character [bugprone-string-literal-with-embedded-nul]

const char textA[] = "\0x01\0x02\0x03\0x04";
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: suspicious embedded NUL character
const wchar_t textW[] = L"\0x01\0x02\0x03\0x04";
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: suspicious embedded NUL character

const char A[] = "\0";
const char B[] = "\0x";
const char C[] = "\0x1";
const char D[] = "\0x11";
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: suspicious embedded NUL character

const wchar_t E[] = L"\0";
const wchar_t F[] = L"\0x";
const wchar_t G[] = L"\0x1";
const wchar_t H[] = L"\0x11";
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: suspicious embedded NUL character

const char I[] = "\000\000\000\000";
const char J[] = "\0\0\0\0\0\0";
const char K[] = "";

const char L[] = "\0x12" "\0x12" "\0x12" "\0x12";
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: suspicious embedded NUL character

void TestA() {
  std::string str1 = "abc\0def";
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: truncated string literal
  std::string str2 = "\0";
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: truncated string literal
  std::string str3("\0");
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: truncated string literal
  std::string str4{"\x00\x01\x02\x03"};
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: truncated string literal

  std::string str;
  str += "abc\0def";
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: truncated string literal
  str = "abc\0def";
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: truncated string literal

  if (str == "abc\0def") return;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: truncated string literal
  if ("abc\0def" == str) return;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: truncated string literal
}

void TestW() {
  std::wstring str1 = L"abc\0def";
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: truncated string literal
  std::wstring str2 = L"\0";
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: truncated string literal
  std::wstring str3(L"\0");
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: truncated string literal
  std::wstring str4{L"\x00\x01\x02\x03"};
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: truncated string literal
}
