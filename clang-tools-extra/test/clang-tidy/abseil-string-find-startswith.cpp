// RUN: %check_clang_tidy %s abseil-string-find-startswith %t

namespace std {
template <typename T> class allocator {};
template <typename T> class char_traits {};
template <typename C, typename T = std::char_traits<C>,
          typename A = std::allocator<C>>
struct basic_string {
  basic_string();
  basic_string(const basic_string &);
  basic_string(const C *, const A &a = A());
  ~basic_string();
  int find(basic_string<C> s, int pos = 0);
  int find(const char *s, int pos = 0);
};
typedef basic_string<char> string;
typedef basic_string<wchar_t> wstring;
} // namespace std

std::string foo(std::string);
std::string bar();

#define A_MACRO(x, y) ((x) == (y))

void tests(std::string s) {
  s.find("a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StartsWith instead of find() == 0 [abseil-string-find-startswith]
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StartsWith(s, "a");{{$}}

  s.find(s) == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StartsWith
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StartsWith(s, s);{{$}}

  s.find("aaa") != 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StartsWith
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StartsWith(s, "aaa");{{$}}

  s.find(foo(foo(bar()))) != 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StartsWith
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StartsWith(s, foo(foo(bar())));{{$}}

  if (s.find("....") == 0) { /* do something */ }
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use absl::StartsWith
  // CHECK-FIXES: {{^[[:space:]]*}}if (absl::StartsWith(s, "....")) { /* do something */ }{{$}}

  0 != s.find("a");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StartsWith
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StartsWith(s, "a");{{$}}

  // expressions that don't trigger the check are here.
  A_MACRO(s.find("a"), 0);
  s.find("a", 1) == 0;
  s.find("a", 1) == 1;
  s.find("a") == 1;
}
