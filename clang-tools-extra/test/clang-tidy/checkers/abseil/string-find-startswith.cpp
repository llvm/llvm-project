// RUN: %check_clang_tidy -std=c++17 %s abseil-string-find-startswith %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:     {abseil-string-find-startswith.StringLikeClasses: \
// RUN:       '::std::basic_string;::std::basic_string_view;::basic_string'}}" \
// RUN:   -- -isystem %clang_tidy_headers

#include <string>

using size_t = decltype(sizeof(int));

namespace std {
struct cxx_string {
  int find(const char *s, int pos = 0);
  int rfind(const char *s, int pos = npos);
  static constexpr size_t npos = -1;
};
} // namespace std

struct basic_string : public std::cxx_string {
  basic_string();
};
typedef basic_string global_string;

std::string foo(std::string);
std::string bar();

#define A_MACRO(x, y) ((x) == (y))

void tests(std::string s, global_string s2, std::string_view sv) {
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

  s2.find("a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StartsWith
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StartsWith(s2, "a");{{$}}

  s.rfind("a", 0) == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StartsWith instead of rfind() == 0 [abseil-string-find-startswith]
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StartsWith(s, "a");{{$}}

  s.rfind(s, 0) == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StartsWith
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StartsWith(s, s);{{$}}

  s.rfind("aaa", 0) != 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StartsWith
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StartsWith(s, "aaa");{{$}}

  s.rfind(foo(foo(bar())), 0) != 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StartsWith
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StartsWith(s, foo(foo(bar())));{{$}}

  if (s.rfind("....", 0) == 0) { /* do something */ }
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use absl::StartsWith
  // CHECK-FIXES: {{^[[:space:]]*}}if (absl::StartsWith(s, "....")) { /* do something */ }{{$}}

  0 != s.rfind("a", 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StartsWith
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StartsWith(s, "a");{{$}}

  s2.rfind("a", 0) == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StartsWith
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StartsWith(s2, "a");{{$}}

  sv.find("a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StartsWith
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StartsWith(sv, "a");{{$}}

  sv.rfind("a", 0) != 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StartsWith
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StartsWith(sv, "a");{{$}}

  // expressions that don't trigger the check are here.
  A_MACRO(s.find("a"), 0);
  A_MACRO(s.rfind("a", 0), 0);
  s.find("a", 1) == 0;
  s.find("a", 1) == 1;
  s.find("a") == 1;
  s.rfind("a", 1) == 0;
  s.rfind("a", 1) == 1;
  s.rfind("a") == 0;
  s.rfind("a") == 1;
}
