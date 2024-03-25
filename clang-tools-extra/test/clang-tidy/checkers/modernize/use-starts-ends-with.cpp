// RUN: %check_clang_tidy -std=c++20 %s modernize-use-starts-ends-with %t -- \
// RUN:   -- -isystem %clang_tidy_headers

#include <string>

std::string foo(std::string);
std::string bar();

class sub_string : public std::string {};
class sub_sub_string : public sub_string {};

struct string_like {
  bool starts_with(const char *s) const;
  size_t find(const char *s, size_t pos = 0) const;
};

struct string_like_camel {
  bool startsWith(const char *s) const;
  size_t find(const char *s, size_t pos = 0) const;
};

struct prefer_underscore_version {
  bool starts_with(const char *s) const;
  bool startsWith(const char *s) const;
  size_t find(const char *s, size_t pos = 0) const;
};

struct prefer_underscore_version_flip {
  bool startsWith(const char *s) const;
  bool starts_with(const char *s) const;
  size_t find(const char *s, size_t pos = 0) const;
};

struct prefer_underscore_version_inherit : public string_like {
  bool startsWith(const char *s) const;
};

void test(std::string s, std::string_view sv, sub_string ss, sub_sub_string sss,
          string_like sl, string_like_camel slc, prefer_underscore_version puv,
          prefer_underscore_version_flip puvf,
          prefer_underscore_version_inherit puvi) {
  s.find("a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of find() == 0
  // CHECK-FIXES: s.starts_with("a");

  (((((s)).find("a")))) == ((0));
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: ((s)).starts_with("a");

  (s + "a").find("a") == ((0));
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: (s + "a").starts_with("a");

  s.find(s) == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: s.starts_with(s);

  s.find("aaa") != 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: !s.starts_with("aaa");

  s.find(foo(foo(bar()))) != 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: !s.starts_with(foo(foo(bar())));

  if (s.find("....") == 0) { /* do something */ }
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: if (s.starts_with("...."))

  0 != s.find("a");
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: !s.starts_with("a");

  s.rfind("a", 0) == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of rfind() == 0
  // CHECK-FIXES: s.starts_with("a");

  s.rfind(s, 0) == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: s.starts_with(s);

  s.rfind("aaa", 0) != 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: !s.starts_with("aaa");

  s.rfind(foo(foo(bar())), 0) != 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: !s.starts_with(foo(foo(bar())));

  if (s.rfind("....", 0) == 0) { /* do something */ }
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: if (s.starts_with("...."))

  0 != s.rfind("a", 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: !s.starts_with("a");

  #define STR(x) std::string(x)
  0 == STR(s).find("a");
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: STR(s).starts_with("a");

  #define STRING s
  if (0 == STRING.find("ala")) { /* do something */}
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: if (STRING.starts_with("ala"))

  #define FIND find
  s.FIND("a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: s.starts_with("a")

  #define PREFIX "a"
  s.find(PREFIX) == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: s.starts_with(PREFIX)

  #define ZERO 0
  s.find("a") == ZERO;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: s.starts_with("a")

  sv.find("a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: sv.starts_with("a");

  sv.rfind("a", 0) != 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: !sv.starts_with("a");

  ss.find("a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: ss.starts_with("a");

  sss.find("a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: ss.starts_with("a");

  sl.find("a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: sl.starts_with("a");

  slc.find("a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use startsWith
  // CHECK-FIXES: slc.startsWith("a");

  puv.find("a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: puv.starts_with("a");

  puvf.find("a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: puvf.starts_with("a");

  // Here, the subclass has startsWith, the superclass has starts_with.
  // We prefer the version from the subclass.
  puvi.find("a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use startsWith
  // CHECK-FIXES: puvi.startsWith("a");

  // Expressions that don't trigger the check are here.
  #define EQ(x, y) ((x) == (y))
  EQ(s.find("a"), 0);

  #define DOTFIND(x, y) (x).find(y)
  DOTFIND(s, "a") == 0;
}
