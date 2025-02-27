// RUN: %check_clang_tidy -std=c++20 %s modernize-use-starts-ends-with %t -- \
// RUN:   -- -isystem %clang_tidy_headers

#include <string.h>
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

void test(std::string s, std::string_view sv, sub_string ss, sub_sub_string sss,
          string_like sl, string_like_camel slc, prefer_underscore_version puv,
          prefer_underscore_version_flip puvf) {
  s.find("a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of find [modernize-use-starts-ends-with]
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
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of rfind [modernize-use-starts-ends-with]
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

  s.compare(0, 1, "a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of compare [modernize-use-starts-ends-with]
  // CHECK-FIXES: s.starts_with("a");

  s.compare(0, 1, "a") != 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of compare [modernize-use-starts-ends-with]
  // CHECK-FIXES: !s.starts_with("a");

  s.compare(0, strlen("a"), "a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: s.starts_with("a");

  s.compare(0, std::strlen("a"), "a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: s.starts_with("a");

  s.compare(0, std::strlen(("a")), "a") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: s.starts_with("a");

  s.compare(0, std::strlen(("a")), (("a"))) == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: s.starts_with("a");

  s.compare(0, s.size(), s) == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: s.starts_with(s);

  s.compare(0, s.length(), s) == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: s.starts_with(s);

  0 != s.compare(0, sv.length(), sv);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: s.starts_with(sv);

  #define LENGTH(x) (x).length()
  s.compare(0, LENGTH(s), s) == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: s.starts_with(s);

  s.compare(ZERO, LENGTH(s), s) == ZERO;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: s.starts_with(s);

  s.compare(ZERO, LENGTH(sv), sv) != 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with
  // CHECK-FIXES: !s.starts_with(sv);

  s.compare(s.size() - 6, 6, "suffix") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use ends_with
  // CHECK-FIXES: s.ends_with("suffix");

  s.compare(s.size() - 6, strlen("abcdef"), "suffix") == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use ends_with
  // CHECK-FIXES: s.ends_with("suffix");

  std::string suffix = "suffix";
  s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use ends_with
  // CHECK-FIXES: s.ends_with(suffix);

  s.rfind("suffix") == s.size() - 6;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use ends_with
  // CHECK-FIXES: s.ends_with("suffix");

  s.rfind("suffix") == s.size() - strlen("suffix");
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use ends_with
  // CHECK-FIXES: s.ends_with("suffix");

  s.rfind(suffix) == s.size() - suffix.size();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use ends_with
  // CHECK-FIXES: s.ends_with(suffix);

  s.rfind(suffix, std::string::npos) == s.size() - suffix.size();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use ends_with
  // CHECK-FIXES: s.ends_with(suffix);

  s.rfind(suffix) == (s.size() - suffix.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use ends_with
  // CHECK-FIXES: s.ends_with(suffix);

  s.rfind(suffix, s.npos) == (s.size() - suffix.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use ends_with
  // CHECK-FIXES: s.ends_with(suffix);

  s.rfind(suffix, s.npos) == (((s.size()) - (suffix.size())));
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use ends_with
  // CHECK-FIXES: s.ends_with(suffix);

  s.rfind(suffix) != s.size() - suffix.size();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use ends_with
  // CHECK-FIXES: !s.ends_with(suffix);

  (s.size() - suffix.size()) == s.rfind(suffix);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use ends_with
  // CHECK-FIXES: s.ends_with(suffix);

  struct S {
    std::string s;
  } t;
  t.s.rfind(suffix) == (t.s.size() - suffix.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use ends_with
  // CHECK-FIXES: t.s.ends_with(suffix);

  // Expressions that don't trigger the check are here.
  #define EQ(x, y) ((x) == (y))
  EQ(s.find("a"), 0);

  #define DOTFIND(x, y) (x).find(y)
  DOTFIND(s, "a") == 0;

  #define STARTS_WITH_COMPARE(x, y) (x).compare(0, (x).size(), (y))
  STARTS_WITH_COMPARE(s, s) == 0;

  s.compare(0, 1, "ab") == 0;
  s.rfind(suffix, 1) == s.size() - suffix.size();

  #define STR(x) std::string(x)
  0 == STR(s).find("a");

  #define STRING s
  if (0 == STRING.find("ala")) { /* do something */}
}

void test_substr() {
    std::string str("hello world");
    std::string prefix = "hello";
    
    // Basic pattern
    str.substr(0, 5) == "hello";
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of substr [modernize-use-starts-ends-with]
    // CHECK-FIXES: str.starts_with("hello");
    
    // With string literal on left side
    "hello" == str.substr(0, 5);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of substr [modernize-use-starts-ends-with]
    // CHECK-FIXES: str.starts_with("hello");
    
    // Inequality comparison
    str.substr(0, 5) != "world";
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of substr [modernize-use-starts-ends-with]
    // CHECK-FIXES: !str.starts_with("world");
    
    // Ensure non-zero start position is not transformed
    str.substr(1, 5) == "hello";
    str.substr(0, 4) == "hello"; // Length mismatch
    
    size_t len = 5;
    str.substr(0, len) == "hello"; // Non-constant length

    // String literal with size calculation
    str.substr(0, strlen("hello")) == "hello";
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of substr [modernize-use-starts-ends-with]
    // CHECK-FIXES: str.starts_with("hello");

    str.substr(0, prefix.size()) == prefix;
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of substr [modernize-use-starts-ends-with]
    // CHECK-FIXES: str.starts_with(prefix);

    str.substr(0, prefix.length()) == prefix;
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of substr [modernize-use-starts-ends-with]
    // CHECK-FIXES: str.starts_with(prefix);

    // Tests to verify macro behavior
    #define MSG "hello"
    str.substr(0, strlen(MSG)) == MSG;
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of substr [modernize-use-starts-ends-with]
    // CHECK-FIXES: str.starts_with(MSG);

    #define STARTS_WITH(X, Y) (X).substr(0, (Y).size()) == (Y)
    STARTS_WITH(str, prefix);

    #define SUBSTR(X, A, B) (X).substr((A), (B))
    SUBSTR(str, 0, 6) == "prefix";

    #define STR() str
    SUBSTR(STR(), 0, 6) == "prefix";
    "prefix" == SUBSTR(STR(), 0, 6);

    str.substr(0, strlen("hello123")) == "hello";
}

void test_operator_rewriting(std::string str, std::string prefix) {
  str.substr(0, prefix.size()) == prefix;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of substr
  // CHECK-FIXES: str.starts_with(prefix);

  str.substr(0, prefix.size()) != prefix;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use starts_with instead of substr
  // CHECK-FIXES: !str.starts_with(prefix);
}
