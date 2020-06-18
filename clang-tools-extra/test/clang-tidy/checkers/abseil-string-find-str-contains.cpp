// RUN: %check_clang_tidy %s abseil-string-find-str-contains %t -- \
// RUN:   -config="{CheckOptions: []}"

using size_t = decltype(sizeof(int));

namespace std {

// Lightweight standin for std::string.
template <typename C>
class basic_string {
public:
  basic_string();
  basic_string(const basic_string &);
  basic_string(const C *);
  ~basic_string();
  int find(basic_string s, int pos = 0);
  int find(const C *s, int pos = 0);
  int find(char c, int pos = 0);
  static constexpr size_t npos = -1;
};
typedef basic_string<char> string;

// Lightweight standin for std::string_view.
template <typename C>
class basic_string_view {
public:
  basic_string_view();
  basic_string_view(const basic_string_view &);
  basic_string_view(const C *);
  ~basic_string_view();
  int find(basic_string_view s, int pos = 0);
  int find(const C *s, int pos = 0);
  int find(char c, int pos = 0);
  static constexpr size_t npos = -1;
};
typedef basic_string_view<char> string_view;

} // namespace std

namespace absl {

// Lightweight standin for absl::string_view.
class string_view {
public:
  string_view();
  string_view(const string_view &);
  string_view(const char *);
  ~string_view();
  int find(string_view s, int pos = 0);
  int find(const char *s, int pos = 0);
  int find(char c, int pos = 0);
  static constexpr size_t npos = -1;
};

} // namespace absl

// Functions that take and return our various string-like types.
std::string foo_ss(std::string);
std::string_view foo_ssv(std::string_view);
absl::string_view foo_asv(absl::string_view);
std::string bar_ss();
std::string_view bar_ssv();
absl::string_view bar_asv();

// Confirms that find==npos and find!=npos work for each supported type, when
// npos comes from the correct type.
void basic_tests() {
  std::string ss;
  ss.find("a") == std::string::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of find() == npos
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(ss, "a");{{$}}

  ss.find("a") != std::string::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of find() != npos
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(ss, "a");{{$}}

  std::string::npos != ss.find("a");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(ss, "a");{{$}}

  std::string_view ssv;
  ssv.find("a") == std::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(ssv, "a");{{$}}

  ssv.find("a") != std::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(ssv, "a");{{$}}

  std::string_view::npos != ssv.find("a");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(ssv, "a");{{$}}

  absl::string_view asv;
  asv.find("a") == absl::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(asv, "a");{{$}}

  asv.find("a") != absl::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(asv, "a");{{$}}

  absl::string_view::npos != asv.find("a");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(asv, "a");{{$}}
}

// Confirms that it works even if you mix-and-match the type for find and for
// npos.  (One of the reasons for this checker is to clean up cases that
// accidentally mix-and-match like this.  absl::StrContains is less
// error-prone.)
void mismatched_npos() {
  std::string ss;
  ss.find("a") == std::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(ss, "a");{{$}}

  ss.find("a") != absl::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(ss, "a");{{$}}

  std::string_view ssv;
  ssv.find("a") == absl::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(ssv, "a");{{$}}

  ssv.find("a") != std::string::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(ssv, "a");{{$}}

  absl::string_view asv;
  asv.find("a") == std::string::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(asv, "a");{{$}}

  asv.find("a") != std::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(asv, "a");{{$}}
}

// Confirms that it works even when the needle or the haystack are more
// complicated expressions.
void subexpression_tests() {
  std::string ss, ss2;
  foo_ss(ss).find(ss2) == std::string::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(foo_ss(ss), ss2);{{$}}

  ss.find(foo_ss(ss2)) != std::string::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(ss, foo_ss(ss2));{{$}}

  foo_ss(bar_ss()).find(foo_ss(ss2)) != std::string::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(foo_ss(bar_ss()), foo_ss(ss2));{{$}}

  std::string_view ssv, ssv2;
  foo_ssv(ssv).find(ssv2) == std::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(foo_ssv(ssv), ssv2);{{$}}

  ssv.find(foo_ssv(ssv2)) != std::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(ssv, foo_ssv(ssv2));{{$}}

  foo_ssv(bar_ssv()).find(foo_ssv(ssv2)) != std::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(foo_ssv(bar_ssv()), foo_ssv(ssv2));{{$}}

  absl::string_view asv, asv2;
  foo_asv(asv).find(asv2) == absl::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(foo_asv(asv), asv2);{{$}}

  asv.find(foo_asv(asv2)) != absl::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(asv, foo_asv(asv2));{{$}}

  foo_asv(bar_asv()).find(foo_asv(asv2)) != absl::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}absl::StrContains(foo_asv(bar_asv()), foo_asv(asv2));{{$}}
}

// Confirms that it works with string literal, char* and const char* parameters.
void string_literal_and_char_ptr_tests() {
  char *c = nullptr;
  const char *cc = nullptr;

  std::string ss;
  ss.find("c") == std::string::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(ss, "c");{{$}}

  ss.find(c) == std::string::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(ss, c);{{$}}

  ss.find(cc) == std::string::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(ss, cc);{{$}}

  std::string_view ssv;
  ssv.find("c") == std::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(ssv, "c");{{$}}

  ssv.find(c) == std::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(ssv, c);{{$}}

  ssv.find(cc) == std::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(ssv, cc);{{$}}

  absl::string_view asv;
  asv.find("c") == absl::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(asv, "c");{{$}}

  asv.find(c) == absl::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(asv, c);{{$}}

  asv.find(cc) == absl::string_view::npos;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use !absl::StrContains instead of
  // CHECK-FIXES: {{^[[:space:]]*}}!absl::StrContains(asv, cc);{{$}}
}

// Confirms that it does *not* match when the parameter to find() is a char,
// because absl::StrContains is not implemented for char.
void no_char_param_tests() {
  std::string ss;
  ss.find('c') == std::string::npos;

  std::string_view ssv;
  ssv.find('c') == std::string_view::npos;

  absl::string_view asv;
  asv.find('c') == absl::string_view::npos;
}

#define COMPARE_MACRO(x, y) ((x) == (y))
#define FIND_MACRO(x, y) ((x).find(y))
#define FIND_COMPARE_MACRO(x, y, z) ((x).find(y) == (z))

// Confirms that it does not match when a macro is involved.
void no_macros() {
  std::string s;
  COMPARE_MACRO(s.find("a"), std::string::npos);
  FIND_MACRO(s, "a") == std::string::npos;
  FIND_COMPARE_MACRO(s, "a", std::string::npos);
}

// Confirms that it does not match when the pos parameter is non-zero.
void no_nonzero_pos() {
  std::string ss;
  ss.find("a", 1) == std::string::npos;

  std::string_view ssv;
  ssv.find("a", 2) == std::string_view::npos;

  absl::string_view asv;
  asv.find("a", 3) == std::string_view::npos;
}

// Confirms that it does not match when it's compared to something other than
// npos, even if the value is the same as npos.
void no_non_npos() {
  std::string ss;
  ss.find("a") == 0;
  ss.find("a") == 1;
  ss.find("a") == -1;

  std::string_view ssv;
  ssv.find("a") == 0;
  ssv.find("a") == 1;
  ssv.find("a") == -1;

  absl::string_view asv;
  asv.find("a") == 0;
  asv.find("a") == 1;
  asv.find("a") == -1;
}

// Confirms that it does not match if the two operands are the same.
void no_symmetric_operands() {
  std::string ss;
  ss.find("a") == ss.find("a");
  std::string::npos == std::string::npos;
}
