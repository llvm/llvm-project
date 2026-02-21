// RUN: %check_clang_tidy %s readability-stringview-substr -std=c++17 %t

namespace std {
template <typename T>
class basic_string_view {
public:
  using size_type = unsigned long;
  static constexpr size_type npos = -1;

  basic_string_view(const char*) {}
  basic_string_view substr(size_type pos, size_type count = npos) const { return *this; }
  void remove_prefix(size_type n) {}
  void remove_suffix(size_type n) {}
  size_type length() const { return 0; }
  size_type size() const { return 0; }
  basic_string_view& operator=(const basic_string_view&) { return *this; }
};

using string_view = basic_string_view<char>;
} // namespace std

void test_basic() {
  std::string_view sv("test");
  std::string_view sv2("test");

  // Case 1: remove_prefix
  sv = sv.substr(5);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer 'remove_prefix' over 'substr' for removing characters from the start [readability-stringview-substr]
  // CHECK-FIXES: sv.remove_prefix(5);

  // Case 3: remove_suffix with length()
  sv = sv.substr(0, sv.length() - 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer 'remove_suffix' over 'substr' for removing characters from the end [readability-stringview-substr]
  // CHECK-FIXES: sv.remove_suffix(3);

  // Case 3: remove_suffix with complex expression
  sv = sv.substr(0, sv.length() - (3 + 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer 'remove_suffix' over 'substr' for removing characters from the end [readability-stringview-substr]
  // CHECK-FIXES: sv.remove_suffix((3 + 2));
}

void test_size_method() {
  std::string_view sv("test");
  std::string_view sv2("test");

  // Case 3: remove_suffix with size()
  sv = sv.substr(0, sv.size() - 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer 'remove_suffix' over 'substr' for removing characters from the end [readability-stringview-substr]
  // CHECK-FIXES: sv.remove_suffix(3);

  // Case 2: redundant self-copy with size()
  sv = sv.substr(0, sv.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant self-copy [readability-stringview-substr]

  // Case 2: redundant self-copy with size() - 0
  sv = sv.substr(0, sv.size() - 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant self-copy [readability-stringview-substr]

  // Case 2: different vars, direct copy
  sv2 = sv.substr(0, sv.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer direct copy over substr [readability-stringview-substr]
  // CHECK-FIXES: sv2 = sv;
}

template <typename T>
void test_template_instantiation() {
  std::basic_string_view<T> sv("test");
  std::basic_string_view<T> sv2("test");

  // Should not match: inside a template instantiation
  sv = sv.substr(0, sv.size() - 3);      // No warning
  sv = sv.substr(0, sv.length());        // No warning
  sv2 = sv.substr(0, sv.size());         // No warning
}

// No matches when instantiated
void test_template_no_matches() {
  test_template_instantiation<char>();    // No warnings
  test_template_instantiation<wchar_t>(); // No warnings
}

void test_copies() {
  std::string_view sv("test");
  std::string_view sv1("test");
  std::string_view sv2("test");

  // Redundant self-copies
  sv = sv.substr(0, sv.length());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant self-copy [readability-stringview-substr]

  sv = sv.substr(0, sv.length() - 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant self-copy [readability-stringview-substr]

  // Direct copies between different variables
  sv1 = sv.substr(0, sv.length());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer direct copy over substr [readability-stringview-substr]
  // CHECK-FIXES: sv1 = sv;

  sv2 = sv.substr(0, sv.length() - 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer direct copy over substr [readability-stringview-substr]
  // CHECK-FIXES: sv2 = sv;
}

void test_zero_forms() {
  std::string_view sv("test");
  const int kZero = 0;
  #define START_POS 0

  // Various forms of zero in first argument
  sv = sv.substr(kZero, sv.length() - 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer 'remove_suffix' over 'substr' for removing characters from the end [readability-stringview-substr]
  // CHECK-FIXES: sv.remove_suffix(3);

  sv = sv.substr(START_POS, sv.length() - 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer 'remove_suffix' over 'substr' for removing characters from the end [readability-stringview-substr]
  // CHECK-FIXES: sv.remove_suffix(3);

  sv = sv.substr((0), sv.length() - 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer 'remove_suffix' over 'substr' for removing characters from the end [readability-stringview-substr]
  // CHECK-FIXES: sv.remove_suffix(3);

  sv = sv.substr(0u, sv.length() - 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer 'remove_suffix' over 'substr' for removing characters from the end [readability-stringview-substr]
  // CHECK-FIXES: sv.remove_suffix(3);

  sv = sv.substr(0UL, sv.length() - 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer 'remove_suffix' over 'substr' for removing characters from the end [readability-stringview-substr]
  // CHECK-FIXES: sv.remove_suffix(3);
}

void test_should_not_match() {
  std::string_view sv("test");
  std::string_view sv2("test");

  // No match: substr used for prefix or mid-view
  sv = sv.substr(1, sv.length() - 3); // No warning
}

void test_different_vars_remove_suffix() {
  std::string_view sv("test");
  std::string_view sv2("test");

  // Different string_views with remove_suffix pattern
  sv = sv2.substr(0, sv2.length() - 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer assignment and remove_suffix over substr [readability-stringview-substr]
  // CHECK-FIXES: sv = sv2;
  // CHECK-FIXES: sv.remove_suffix(3);
}

void f(std::string_view) {}
void test_expr_with_cleanups() {
  std::string_view sv("test");
  const auto copy = sv = sv.substr(0, sv.length() - 3); // No warning
  f(sv = sv.substr(0, sv.length() - 3)); // No warning
}
