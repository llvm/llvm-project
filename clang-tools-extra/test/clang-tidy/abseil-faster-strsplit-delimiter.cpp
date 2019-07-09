// RUN: %check_clang_tidy -std=c++11-or-later %s abseil-faster-strsplit-delimiter %t
// FIXME: Fix the checker to work in C++17 mode.

namespace absl {

class string_view {
  public:
    string_view();
    string_view(const char *);
};

namespace strings_internal {
struct Splitter {};
struct MaxSplitsImpl {
  MaxSplitsImpl();
  ~MaxSplitsImpl();
};
} //namespace strings_internal

template <typename Delim>
strings_internal::Splitter StrSplit(absl::string_view, Delim) {
  return {};
}
template <typename Delim, typename Pred>
strings_internal::Splitter StrSplit(absl::string_view, Delim, Pred) {
  return {};
}

class ByAnyChar {
  public:
    explicit ByAnyChar(absl::string_view);
    ~ByAnyChar();
};

template <typename Delim>
strings_internal::MaxSplitsImpl MaxSplits(Delim, int) {
  return {};
}

} //namespace absl

void SplitDelimiters() {
  absl::StrSplit("ABC", "A");
  // CHECK-MESSAGES: [[@LINE-1]]:25: warning: absl::StrSplit() called with a string literal consisting of a single character; consider using the character overload [abseil-faster-strsplit-delimiter]
  // CHECK-FIXES: absl::StrSplit("ABC", 'A');

  absl::StrSplit("ABC", "\x01");
  // CHECK-MESSAGES: [[@LINE-1]]:25: warning: absl::StrSplit() called with a string literal consisting of a single character; consider using the character overload [abseil-faster-strsplit-delimiter]
  // CHECK-FIXES: absl::StrSplit("ABC", '\x01');

  absl::StrSplit("ABC", "\001");
  // CHECK-MESSAGES: [[@LINE-1]]:25: warning: absl::StrSplit() called with a string literal consisting of a single character; consider using the character overload [abseil-faster-strsplit-delimiter]
  // CHECK-FIXES: absl::StrSplit("ABC", '\001');

  absl::StrSplit("ABC", R"(A)");
  // CHECK-MESSAGES: [[@LINE-1]]:25: warning: absl::StrSplit() called with a string literal consisting of a single character; consider using the character overload [abseil-faster-strsplit-delimiter]
  // CHECK-FIXES: absl::StrSplit("ABC", 'A');

  absl::StrSplit("ABC", R"(')");
  // CHECK-MESSAGES: [[@LINE-1]]:25: warning: absl::StrSplit() called with a string literal consisting of a single character; consider using the character overload [abseil-faster-strsplit-delimiter]
  // CHECK-FIXES: absl::StrSplit("ABC", '\'');

  absl::StrSplit("ABC", R"(
)");
  // CHECK-MESSAGES: [[@LINE-2]]:25: warning: absl::StrSplit() called with a string literal consisting of a single character; consider using the character overload [abseil-faster-strsplit-delimiter]
  // CHECK-FIXES: absl::StrSplit("ABC", '\n');

  absl::StrSplit("ABC", R"delimiter(A)delimiter");
  // CHECK-MESSAGES: [[@LINE-1]]:25: warning: absl::StrSplit() called with a string literal consisting of a single character; consider using the character overload [abseil-faster-strsplit-delimiter]
  // CHECK-FIXES: absl::StrSplit("ABC", 'A');

  absl::StrSplit("ABC", absl::ByAnyChar("\n"));
  // CHECK-MESSAGES: [[@LINE-1]]:41: warning: absl::StrSplit()
  // CHECK-FIXES: absl::StrSplit("ABC", '\n');

  // Works with predicate
  absl::StrSplit("ABC", "A", [](absl::string_view) { return true; });
  // CHECK-MESSAGES: [[@LINE-1]]:25: warning: absl::StrSplit()
  // CHECK-FIXES: absl::StrSplit("ABC", 'A', [](absl::string_view) { return true; });

  // Doesn't do anything with other strings lenghts.
  absl::StrSplit("ABC", "AB");
  absl::StrSplit("ABC", absl::ByAnyChar(""));
  absl::StrSplit("ABC", absl::ByAnyChar(" \t"));

  // Escapes a single quote in the resulting character literal.
  absl::StrSplit("ABC", "'");
  // CHECK-MESSAGES: [[@LINE-1]]:25: warning: absl::StrSplit()
  // CHECK-FIXES: absl::StrSplit("ABC", '\'');

  absl::StrSplit("ABC", "\"");
  // CHECK-MESSAGES: [[@LINE-1]]:25: warning: absl::StrSplit()
  // CHECK-FIXES: absl::StrSplit("ABC", '\"');

  absl::StrSplit("ABC", absl::MaxSplits("\t", 1));
  // CHECK-MESSAGES: [[@LINE-1]]:41: warning: absl::MaxSplits()
  // CHECK-FIXES: absl::StrSplit("ABC", absl::MaxSplits('\t', 1));

  auto delim = absl::MaxSplits(absl::ByAnyChar(" "), 1);
  // CHECK-MESSAGES: [[@LINE-1]]:48: warning: absl::MaxSplits()
  // CHECK-FIXES: auto delim = absl::MaxSplits(' ', 1);
}

#define MACRO(str) absl::StrSplit("ABC", str)

void Macro() {
  MACRO("A");
}

template <typename T>
void FunctionTemplate() {
  // This one should not warn because ByAnyChar is a dependent type.
  absl::StrSplit("TTT", T("A"));

  // This one will warn, but we are checking that we get a correct warning only
  // once.
  absl::StrSplit("TTT", "A");
  // CHECK-MESSAGES: [[@LINE-1]]:25: warning: absl::StrSplit()
  // CHECK-FIXES: absl::StrSplit("TTT", 'A');
}

void FunctionTemplateCaller() {
  FunctionTemplate<absl::ByAnyChar>();
  FunctionTemplate<absl::string_view>();
}
