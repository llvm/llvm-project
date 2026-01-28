// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s
// expected-no-diagnostics

void clang_analyzer_warnIfReached();

// Forward declaration
namespace N { // 1st
template <class T> struct Wrapper;
} // namespace N

// This becomes the lexical parent for implicit specializations
namespace N { // 2nd
template <class T> struct Wrapper; // DeclWithIssue changes to this 1st.
template <class T> void trigger() {
  Wrapper<T>::get();
}
} // namespace N

// [[clang::suppress]] is here, at the primary template
namespace N { // 3rd
template <class T> struct Wrapper {
  static void get() {
    // This [[clang::suppress]] should suppress the warning.
    // Bug: Without the fix, it doesn't work because the implicit
    // specialization's lexical parent points to second namespace block.
    [[clang::suppress]] clang_analyzer_warnIfReached(); // no-warning
  }
};
} // namespace N

void rdar_168941095() {
  N::trigger<int>();
}
