// RUN: %check_clang_tidy %s performance-use-std-move %t -std=c++20,c++23
// RUN: %clang -std=c++20 -fsyntax-only -nostdinc++ -isystem %clang_tidy_headers/std %t.cpp

// CHECK-FIXES: #include <utility>

template <bool Enabled> struct Constrained {
  Constrained(const Constrained &);
  Constrained(Constrained &&) requires Enabled;
  Constrained &operator=(const Constrained &);
  Constrained &operator=(Constrained &&) requires Enabled;
};

// Counterfactual constraint satisfaction is outside the supported move search.
void enabledMove(Constrained<true> source) {
  Constrained<true> target(source);
}
void disabledMove(Constrained<false> source) {
  Constrained<false> target(source);
}
void enabledAssignment(Constrained<true> &target, Constrained<true> source) {
  target = source;
}
void disabledAssignment(Constrained<false> &target, Constrained<false> source) {
  target = source;
}

struct Ordinary {
  Ordinary(const Ordinary &);
  Ordinary(Ordinary &&);
};
void ordinaryMove(Ordinary source) {
  Ordinary target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: Ordinary target(std::move(source));
}

template <bool Explicit> struct ConditionalExplicit {
  ConditionalExplicit(const ConditionalExplicit &);
  explicit(Explicit) ConditionalExplicit(ConditionalExplicit &&);
};
void explicitDirectInitialization(ConditionalExplicit<true> source) {
  ConditionalExplicit<true> target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: ConditionalExplicit<true> target(std::move(source));
}
void explicitCopyInitialization(ConditionalExplicit<true> source) {
  ConditionalExplicit<true> target = source;
}
void nonexplicitCopyInitialization(ConditionalExplicit<false> source) {
  ConditionalExplicit<false> target = source;
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: ConditionalExplicit<false> target = std::move(source);
}
