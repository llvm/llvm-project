// RUN: %check_clang_tidy -std=c++20 %s modernize-loop-convert %t -- -- -I %S/Inputs/loop-convert

namespace std {
struct strong_ordering {
  int n;
  constexpr operator int() const { return n; }
  static const strong_ordering equal, greater, less;
};
constexpr strong_ordering strong_ordering::equal = {0};
constexpr strong_ordering strong_ordering::greater = {1};
constexpr strong_ordering strong_ordering::less = {-1};
} // namespace std

struct HasSpaceshipMem {
  typedef int value_type;

  struct iterator {
    value_type &operator*();
    const value_type &operator*() const;
    iterator &operator++();
    void insert(value_type);
    value_type X;
    constexpr auto operator<=>(const HasSpaceshipMem::iterator &) const = default;
  };

  iterator begin();
  iterator end();
};

struct OpEqOnly {
  typedef int value_type;
  struct iterator {
    value_type &operator*();
    const value_type &operator*() const;
    iterator &operator++();
    bool operator==(const iterator &other) const;
    void insert(value_type);
    value_type X;
  };
  iterator begin();
  iterator end();
};

void rewritten() {
  OpEqOnly Oeo;
  for (OpEqOnly::iterator It = Oeo.begin(), E = Oeo.end(); It != E; ++It) {
    (void)*It;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : Oeo)
  // CHECK-FIXES-NEXT: (void)It;

  HasSpaceshipMem Hsm;
  for (HasSpaceshipMem::iterator It = Hsm.begin(), E = Hsm.end(); It != E; ++It) {
    (void)*It;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : Hsm)
  // CHECK-FIXES-NEXT: (void)It;
}
