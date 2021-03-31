// RUN: %check_clang_tidy -std=c++20 %s readability-container-size-empty %t -- -- -fno-delayed-template-parsing

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

template <typename T>
struct OpEqOnly {
  OpEqOnly();
  bool operator==(const OpEqOnly<T> &other) const;
  unsigned long size() const;
  bool empty() const;
};

template <typename T>
struct HasSpaceshipMem {
  HasSpaceshipMem();
  bool operator<=>(const HasSpaceshipMem<T> &other) const = default;
  unsigned long size() const;
  bool empty() const;
};

void returnsVoid() {
  OpEqOnly<int> OEO;
  HasSpaceshipMem<int> HSM;

  if (OEO != OpEqOnly<int>())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness
  // CHECK-FIXES: {{^  }}if (!OEO.empty()){{$}}
  // CHECK-MESSAGES: :19:8: note: method 'OpEqOnly'::empty() defined here
  if (HSM != HasSpaceshipMem<int>())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness
  // CHECK-FIXES: {{^  }}if (!HSM.empty()){{$}}
  // CHECK-MESSAGES: :27:8: note: method 'HasSpaceshipMem'::empty() defined here
}
