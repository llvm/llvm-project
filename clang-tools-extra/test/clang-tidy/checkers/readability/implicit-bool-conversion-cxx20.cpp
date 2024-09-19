// RUN: %check_clang_tidy -std=c++20 %s readability-implicit-bool-conversion %t

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

namespace PR93409 {
  struct X
  {
      auto operator<=>(const X&) const = default;
      bool m_b;
  };

  struct Y
  {
      auto operator<=>(const Y&) const = default;
      X m_x;
  };
  
  bool compare(const Y& y1, const Y& y2)
  {
     return y1 == y2 || y1 < y2 || y1 > y2;
  }
}
