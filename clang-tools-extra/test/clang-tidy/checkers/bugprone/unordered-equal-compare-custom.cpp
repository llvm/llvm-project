// RUN: %check_clang_tidy %s bugprone-unordered-equal-compare %t -- \
// RUN:   -config="{CheckOptions: {bugprone-unordered-equal-compare.Containers: '::boost::unordered_set'}}"

namespace std {
template <class InputIt1, class InputIt2>
bool equal(InputIt1, InputIt1, InputIt2);

template <class T> struct unordered_set {
  struct iterator {};
  iterator begin() const;
  iterator end() const;
};
} // namespace std

namespace boost {
template <class T> struct unordered_set {
  struct iterator {};
  iterator begin() const;
  iterator end() const;
};
} // namespace boost

// The custom container from the 'Containers' option is flagged.
void bad_boost(boost::unordered_set<int> &a, boost::unordered_set<int> &b) {
  std::equal(a.begin(), a.end(), b.begin());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing an unordered container with 'std::equal' is order-dependent [bugprone-unordered-equal-compare]
}

// The default std container is not in the configured list, so it is ignored.
void ok_std(std::unordered_set<int> &a, std::unordered_set<int> &b) {
  std::equal(a.begin(), a.end(), b.begin());
}
