// RUN: %check_clang_tidy %s bugprone-unordered-equal-compare %t

namespace std {
template <class InputIt1, class InputIt2>
bool equal(InputIt1, InputIt1, InputIt2);
template <class InputIt1, class InputIt2>
bool equal(InputIt1, InputIt1, InputIt2, InputIt2);

template <class T> struct set {
  struct iterator {};
  iterator begin() const;
  iterator end() const;
};
template <class T> struct unordered_set {
  struct iterator {};
  iterator begin() const;
  iterator end() const;
  iterator cbegin() const;
  iterator cend() const;
};
template <class T> struct unordered_multiset {
  struct iterator {};
  iterator begin() const;
  iterator end() const;
};
template <class K, class V> struct unordered_map {
  struct iterator {};
  iterator begin() const;
  iterator end() const;
};
template <class K, class V> struct unordered_multimap {
  struct iterator {};
  iterator begin() const;
  iterator end() const;
};
} // namespace std

void bad_set(std::unordered_set<int> &a, std::unordered_set<int> &b) {
  std::equal(a.begin(), a.end(), b.begin());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing an unordered container with 'std::equal' is order-dependent [bugprone-unordered-equal-compare]
}

void bad_multiset(std::unordered_multiset<int> &a,
                  std::unordered_multiset<int> &b) {
  std::equal(a.begin(), a.end(), b.begin());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing an unordered container with 'std::equal' is order-dependent [bugprone-unordered-equal-compare]
}

void bad_map(std::unordered_map<int, int> &a, std::unordered_map<int, int> &b) {
  std::equal(a.begin(), a.end(), b.begin(), b.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing an unordered container with 'std::equal' is order-dependent [bugprone-unordered-equal-compare]
}

void bad_multimap(std::unordered_multimap<int, int> &a,
                  std::unordered_multimap<int, int> &b) {
  std::equal(a.begin(), a.end(), b.begin());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing an unordered container with 'std::equal' is order-dependent [bugprone-unordered-equal-compare]
}

// The const iterator accessors are matched too.
void bad_cbegin(std::unordered_set<int> &a, std::unordered_set<int> &b) {
  std::equal(a.cbegin(), a.cend(), b.cbegin());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing an unordered container with 'std::equal' is order-dependent [bugprone-unordered-equal-compare]
}

// Only one argument has to come from an unordered container.
void bad_one_side(std::unordered_set<int> &a, std::set<int>::iterator first,
                  std::set<int>::iterator last) {
  std::equal(first, last, a.begin());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing an unordered container with 'std::equal' is order-dependent [bugprone-unordered-equal-compare]
}

// A concrete call inside a template still fires, and the explicit instantiation
// must not report it a second time.
template <class T>
bool in_template(std::unordered_set<int> &a, std::unordered_set<int> &b) {
  return std::equal(a.begin(), a.end(), b.begin());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: comparing an unordered container with 'std::equal' is order-dependent [bugprone-unordered-equal-compare]
}
template bool in_template<int>(std::unordered_set<int> &,
                               std::unordered_set<int> &);

// A dependent container type isn't matched as written, so nothing is reported
// even once the template is instantiated with an unordered container.
template <class C>
bool dependent(C &a, C &b) {
  return std::equal(a.begin(), a.end(), b.begin());
}
void use_dependent(std::unordered_set<int> &a, std::unordered_set<int> &b) {
  dependent(a, b);
}

// The warning still fires through a macro, pointing at the expansion.
#define EQUAL_BEGIN(a, b) std::equal(a.begin(), a.end(), b.begin())
void via_macro(std::unordered_set<int> &a, std::unordered_set<int> &b) {
  EQUAL_BEGIN(a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: comparing an unordered container with 'std::equal' is order-dependent [bugprone-unordered-equal-compare]
}

void ok_ordered(std::set<int> &a, std::set<int> &b) {
  std::equal(a.begin(), a.end(), b.begin());
}
