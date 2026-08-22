// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

// GH211926: constexpr range-based for loop variables used to be diagnosed
// with a note about the compiler-synthesized '__begin1' variable.

namespace std {
typedef decltype(sizeof(int)) size_t;
template <typename T> struct initializer_list {
  const T *p;
  size_t n;
  initializer_list(const T *p, size_t n);
  const T *begin() const; // expected-note {{selected 'begin' function with iterator type 'const int *'}}
  const T *end() const;
};
} // namespace std

void init_list() {
  for (constexpr auto x : {1, 2, 3}) { // expected-error {{constexpr variable 'x' must be initialized by a constant expression; a range-based for loop variable is initialized on each iteration from the loop's iterator, whose value is not known at compile time}}
  }
}

void c_array() {
  int arr[3] = {1, 2, 3};
  for (constexpr int x : arr) { // expected-error {{constexpr variable 'x' must be initialized by a constant expression; a range-based for loop variable is initialized on each iteration}}
  }
}

struct Vec {
  int data[3];
  const int *begin() const; // expected-note {{selected 'begin' function with iterator type 'const int *'}}
  const int *end() const;
};

void container() {
  Vec v = {{1, 2, 3}};
  for (constexpr int x : v) { // expected-error {{constexpr variable 'x' must be initialized by a constant expression; a range-based for loop variable is initialized on each iteration}}
  }
}

template <typename T> void dependent(T &range) {
  for (constexpr auto x : range) { // expected-error {{constexpr variable 'x' must be initialized by a constant expression; a range-based for loop variable is initialized on each iteration}}
  }
}

void instantiate() {
  int arr[3] = {1, 2, 3};
  dependent(arr); // expected-note {{in instantiation of function template specialization}}
}

void plain_loop_var() {
  for (auto x : {1, 2, 3})
    (void)x;
  int arr[3] = {1, 2, 3};
  for (int &x : arr)
    x = 0;
}

// Still valid when the initializer is a constant expression (CWG1204).
struct StatelessIter {
  struct It {
    int pos;
    It &operator++();
    bool operator!=(const It &other) const;
  };
  It begin();
  It end();
};
constexpr int operator*(const StatelessIter::It &) { return 7; }

void stateless_iterator() {
  for (constexpr int x : StatelessIter()) {
    static_assert(x == 7, "");
  }
}

void ordinary_constexpr() {
  constexpr int ok = 42;
  static_assert(ok == 42, "");

  int runtime = 0;             // expected-note {{declared here}}
  constexpr int bad = runtime; // expected-error {{constexpr variable 'bad' must be initialized by a constant expression}} expected-note {{read of non-const variable 'runtime' is not allowed in a constant expression}}

  for (constexpr int i = 0; i != 0;) {
  }
}
