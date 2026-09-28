// RUN: %clang_analyze_cc1 -std=c++23 -analyzer-checker=alpha.webkit.UnborrowedLocalVarsChecker -verify %s

#include "mock-canborrow.h"

void someFunction();

void borrow_function_get_loop(Vector<char> &vec) {
  for (char &c : borrow(vec).get()) {
    someFunction();
    (void)c;
  }
}

struct ReverseAdaptor {};
struct ReversedChars : std::ranges::view_interface<ReversedChars> {
  explicit ReversedChars(Vector<char> &);
  struct Iterator {
    char &operator*() const;
    Iterator &operator++();
    bool operator!=(const Iterator &) const;
  };
  Iterator begin() const;
  Iterator end() const;
  ReversedChars zipWith(Vector<int> &) const;
};
inline constexpr ReverseAdaptor reversed{};
ReversedChars operator|(Vector<char> &, const ReverseAdaptor &);
ReversedChars operator|(ReversedChars &&, const ReverseAdaptor &);

void unguarded_global_adaptor_pipe_loop(Vector<char> &vec) {
  for (char &c : vec | reversed) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}

void borrowed_global_adaptor_pipe_loop(Vector<char> &vec) {
  for (char &c : borrow(vec).get() | reversed) {
    someFunction();
    (void)c;
  }
}

void chained_pipe_unguarded(Vector<char> &vec) {
  ReversedChars rv = vec | reversed | reversed;
  // expected-warning@-1{{Local variable 'rv' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)rv;
}

void chained_pipe_borrowed(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  ReversedChars rv = b.get() | reversed | reversed;
  someFunction();
  (void)rv;
}

void unguarded_pipe_loop(Vector<char> &vec) {
  for (char &c : vec | ReverseAdaptor()) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}

void borrow_get_pipe_loop(Vector<char> &vec) {
  for (char &c : borrow(vec).get() | ReverseAdaptor()) {
    someFunction();
    (void)c;
  }
}

void named_view(Vector<char> &vec) {
  ReversedChars rv = vec | ReverseAdaptor();
  // expected-warning@-1{{Local variable 'rv' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)rv;
}

void named_view_borrowed(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  ReversedChars rv = b.get() | ReverseAdaptor();
  someFunction();
  (void)rv;
}

void constructed_view(Vector<char> &vec) {
  ReversedChars rv(vec);
  // expected-warning@-1{{Local variable 'rv' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)rv;
}

struct PlainReversed : std::ranges::view_interface<PlainReversed> {
  explicit PlainReversed(Vector<char> &);
  std::reverse_iterator<char *> begin() const;
  std::reverse_iterator<char *> end() const;
};

void std_reverse_iterator_loop(Vector<char> &vec) {
  for (char &c : PlainReversed(vec)) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}

void std_reverse_iterator_borrowed(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  for (char &c : PlainReversed(b.get())) {
    someFunction();
    (void)c;
  }
}

void data_from_vector(Vector<char> &vec) {
  char *p = std::data(vec);
  // expected-warning@-1{{Local variable 'p' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)p;
}

void data_from_borrow(Vector<char> &vec) {
  Borrow<Vector<char>> b(vec);
  char *p = std::data(b.get());
  someFunction();
  (void)p;
}

void get_from_element(Vector<std::pair<int, int>> &vec) {
  auto &first = std::get<0>(vec[0]);
  // expected-warning@-1{{Local variable 'first' is a loan on CanBorrow type 'Vector<std::pair<int, int>>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)first;
}

void get_from_borrowed_element(Vector<std::pair<int, int>> &vec) {
  Borrow<Vector<std::pair<int, int>>> b(vec);
  auto &first = std::get<0>(b.get()[0]);
  someFunction();
  (void)first;
}

struct AnnotatedView : std::ranges::view_interface<AnnotatedView> {
  AnnotatedView(Vector<char> &tracked LIFETIME_BOUND, Vector<char> &untracked);
};

void trusted_annotations(Vector<char> &tracked, Vector<char> &untracked) {
  Borrow<Vector<char>> b(tracked);
  AnnotatedView v(b.get(), untracked);
  someFunction();
  (void)v;
}

void member_arg_unguarded(Vector<char> &vec, Vector<int> &ints) {
  Borrow<Vector<char>> b(vec);
  ReversedChars rv = (b.get() | reversed).zipWith(ints);
  // expected-warning@-1{{Local variable 'rv' is a loan on CanBorrow type 'Vector<int>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)rv;
}

void member_object_unguarded(Vector<char> &vec, Vector<int> &ints) {
  Borrow<Vector<int>> b(ints);
  ReversedChars rv = (vec | reversed).zipWith(b.get());
  // expected-warning@-1{{Local variable 'rv' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)rv;
}

void member_both_borrowed(Vector<char> &vec, Vector<int> &ints) {
  Borrow<Vector<char>> bc(vec);
  Borrow<Vector<int>> bi(ints);
  ReversedChars rv = (bc.get() | reversed).zipWith(bi.get());
  someFunction();
  (void)rv;
}

struct ZipView : std::ranges::view_interface<ZipView> {
  ZipView(Vector<char> &, Vector<int> &);
  struct Iterator {
    char &operator*() const;
    Iterator &operator++();
    bool operator!=(const Iterator &) const;
  };
  Iterator begin() const;
  Iterator end() const;
};
struct ZipAdaptor {
  ZipView operator()(Vector<char> &, Vector<int> &) const;
};
inline constexpr ZipAdaptor zip{};

void zip_unguarded(Vector<char> &vec, Vector<int> &ints) {
  for (char &c : zip(vec, ints)) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}

void zip_first_borrowed(Vector<char> &vec, Vector<int> &ints) {
  Borrow<Vector<char>> b(vec);
  for (char &c : zip(b.get(), ints)) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<int>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}

void zip_second_borrowed(Vector<char> &vec, Vector<int> &ints) {
  Borrow<Vector<int>> b(ints);
  for (char &c : zip(vec, b.get())) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}

void zip_both_borrowed(Vector<char> &vec, Vector<int> &ints) {
  Borrow<Vector<char>> bc(vec);
  Borrow<Vector<int>> bi(ints);
  for (char &c : zip(bc.get(), bi.get())) {
    someFunction();
    (void)c;
  }
}

void zip_constructed_second_borrowed(Vector<char> &vec, Vector<int> &ints) {
  Borrow<Vector<int>> b(ints);
  ZipView z(vec, b.get());
  // expected-warning@-1{{Local variable 'z' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
  someFunction();
  (void)z;
}

void zip_constructed_both_borrowed(Vector<char> &vec, Vector<int> &ints) {
  Borrow<Vector<char>> bc(vec);
  Borrow<Vector<int>> bi(ints);
  ZipView z(bc.get(), bi.get());
  someFunction();
  (void)z;
}

struct NonStdAdaptor {};
struct NonStdReversed {
  char *b;
  char *e;
  char *begin() const;
  char *end() const;
};
NonStdReversed operator|(Vector<char> &, NonStdAdaptor);

void non_std_pipe_loop(Vector<char> &vec) {
  for (char &c : vec | NonStdAdaptor()) {
    someFunction();
    (void)c;
  }
}

void reference_loop(Vector<char> &vec) {
  for (char &c : vec) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}
