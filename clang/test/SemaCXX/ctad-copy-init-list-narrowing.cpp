// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused-value -std=c++20 %s

namespace std {
typedef decltype(sizeof(int)) size_t;

template <typename E>
struct initializer_list {
  const E *p;
  size_t n;
  initializer_list(const E *p, size_t n) : p(p), n(n) {}
};

// Classes to use to reproduce the exact scenario present in #62925.
template<class T, class Y>
class pair {
    public:
    pair(T f, Y s) {}
};

template<class T, class Y>
class map {
    public:
    map(std::initializer_list<pair<T, Y>>) {}
    map(std::initializer_list<pair<T, Y>>, int a) {}
};

} // namespace std

// This is the almost the exact code that was in issue #62925.
void testOneLevelNesting() {
  std::map mOk = {std::pair{5, 'a'}, {6, 'b'}, {7, 'c'}};

  // Verify that narrowing conversion is disabled in the first level of nesting.
  std::map mNarrow = {std::pair{5, 'a'}, {6.0f, 'b'}, {7, 'c'}}; // expected-error {{type 'float' cannot be narrowed to 'int' in initializer list}} // expected-note {{insert an explicit cast to silence this issue}}
}

void testMultipleLevelNesting() {
  std::map aOk = {{std::pair{5, 'c'}, {5, 'c'}}, 5};

  // Verify that narrowing conversion is disabled when it is not in a nested
  // in another std::initializer_list, but it happens in the most outer one.
  std::map aNarrowNested = {{std::pair{5, 'c'}, {5.0f, 'c'}}, 5}; // expected-error {{type 'float' cannot be narrowed to 'int' in initializer list}} // expected-note {{insert an explicit cast to silence this issue}}

  // Verify that narrowing conversion is disabled in the first level of nesting.
  std::map aNarrow = {{std::pair{5, 'c'}, {5, 'c'}}, 5.0f}; // expected-error {{type 'float' cannot be narrowed to 'int' in initializer list}} // expected-note {{insert an explicit cast to silence this issue}}
}
