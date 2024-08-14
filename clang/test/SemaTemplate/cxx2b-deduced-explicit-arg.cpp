// RUN: %clang_cc1 -fsyntax-only -std=c++2b -verify %s

namespace std {
struct rv {};

template <bool B, typename T> struct enable_if;
template <typename T> struct enable_if<true, T> { typedef T type; };

template <typename U, typename T>
typename enable_if<__is_convertible(T, rv), U>::type forward(T &);
template <typename U, typename T>
typename enable_if<!__is_convertible(T, rv), U &>::type forward(T &);
}

struct Foo {
  template <class T>
  constexpr auto operator[](this T &&self, auto... i)        // expected-note {{candidate template ignored: substitution failure [with T = Foo &, i:auto = <>]: member '_evaluate' used before its declaration}}
      -> decltype(_evaluate(std::forward<T>(self), i...)) {
    return self._evaluate(i...);
  }

private:
  template <class T>
  constexpr auto _evaluate(this T &&self, auto... i) -> decltype((i + ...));
};

int main() {
  Foo foo;
  return foo[]; // expected-error {{no viable overloaded operator[] for type 'Foo'}}
}
