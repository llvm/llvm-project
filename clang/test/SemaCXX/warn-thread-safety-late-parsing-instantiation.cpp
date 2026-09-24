// Under -fexperimental-late-parse-attributes, an attribute on a parameter may
// name a parameter declared later in the same prototype. In a template, it is
// instantiated once every parameter is, so it names the instantiated one.
//
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify -Wthread-safety -std=c++20 %s

#define RELEASE(...) __attribute__((release_capability(__VA_ARGS__)))

class __attribute__((capability("mutex"))) Mutex {};

// Each callback releases a lock that is not held, which is reported only if
// its attribute was instantiated naming the later parameter.
template <typename T>
void put_later(void (*release)(T) RELEASE(mu), Mutex *mu, T t) {
  release(t); // expected-warning{{releasing mutex 'mu' that was not held}}
}
template void put_later<int>(void (*)(int), Mutex *, int); // expected-note{{in instantiation of function template specialization 'put_later<int>' requested here}}

// After a parameter pack.
template <typename... Ts>
void put_later_pack(void (*release)(Ts...) RELEASE(mu), Mutex *mu, Ts... ts) {
  release(ts...); // expected-warning{{releasing mutex 'mu' that was not held}}
}
template void put_later_pack<int, long>(void (*)(int, long), Mutex *, int, long); // expected-note{{in instantiation of function template specialization 'put_later_pack<int, long>' requested here}}

template <typename T>
struct Methods {
  void dependent(void (*release)(T) RELEASE(mu), Mutex *mu, T t) {
    release(t); // expected-warning{{releasing mutex 'mu' that was not held}}
  }
  // A prototype that does not depend on the template.
  void independent(void (*release)(int) RELEASE(mu), Mutex *mu) {
    release(0); // expected-warning{{releasing mutex 'mu' that was not held}}
  }
  template <typename U>
  void member_template(void (*release)(U) RELEASE(mu), Mutex *mu, U u) {
    release(u); // expected-warning{{releasing mutex 'mu' that was not held}}
  }
};
template struct Methods<int>; // expected-note{{in instantiation of member function 'Methods<int>::dependent' requested here}} \
                              // expected-note{{in instantiation of member function 'Methods<int>::independent' requested here}}
template void Methods<int>::member_template<long>(void (*)(long), Mutex *, long); // expected-note{{in instantiation of function template specialization 'Methods<int>::member_template<long>' requested here}}

auto generic_lambda = [](auto x, void (*release)(int) RELEASE(mu), Mutex *mu) {
  release(x); // expected-warning{{releasing mutex 'mu' that was not held}}
};
void use_generic_lambda(void (*release)(int), Mutex *mu) {
  generic_lambda(0, release, mu); // expected-note{{in instantiation of function template specialization}}
}
