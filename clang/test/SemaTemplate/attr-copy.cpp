// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

#if !__has_cpp_attribute(gnu::copy)
#error gnu::copy is not supported
#endif

[[gnu::nonnull(1)]] void source(int *);
[[gnu::copy(source)]] void copied(int *);
void use() {
  copied(nullptr); // expected-warning {{null passed to a callee that requires a non-null argument}}
}

void nothrow_source() __attribute__((nothrow));
void nothrow_copy() __attribute__((copy(nothrow_source)));
static_assert(noexcept(nothrow_copy()), "copied nothrow");

namespace overloads {
void source(int *) __attribute__((nonnull(1)));
void source(double *);
void copied(int *) __attribute__((copy(static_cast<void (*)(int *)>(source))));
void use() {
  copied(nullptr); // expected-warning {{null passed to a callee that requires a non-null argument}}
}
}

template <typename T> struct Source {
  static int value __attribute__((aligned(32)));
  static void function(int *) __attribute__((nonnull(1)));
};

template <typename T> struct Destination {
  int value __attribute__((copy(Source<T>::value)));
  static void function(int *) __attribute__((copy(Source<T>::function)));
};

static_assert(alignof(Destination<int>) == 32, "dependent variable");
void use_template() {
  Destination<int>::function(nullptr); // expected-warning {{null passed to a callee that requires a non-null argument}}
}

struct __attribute__((aligned(64))) Aligned {};
template <typename T> struct __attribute__((copy((T *)0))) TypeCopy {};
static_assert(alignof(TypeCopy<Aligned>) == 64, "dependent type");

template <typename T> struct __attribute__((aligned(sizeof(T) * 32))) AlignedTemplate {};
struct __attribute__((copy((AlignedTemplate<int> *)0))) SpecializationCopy {};
static_assert(alignof(SpecializationCopy) == sizeof(int) * 32, "source specialization");

template <typename T> struct Bad {
  static void function() __attribute__((copy(Source<T>::function))); // expected-error {{'nonnull' attribute parameter 1 is out of bounds}}
};
Bad<int> bad; // expected-note {{in instantiation of template class 'Bad<int>' requested here}}

template <void (*F)(int *)> struct FunctionCopy {
  static void function(int *) __attribute__((copy(F)));
};
void use_nttp() {
  FunctionCopy<source>::function(nullptr); // expected-warning {{null passed to a callee that requires a non-null argument}}
}
