// RUN: %clang_cc1 -std=c++20 -verify %s
// expected-no-diagnostics

static constexpr int PRIMARY = 0;
static constexpr int SPECIALIZATION_CONCEPT = 1;
static constexpr int SPECIALIZATION_REQUIRES = 2;

template <class T>
concept Concept = (sizeof(T) >= 2 * sizeof(int));

struct XY {
  int x;
  int y;
};

namespace members {

template <class T, class U> struct S {
  static constexpr int primary();
};

template <class T, class U> constexpr int S<T, U>::primary() {
  return PRIMARY;
};

template <Concept C, class U> struct S<C, U> {
  static constexpr int specialization();
};

template <class T, class U>
  requires(sizeof(T) == sizeof(int))
struct S<T, U> {
  static constexpr int specialization();
};

template <Concept C, class U> constexpr int S<C, U>::specialization() {
  return SPECIALIZATION_CONCEPT;
}

template <class T, class U>
  requires(sizeof(T) == sizeof(int))
constexpr int S<T, U>::specialization() {
  return SPECIALIZATION_REQUIRES;
}

static_assert(S<char, double>::primary() == PRIMARY);
static_assert(S<XY, double>::specialization() == SPECIALIZATION_CONCEPT);
static_assert(S<int, double>::specialization() == SPECIALIZATION_REQUIRES);

} // namespace members

namespace enumerations {

template <class T, class U> struct S {
  enum class E : int;
};

template <class T, class U> enum class S<T, U>::E { Value = PRIMARY };

template <Concept C, class U> struct S<C, U> {
  enum class E : int;
};

template <Concept C, class U>
enum class S<C, U>::E {
  Value = SPECIALIZATION_CONCEPT
};

template <class T, class U>
  requires(sizeof(T) == sizeof(int))
struct S<T, U> {
  enum class E : int;
};

template <class T, class U>
  requires(sizeof(T) == sizeof(int))
enum class S<T, U>::E {
  Value = SPECIALIZATION_REQUIRES
};

static_assert(static_cast<int>(S<char, double>::E::Value) == PRIMARY);
static_assert(static_cast<int>(S<XY, double>::E::Value) ==
              SPECIALIZATION_CONCEPT);
static_assert(static_cast<int>(S<int, double>::E::Value) ==
              SPECIALIZATION_REQUIRES);

} // namespace  enumerations

namespace multiple_template_parameter_lists {

template <class Outer>
struct S {
  template <class Inner>
  static constexpr int primary(Inner);
};

template <class Outer>
template <class Inner>
constexpr int S<Outer>::primary(Inner) {
  return PRIMARY;
};

template <Concept Outer>
struct S<Outer> {
  template <class Inner>
  static constexpr int specialization(Inner);
};

template <Concept Outer>
template <class Inner>
constexpr int S<Outer>::specialization(Inner) { return SPECIALIZATION_CONCEPT; }

template <class Outer>
  requires(sizeof(Outer) == sizeof(int))
struct S<Outer> {
  template <class Inner>
  static constexpr int specialization(Inner);
};

template <class Outer>
  requires(sizeof(Outer) == sizeof(int))
template <class Inner>
constexpr int S<Outer>::specialization(Inner) { return SPECIALIZATION_REQUIRES; }

static_assert(S<char>::primary("str") == PRIMARY);
static_assert(S<XY>::specialization("str") == SPECIALIZATION_CONCEPT);
static_assert(S<int>::specialization("str") == SPECIALIZATION_REQUIRES);

} // namespace multiple_template_parameter_lists
