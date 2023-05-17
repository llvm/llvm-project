// RUN: %check_clang_tidy %s bugprone-fold-init-type -std=c++17 %t

namespace std {
template <class InputIt, class T>
T accumulate(InputIt first, InputIt last, T init) {
  // When `InputIt::operator*` returns a deduced `auto` type that refers to a
  // dependent type, the return type is deduced only if `InputIt::operator*`
  // is instantiated. In practice this happens somewhere in the implementation
  // of `accumulate`. For tests, do it here.
  (void)*first;
}

template <class InputIt, class T>
T reduce(InputIt first, InputIt last, T init) { (void)*first; }
template <class ExecutionPolicy, class InputIt, class T>
T reduce(ExecutionPolicy &&policy,
         InputIt first, InputIt last, T init) { (void)*first; }

struct parallel_execution_policy {};
constexpr parallel_execution_policy par{};

template <class InputIt1, class InputIt2, class T>
T inner_product(InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T value) { (void)*first1; (void)*first2; }

template <class ExecutionPolicy, class InputIt1, class InputIt2, class T>
T inner_product(ExecutionPolicy &&policy, InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T value) { (void)*first1; (void)*first2; }

} // namespace std

struct FloatIterator {
  const float &operator*() const;
};

struct DerivedFloatIterator : public FloatIterator {
};

template <typename ValueType> struct ByValueTemplateIterator {
  ValueType operator*() const;
};

template <typename ValueType> struct ByRefTemplateIterator {
  ValueType &operator*();
};

template <typename ValueType> struct ByRefTemplateIteratorWithAlias {
  using reference = const ValueType&;
  reference operator*();
};

template <typename ValueType> struct AutoByValueTemplateIterator {
  auto operator*() const { return ValueType{}; }
};

template <typename ValueType> struct AutoByRefTemplateIterator {
  decltype(auto) operator*() const { return value_; }
  ValueType value_;
};

template <typename ValueType>
struct InheritingByConstRefTemplateIterator
    : public ByRefTemplateIterator<const ValueType> {};

using TypedeffedIterator = FloatIterator;

// Positives.

int accumulatePositive1() {
  float a[1] = {0.5f};
  return std::accumulate(a, a + 1, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int accumulatePositive2() {
  FloatIterator it;
  return std::accumulate(it, it, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int accumulatePositive3() {
  DerivedFloatIterator it;
  return std::accumulate(it, it, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int accumulatePositive4() {
  double a[1] = {0.0};
  return std::accumulate(a, a + 1, 0.0f);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'double' into type 'float'
}

int accumulatePositive5() {
  ByValueTemplateIterator<unsigned> it;
  return std::accumulate(it, it, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'unsigned int' into type 'int'
}

int accumulatePositive6() {
  ByRefTemplateIterator<unsigned> it;
  return std::accumulate(it, it, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'unsigned int' into type 'int'
}

int accumulatePositive7() {
  AutoByValueTemplateIterator<unsigned> it;
  return std::accumulate(it, it, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'unsigned int' into type 'int'
}

int accumulatePositive8() {
  TypedeffedIterator it;
  return std::accumulate(it, it, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int accumulatePositive9() {
  InheritingByConstRefTemplateIterator<unsigned> it;
  return std::accumulate(it, it, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'unsigned int' into type 'int'
}

int accumulatePositive10() {
  AutoByRefTemplateIterator<unsigned> it;
  return std::accumulate(it, it, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'unsigned int' into type 'int'
}

int accumulatePositive11() {
  ByRefTemplateIteratorWithAlias<unsigned> it;
  return std::accumulate(it, it, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'unsigned int' into type 'int'
}

int reducePositive1() {
  float a[1] = {0.5f};
  return std::reduce(a, a + 1, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int reducePositive2() {
  float a[1] = {0.5f};
  return std::reduce(std::par, a, a + 1, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int innerProductPositive1() {
  float a[1] = {0.5f};
  int b[1] = {1};
  return std::inner_product(std::par, a, a + 1, b, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int innerProductPositive2() {
  float a[1] = {0.5f};
  int b[1] = {1};
  return std::inner_product(std::par, a, a + 1, b, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

// Negatives.

int negative1() {
  float a[1] = {0.5f};
  // This is OK because types match.
  return std::accumulate(a, a + 1, 0.0);
}

int negative2() {
  float a[1] = {0.5f};
  // This is OK because double is bigger than float.
  return std::accumulate(a, a + 1, 0.0);
}

int negative3() {
  float a[1] = {0.5f};
  // This is OK because the user explicitly specified T.
  return std::accumulate<float *, float>(a, a + 1, 0);
}

int negative4() {
  ByValueTemplateIterator<unsigned> it;
  // For now this is OK.
  return std::accumulate(it, it, 0.0);
}

int negative5() {
  float a[1] = {0.5f};
  float b[1] = {1.0f};
  return std::inner_product(std::par, a, a + 1, b, 0.0f);
}

namespace blah {
namespace std {
template <class InputIt, class T>
T accumulate(InputIt, InputIt, T); // We should not care about this one.
}

int negative5() {
  float a[1] = {0.5f};
  // Note that this is using blah::std::accumulate.
  return std::accumulate(a, a + 1, 0);
}
}
