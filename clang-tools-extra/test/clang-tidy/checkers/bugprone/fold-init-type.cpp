// RUN: %check_clang_tidy %s bugprone-fold-init-type -std=c++17 %t

namespace std {
template <class InputIt, class T>
T accumulate(InputIt first, InputIt last, T init) {
  // When `InputIt::operator*` returns a deduced `auto` type that refers to a
  // dependent type, the return type is deduced only if `InputIt::operator*`
  // is instantiated. In practice this happens somewhere in the implementation
  // of `accumulate`. For tests, do it here.
  (void)*first;
  return init;
}
template <class InputIt, class T, class BinaryOp>
T accumulate(InputIt first, InputIt last, T init, BinaryOp op) {
  (void)*first;
  return init;
}

template <class InputIt, class T>
T reduce(InputIt first, InputIt last, T init) { (void)*first; return init; }
template <class InputIt, class T, class BinaryOp>
T reduce(InputIt first, InputIt last, T init, BinaryOp op) { (void)*first; return init; }
template <class ExecutionPolicy, class InputIt, class T>
T reduce(ExecutionPolicy &&policy,
         InputIt first, InputIt last, T init) { (void)*first; return init; }
template <class ExecutionPolicy, class InputIt, class T, class BinaryOp>
T reduce(ExecutionPolicy &&policy,
         InputIt first, InputIt last, T init, BinaryOp op) { (void)*first; return init; }

struct parallel_execution_policy {};
constexpr parallel_execution_policy par{};

template <class InputIt1, class InputIt2, class T>
T inner_product(InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T value) { (void)*first1; (void)*first2; return value;  }
template <class InputIt1, class InputIt2, class T, class BinaryOp1, class BinaryOp2>
T inner_product(InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T value, BinaryOp1 op1, BinaryOp2 op2) { (void)*first1; (void)*first2; return value; }

template <class ExecutionPolicy, class InputIt1, class InputIt2, class T>
T inner_product(ExecutionPolicy &&policy, InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T value) { (void)*first1; (void)*first2; return value; }
template <class ExecutionPolicy, class InputIt1, class InputIt2, class T, class BinaryOp1, class BinaryOp2>
T inner_product(ExecutionPolicy &&policy, InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T value, BinaryOp1 op1, BinaryOp2 op2) { (void)*first1; (void)*first2; return value; }

template <class T = void> struct plus {
  T operator()(T a, T b) const { return a + b; }
};
template <> struct plus<void> {
  template <class T, class U>
  auto operator()(T a, U b) const { return a + b; }
};

template <class T = void> struct minus {
  T operator()(T a, T b) const { return a - b; }
};
template <> struct minus<void> {
  template <class T, class U>
  auto operator()(T a, U b) const { return a - b; }
};

template <class T = void> struct multiplies {
  T operator()(T a, T b) const { return a * b; }
};
template <> struct multiplies<void> {
  template <class T, class U>
  auto operator()(T a, U b) const { return a * b; }
};

template <class T = void> struct bit_or {
  T operator()(T a, T b) const { return a | b; }
};
template <> struct bit_or<void> {
  template <class T, class U>
  auto operator()(T a, U b) const { return a | b; }
};

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

int accumulatePositiveOp1() {
  float a[1] = {0.5f};
  return std::accumulate(a, a + 1, 0, std::plus<>());
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int accumulatePositiveOp2() {
  float a[1] = {0.5f};
  return std::accumulate(a, a + 1, 0, std::minus<>());
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int accumulatePositiveOp3() {
  float a[1] = {0.5f};
  return std::accumulate(a, a + 1, 0, std::multiplies<>());
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int reducePositiveOp1() {
  float a[1] = {0.5f};
  return std::reduce(a, a + 1, 0, std::plus<>());
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int reducePositiveOp2() {
  float a[1] = {0.5f};
  return std::reduce(std::par, a, a + 1, 0, std::plus<>());
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int innerProductPositiveOp1() {
  float a[1] = {0.5f};
  int b[1] = {1};
  return std::inner_product(a, a + 1, b, 0, std::plus<>(), std::multiplies<>());
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int innerProductPositiveOp2() {
  float a[1] = {0.5f};
  int b[1] = {1};
  return std::inner_product(std::par, a, a + 1, b, 0, std::plus<>(), std::multiplies<>());
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int accumulatePositiveBitOr() {
  unsigned a[1] = {1};
  return std::accumulate(a, a + 1, 0, std::bit_or<>());
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'unsigned int' into type 'int'
}

int accumulatePositiveExplicitFunctor1() {
  float a[1] = {0.5f};
  return std::accumulate(a, a + 1, 0, std::plus<int>());
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int accumulatePositiveExplicitFunctor2() {
  float a[1] = {0.5f};
  return std::accumulate(a, a + 1, 0, std::plus<float>());
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int accumulatePositiveExplicitFunctor3() {
  float a[1] = {0.5f};
  return std::accumulate(a, a + 1, 0, std::multiplies<double>());
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int accumulatePositiveExplicitFunctor4() {
  double a[1] = {0.5};
  return std::accumulate(a, a + 1, 0.0f, std::plus<double>());
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'double' into type 'float'
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

int negativeOp1() {
  float a[1] = {0.5f};
  // This is OK because types match.
  return std::accumulate(a, a + 1, 0.0f, std::plus<>());
}

int negativeOp2() {
  float a[1] = {0.5f};
  // This is OK because double is bigger than float.
  return std::reduce(a, a + 1, 0.0, std::plus<>());
}

int negativeOp3() {
  float a[1] = {0.5f};
  // This is OK because types are compatible even with explicit functor.
  return std::accumulate(a, a + 1, 0.0, std::plus<double>());
}

int negativeLambda1() {
  float a[1] = {0.5f};
  // This is currently a known limitation.
  return std::accumulate(a, a + 1, 0, [](int acc, float val) {
    return acc + (val > 0.0f ? 1 : 0);
  });
}

int negativeLambda2() {
  float a[1] = {0.5f};
  // This is currently a known limitation.
  return std::reduce(a, a + 1, 0, [](int acc, float val) {
    return acc + static_cast<int>(val);
  });
}

int negativeInnerProductMixedOps() {
  float a[1] = {0.5f};
  int b[1] = {1};
  // Only one op is transparent, the other is a lambda.
  return std::inner_product(a, a + 1, b, 0, std::plus<>(),
                            [](float x, int y) { return x * y; });
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
