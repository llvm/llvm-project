// RUN: %check_clang_tidy %s bugprone-fold-init-type %t

namespace std {
template <class InputIt, class T>
T accumulate(InputIt first, InputIt last, T init);

template <class InputIt, class T>
T reduce(InputIt first, InputIt last, T init);
template <class ExecutionPolicy, class InputIt, class T>
T reduce(ExecutionPolicy &&policy,
         InputIt first, InputIt last, T init);

struct parallel_execution_policy {};
constexpr parallel_execution_policy par{};

template <class InputIt1, class InputIt2, class T>
T inner_product(InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T value);

template <class ExecutionPolicy, class InputIt1, class InputIt2, class T>
T inner_product(ExecutionPolicy &&policy, InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T value);

} // namespace std

struct FloatIterator {
  typedef float value_type;
};
template <typename ValueType>
struct TypedefTemplateIterator { typedef ValueType value_type; };
template <typename ValueType>
struct UsingTemplateIterator { using value_type = ValueType; };
template <typename ValueType>
struct DependentTypedefTemplateIterator { typedef typename ValueType::value_type value_type; };
template <typename ValueType>
struct DependentUsingTemplateIterator : public TypedefTemplateIterator<ValueType> { using typename TypedefTemplateIterator<ValueType>::value_type; };
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
  double a[1] = {0.0};
  return std::accumulate(a, a + 1, 0.0f);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'double' into type 'float'
}

int accumulatePositive4() {
  TypedefTemplateIterator<unsigned> it;
  return std::accumulate(it, it, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'unsigned int' into type 'int'
}

int accumulatePositive5() {
  UsingTemplateIterator<unsigned> it;
  return std::accumulate(it, it, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'unsigned int' into type 'int'
}

int accumulatePositive6() {
  DependentTypedefTemplateIterator<UsingTemplateIterator<unsigned>> it;
  return std::accumulate(it, it, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'unsigned int' into type 'int'
}

int accumulatePositive7() {
  TypedeffedIterator it;
  return std::accumulate(it, it, 0);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: folding type 'float' into type 'int'
}

int accumulatePositive8() {
  DependentUsingTemplateIterator<unsigned> it;
  return std::accumulate(it, it, 0);
  // FIXME: this one should trigger too.
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
  TypedefTemplateIterator<unsigned> it;
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
