// expected-no-diagnostics
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

namespace std {
  template<typename T>
  concept integral = __is_integral(T);

  template<typename T>
  concept floating_point = __is_floating_point(T);
}

// Constrained auto in abbreviated function template
void find(auto value) {
  std::integral auto var = value;
}

// Constrained auto at namespace scope (non-dependent context)
// Should be deduced immediately
std::integral auto globalVar = 42;

// Multiple constrained autos in template function
template<typename T>
void multipleConstrainedAutos(T value) {
  std::integral auto x = 10;
  std::floating_point auto y = 3.14;
  std::integral auto z = value; // dependent on T
}

// Constrained auto with qualifiers
void testQualifiers(auto value) {
  const std::integral auto cv1 = value;
  std::integral auto const cv2 = value;
}

// Nested constrained auto
void testNested(auto outer) {
  auto lambda = [](auto inner) {
    std::integral auto nested = inner;
    return nested;
  };

  std::integral auto result = lambda(outer);
}

// Constrained auto with references
void testReferences(auto value) {
  std::integral auto& ref = value;
  const std::integral auto& cref = value;
}

// Regular unconstrained auto (should not be affected by the fix)
void testUnconstrainedAuto(auto value) {
  auto regular = value;
  decltype(auto) decl_auto = (value);
}

// Constrained auto in class template member
template<typename T>
struct Container {
  void process(auto item) {
    std::integral auto local = item;
  }
};

// Constrained auto deduction from function call
std::integral auto getInteger() { return 42; }

void testFunctionReturn(auto param) {
  std::integral auto fromFunc = getInteger();
  std::integral auto fromParam = param;
}

// Ensure the fix doesn't break normal non-template constrained auto
void normalFunction() {
  std::integral auto x = 100;
  // This should be immediately deduced to int, not dependent
}

// Instantiate templates to verify no crashes
void instantiateAll() {
  find(42);
  multipleConstrainedAutos(5);
  testQualifiers(7);
  testNested(8);
  int val = 10;
  testReferences(val);
  testUnconstrainedAuto(11);
  Container<int> c;
  c.process(12);
  testFunctionReturn(13);
  normalFunction();
}
