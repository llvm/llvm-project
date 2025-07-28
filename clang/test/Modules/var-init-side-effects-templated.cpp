// Tests referencing variable with initializer containing side effect across module boundary

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %s -o %t

export module Foo;

export template <class Float>
struct Wrapper {
  double value;
};

export constexpr Wrapper<double> Compute() {
  return Wrapper<double>{1.0};
}

export template <typename Float>
Wrapper<Float> ComputeInFloat() {
  const Wrapper<Float> a = Compute();
  return a;
}
