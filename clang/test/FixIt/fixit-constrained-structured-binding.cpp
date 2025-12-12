// RUN: not %clang_cc1 -std=c++20 -fdiagnostics-parseable-fixits -x c++ %s 2> %t
// RUN: FileCheck %s < %t

template<typename T>
concept UnaryC = true;
template<typename T, typename U>
concept BinaryC = true;

struct S{ int i, j; };
S get_S();

template<typename T>
T get_T();

void use() {
  UnaryC auto [a, b] = get_S();
  // CHECK: error: structured binding declaration cannot be declared with constrained 'auto'
  // CHECK: fix-it:{{.*}}:{16:3-16:10}:""
  BinaryC<int> auto [c, d] = get_S();
  // CHECK: error: structured binding declaration cannot be declared with constrained 'auto'
  // CHECK: fix-it:{{.*}}:{19:3-19:16}:""
}

template<typename T>
void TemplUse() {
  UnaryC auto [a, b] = get_T<T>();
  // CHECK: error: structured binding declaration cannot be declared with constrained 'auto'
  // XCHECK: fix-it:{{.*}}:{26:3-26:10}:""
  BinaryC<T> auto [c, d] = get_T<T>();
  // CHECK: error: structured binding declaration cannot be declared with constrained 'auto'
  // XCHECK: fix-it:{{.*}}:{29:3-29:14}:""
}

