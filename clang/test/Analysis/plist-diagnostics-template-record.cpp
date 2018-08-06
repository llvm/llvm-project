// RUN: %clang_analyze_cc1 -analyzer-output=plist -o %t.plist -std=c++11 -analyzer-checker=core %s
// RUN: FileCheck --input-file=%t.plist %s

bool ret();

template <class A, class B, class C, int N>
struct DivByZero {
  int i;
  DivByZero(bool b) {
    if (ret())
      i = 50 / (b - 1);
  }
};

template <class B, class C, int N>
struct DivByZero<char, B, C, N> {
  int i;
  DivByZero(bool b) {
    if (ret())
      i = 50 / (b - 1);
  }
};

template <typename... Args>
struct DivByZeroVariadic {
  int i;
  DivByZeroVariadic(bool b) {
    if (ret())
      i = 50 / (b - 1);
  }
};

int main() {
  DivByZero<int, float, double, 0> a(1);
  DivByZero<char, float, double, 0> a2(1);
  DivByZeroVariadic<char, float, double, decltype(nullptr)> a3(1);
}

// CHECK:      <string>Calling constructor for &apos;DivByZero&lt;int, float, double, 0&gt;&apos;</string>
// CHECK:      <string>Calling constructor for &apos;DivByZero&lt;char, float, double, 0&gt;&apos;</string>
// CHECK:      <string>Calling constructor for &apos;DivByZeroVariadic&lt;char, float, double, nullptr_t&gt;&apos;</string>

