// RUN: %check_clang_tidy %s cppcoreguidelines-pro-bounds-constant-array-index %t

typedef __SIZE_TYPE__ size_t;

namespace std {
  template<typename T, size_t N>
  struct array {
    T& operator[](size_t n);
    T& at(size_t n);
  };
}


namespace gsl {
  template<class T, size_t N>
  T& at( T(&a)[N], size_t index );

  template<class T, size_t N>
  T& at( std::array<T, N> &a, size_t index );
}

constexpr int const_index(int base) {
  return base + 3;
}

void f(std::array<int, 10> a, int pos) {
  a [ pos / 2 /*comment*/] = 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use array subscript when the index is not an integer constant expression [cppcoreguidelines-pro-bounds-constant-array-index]
  int j = a[pos - 1];
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: do not use array subscript when the index is not an integer constant expression

  a.at(pos-1) = 2; // OK, at() instead of []
  gsl::at(a, pos-1) = 2; // OK, gsl::at() instead of []

  a[-1] = 3;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: std::array<> index -1 is negative [cppcoreguidelines-pro-bounds-constant-array-index]
  a[10] = 4;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: std::array<> index 10 is past the end of the array (which contains 10 elements) [cppcoreguidelines-pro-bounds-constant-array-index]

  a[const_index(7)] = 3;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: std::array<> index 10 is past the end of the array (which contains 10 elements)

  a[0] = 3; // OK, constant index and inside bounds
  a[1] = 3; // OK, constant index and inside bounds
  a[9] = 3; // OK, constant index and inside bounds
  a[const_index(6)] = 3; // OK, constant index and inside bounds

  using MyArray = std::array<int, 10>;
  MyArray m{};
  m [ pos / 2 /*comment*/] = 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use array subscript when the index is not an integer constant expression [cppcoreguidelines-pro-bounds-constant-array-index]
  int jj = m[pos - 1];
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not use array subscript when the index is not an integer constant expression

  m.at(pos-1) = 2; // OK, at() instead of []
  gsl::at(m, pos-1) = 2; // OK, gsl::at() instead of []
  m[-1] = 3;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: std::array<> index -1 is negative [cppcoreguidelines-pro-bounds-constant-array-index]
  m[10] = 4;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: std::array<> index 10 is past the end of the array (which contains 10 elements) [cppcoreguidelines-pro-bounds-constant-array-index]

  m[const_index(7)] = 3;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: std::array<> index 10 is past the end of the array (which contains 10 elements)

  m[0] = 3; // OK, constant index and inside bounds
  m[1] = 3; // OK, constant index and inside bounds
  m[9] = 3; // OK, constant index and inside bounds
  m[const_index(6)] = 3; // OK, constant index and inside bounds
}

void g() {
  int a[10];
  for (int i = 0; i < 10; ++i) {
    a[i] = i;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not use array subscript when the index is not an integer constant expression
    // CHECK-FIXES: gsl::at(a, i) = i;
    gsl::at(a, i) = i; // OK, gsl::at() instead of []
  }

  a[-1] = 3; // flagged by clang-diagnostic-array-bounds
  a[10] = 4; // flagged by clang-diagnostic-array-bounds
  a[const_index(7)] = 3; // flagged by clang-diagnostic-array-bounds

  a[0] = 3; // OK, constant index and inside bounds
  a[1] = 3; // OK, constant index and inside bounds
  a[9] = 3; // OK, constant index and inside bounds
  a[const_index(6)] = 3; // OK, constant index and inside bounds
}

struct S {
  int& operator[](int i);
};

void customOperator() {
  S s;
  int i = 0;
  s[i] = 3; // OK, custom operator
}

namespace ArrayInitIndexExpr {
struct A {
  // The compiler-generated copy constructor uses an ArraySubscriptExpr. Don't warn.
  int x[3];
};

void implicitCopyMoveCtor() {
  // Force the compiler to generate a copy constructor.
  A a;
  A a2(a);

  // Force the compiler to generate a move constructor.
  A a3 = (A&&) a;
}

void lambdaCapture() {
  int arr[3];

  // Capturing an array by value uses an ArraySubscriptExpr. Don't warn. 
  [arr](){};
}

#if __cplusplus >= 201703L
void structuredBindings() {
  int arr[3];

  // Creating structured bindings by value uses an ArraySubscriptExpr. Don't warn.
  auto [a,b,c] = arr;
}
#endif
} // namespace ArrayInitIndexExpr
