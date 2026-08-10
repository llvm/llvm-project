// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-loop-convert %t

struct S {
  int a{0};
  int b{1};
};

template <typename T, unsigned long Size>
struct array {

  T *begin() { return data; }
  const T* cbegin() const { return data; }
  T *end() { return data + Size; }
  const T *cend() const { return data + Size; }

  T data[Size];
};

const int N = 6;
S Arr[N];

void f() {
  int Sum = 0;

  for (int I = 0; I < N; ++I) {
    auto [a, b] = Arr[I];
    Sum += a + b;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead [modernize-loop-convert]
  // CHECK-FIXES: for (auto [a, b] : Arr) {
  // CHECK-FIXES-NEXT: Sum += a + b;

  array<S, N> arr;
  for (auto* It = arr.begin(); It != arr.end(); ++It) {
    auto [a, b] = *It;
    Sum = a + b;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead [modernize-loop-convert]
  // CHECK-FIXES: for (auto [a, b] : arr) {
  // CHECK-FIXES-NEXT: Sum = a + b;

  for (auto* It = arr.cbegin(); It != arr.cend(); ++It) {
    auto [a, b] = *It;
    Sum = a + b;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead [modernize-loop-convert]
  // CHECK-FIXES: for (auto [a, b] : arr) {
  // CHECK-FIXES-NEXT: Sum = a + b;
}
