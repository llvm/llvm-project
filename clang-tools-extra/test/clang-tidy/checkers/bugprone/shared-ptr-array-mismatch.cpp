// RUN: %check_clang_tidy %s bugprone-shared-ptr-array-mismatch %t

namespace std {

template <typename T>
struct shared_ptr {
  template <class Y>
  explicit shared_ptr(Y *) {}
  template <class Y, class Deleter>
  shared_ptr(Y *, Deleter) {}
};

} // namespace std

struct A {};

void f1() {
  std::shared_ptr<int> P1{new int};
  std::shared_ptr<int> P2{new int[10]};
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
  // CHECK-FIXES: std::shared_ptr<int[]> P2{new int[10]};
  // clang-format off
  std::shared_ptr<  int  > P3{new int[10]};
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
  // CHECK-FIXES: std::shared_ptr<  int[]  > P3{new int[10]};
  // clang-format on
  std::shared_ptr<int> P4(new int[10]);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
  // CHECK-FIXES: std::shared_ptr<int[]> P4(new int[10]);
  new std::shared_ptr<int>(new int[10]);
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
  std::shared_ptr<int[]> P5(new int[10]);
  std::shared_ptr<int> P6(new int[10], [](const int *Ptr) {});
}

void f2() {
  std::shared_ptr<A> P1(new A);
  std::shared_ptr<A> P2(new A[10]);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
  // CHECK-FIXES: std::shared_ptr<A[]> P2(new A[10]);
  std::shared_ptr<A[]> P3(new A[10]);
}

void f3() {
  std::shared_ptr<int> P1{new int}, P2{new int[10]}, P3{new int[10]};
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
  // CHECK-MESSAGES: :[[@LINE-2]]:57: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
}

struct S {
  std::shared_ptr<int> P1;
  std::shared_ptr<int> P2{new int[10]};
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
  std::shared_ptr<int> P3{new int}, P4{new int[10]};
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
  S() : P1{new int[10]} {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
};

void f_parm(std::shared_ptr<int>);

void f4() {
  f_parm(std::shared_ptr<int>{new int[10]});
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
}

std::shared_ptr<int> f_ret() {
  return std::shared_ptr<int>(new int[10]);
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
}

template <class T>
void f_tmpl() {
  std::shared_ptr<T> P1{new T[10]};
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
  // CHECK-FIXES: std::shared_ptr<T[]> P1{new T[10]};
}

void f5() {
  f_tmpl<char>();
}

#define CHAR_PTR_TYPE std::shared_ptr<char>
#define CHAR_PTR_VAR(X) \
  X { new char[10] }
#define CHAR_PTR_INIT(X, Y) \
  std::shared_ptr<char> X { Y }

void f6() {
  CHAR_PTR_TYPE P1{new char[10]};
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
  std::shared_ptr<char> CHAR_PTR_VAR(P2);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
  // CHECK-FIXES: std::shared_ptr<char[]> CHAR_PTR_VAR(P2);
  CHAR_PTR_INIT(P3, new char[10]);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
}
