// RUN: %check_clang_tidy %s bugprone-unique-ptr-array-mismatch %t

namespace std {

template<class T> struct default_delete {};
template<class T> struct default_delete<T[]> {};

template<class T, class Deleter = std::default_delete<T>>
class unique_ptr {
public:
  explicit unique_ptr(T* p) noexcept;
  unique_ptr(T* p, Deleter d1 ) noexcept;
};

template <class T, class Deleter>
class unique_ptr<T[], Deleter> {
public:
  template<class U>
  explicit unique_ptr(U p) noexcept;
  template<class U>
  unique_ptr(U p, Deleter d1) noexcept;
};

} // namespace std

struct A {};

using PtrT = std::unique_ptr<A>;
using PtrTArr = std::unique_ptr<A[]>;

void f1() {
  std::unique_ptr<int> P1{new int};
  std::unique_ptr<int> P2{new int[10]};
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
  // CHECK-FIXES: std::unique_ptr<int[]> P2{new int[10]};
  // clang-format off
  std::unique_ptr<  int  > P3{new int[10]};
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
  // CHECK-FIXES: std::unique_ptr<  int[]  > P3{new int[10]};
  // clang-format on
  std::unique_ptr<int> P4(new int[10]);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
  // CHECK-FIXES: std::unique_ptr<int[]> P4(new int[10]);
  new std::unique_ptr<int>(new int[10]);
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
  std::unique_ptr<int[]> P5(new int[10]);

  A deleter;
  std::unique_ptr<int, A> P6(new int[10], deleter);
  std::unique_ptr<int, A> P7(new int[10]);
  std::default_delete<int[]> def_del;
  std::unique_ptr<int, std::default_delete<int[]>> P8(new int[10], def_del);

  new PtrT(new A[10]);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
  new PtrTArr(new A[10]);
}

void f2() {
  std::unique_ptr<A> P1(new A);
  std::unique_ptr<A> P2(new A[10]);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
  // CHECK-FIXES: std::unique_ptr<A[]> P2(new A[10]);
  std::unique_ptr<A[]> P3(new A[10]);
}

void f3() {
  std::unique_ptr<int> P1{new int}, P2{new int[10]}, P3{new int[10]};
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
  // CHECK-MESSAGES: :[[@LINE-2]]:57: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
}

struct S {
  std::unique_ptr<int> P1;
  std::unique_ptr<int> P2{new int[10]};
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
  std::unique_ptr<int> P3{new int}, P4{new int[10]};
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
  S() : P1{new int[10]} {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
};

void f_parm(std::unique_ptr<int>);

void f4() {
  f_parm(std::unique_ptr<int>{new int[10]});
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
}

std::unique_ptr<int> f_ret() {
  return std::unique_ptr<int>(new int[10]);
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
}

template <class T>
void f_tmpl() {
  std::unique_ptr<T> P1{new T[10]};
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
  // CHECK-FIXES: std::unique_ptr<T[]> P1{new T[10]};
}

void f5() {
  f_tmpl<char>();
}

template <class T>
void f_tmpl_1() {
  std::unique_ptr<T> P1{new T[10]};
  // FIXME_CHECK-MESSAGES: :[[@LINE-1]]:25: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
  // FIXME_CHECK-FIXES: std::unique_ptr<T[]> P1{new T[10]};
}

#define CHAR_PTR_TYPE std::unique_ptr<char>
#define CHAR_PTR_VAR(X) \
  X { new char[10] }
#define CHAR_PTR_INIT(X, Y) \
  std::unique_ptr<char> X { Y }

void f6() {
  CHAR_PTR_TYPE P1{new char[10]};
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
  std::unique_ptr<char> CHAR_PTR_VAR(P2);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
  // CHECK-FIXES: std::unique_ptr<char[]> CHAR_PTR_VAR(P2);
  CHAR_PTR_INIT(P3, new char[10]);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: unique pointer to non-array is initialized with array [bugprone-unique-ptr-array-mismatch]
}
