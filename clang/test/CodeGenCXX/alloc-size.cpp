// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -O0 %s -o - 2>&1 -std=c++11 | FileCheck %s

namespace templates {
void *my_malloc(int N) __attribute__((alloc_size(1)));
void *my_calloc(int N, int M) __attribute__((alloc_size(1, 2)));

struct MyType {
  int arr[4];
};

template <typename T> int callMalloc();

template <typename T, int N> int callCalloc();

// CHECK-LABEL: define{{.*}} i32 @_ZN9templates6testItEv()
int testIt() {
  // CHECK: call i32 @_ZN9templates10callMallocINS_6MyTypeEEEiv
  // CHECK: call i32 @_ZN9templates10callCallocINS_6MyTypeELi4EEEiv
  return callMalloc<MyType>() + callCalloc<MyType, 4>();
}

// CHECK-LABEL: define linkonce_odr i32
// @_ZN9templates10callMallocINS_6MyTypeEEEiv
template <typename T> int callMalloc() {
  static_assert(sizeof(T) == 16, "");
  // CHECK: ret i32 16
  return __builtin_object_size(my_malloc(sizeof(T)), 0);
}

// CHECK-LABEL: define linkonce_odr i32
// @_ZN9templates10callCallocINS_6MyTypeELi4EEEiv
template <typename T, int N> int callCalloc() {
  static_assert(sizeof(T) * N == 64, "");
  // CHECK: ret i32 64
  return __builtin_object_size(my_malloc(sizeof(T) * N), 0);
}
}

namespace templated_alloc_size {
using size_t = unsigned long;

// We don't need bodies for any of these, because they're only used in
// __builtin_object_size, and that shouldn't need anything but a function
// decl with alloc_size on it.
template <typename T>
T *my_malloc(size_t N = sizeof(T)) __attribute__((alloc_size(1)));

template <typename T>
T *my_calloc(size_t M, size_t N = sizeof(T)) __attribute__((alloc_size(2, 1)));

template <size_t N>
void *dependent_malloc(size_t NT = N) __attribute__((alloc_size(1)));

template <size_t N, size_t M>
void *dependent_calloc(size_t NT = N, size_t MT = M)
    __attribute__((alloc_size(1, 2)));

template <typename T, size_t M>
void *dependent_calloc2(size_t NT = sizeof(T), size_t MT = M)
    __attribute__((alloc_size(1, 2)));

// CHECK-LABEL: define{{.*}} i32 @_ZN20templated_alloc_size6testItEv
int testIt() {
  // 122 = 4 + 5*4 + 6 + 7*8 + 4*9
  // CHECK: ret i32 122
  return __builtin_object_size(my_malloc<int>(), 0) +
         __builtin_object_size(my_calloc<int>(5), 0) +
         __builtin_object_size(dependent_malloc<6>(), 0) +
         __builtin_object_size(dependent_calloc<7, 8>(), 0) +
         __builtin_object_size(dependent_calloc2<int, 9>(), 0);
}
} // namespace templated_alloc_size

// Be sure that an ExprWithCleanups doesn't deter us.
namespace alloc_size_with_cleanups {
struct Foo {
  ~Foo();
};

void *my_malloc(const Foo &, int N) __attribute__((alloc_size(2)));

// CHECK-LABEL: define{{.*}} i32 @_ZN24alloc_size_with_cleanups6testItEv
int testIt() {
  int *const p = (int *)my_malloc(Foo{}, 3);
  // CHECK: ret i32 3
  return __builtin_object_size(p, 0);
}
} // namespace alloc_size_with_cleanups

class C {
public:
  void *my_malloc(int N) __attribute__((alloc_size(2)));
  void *my_calloc(int N, int M) __attribute__((alloc_size(2, 3)));
};

// CHECK-LABEL: define{{.*}} i32 @_Z16callMemberMallocv
int callMemberMalloc() {
  // CHECK: ret i32 16
  return __builtin_object_size(C().my_malloc(16), 0);
}

// CHECK-LABEL: define{{.*}} i32 @_Z16callMemberCallocv
int callMemberCalloc() {
  // CHECK: ret i32 32
  return __builtin_object_size(C().my_calloc(16, 2), 0);
}
