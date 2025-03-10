// RUN: %clang_cc1 -std=c++2c -verify -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

namespace std {

template <typename T> struct tuple_size;

template <int, typename> struct tuple_element;

} // namespace std

namespace Case1 {

struct S {
  int a, b;
  bool called_operator_bool = false;

  operator bool() {
    called_operator_bool = true;
    return a != b;
  }

  template <int I> int get() {
    if (!called_operator_bool)
      return a + b;
    return I == 0 ? a : b;
  }
};

} // namespace Case1

template <> struct std::tuple_size<Case1::S> {
  static const int value = 2;
};

template <int I> struct std::tuple_element<I, Case1::S> {
  using type = int;
};

namespace Case1 {

void foo() {
  if (S s(1, 2); auto [a, b] = s) {
    __builtin_assume(a == 1);
    __builtin_assume(b == 2);
  }
// CHECK: %[[call:.+]] = call {{.*}} i1 @_ZN5Case11ScvbEv
// CHECK: %{{.*}} = call {{.*}} i32 @_ZN5Case11S3getILi0EEEiv
// CHECK: %{{.*}} = call {{.*}} i32 @_ZN5Case11S3getILi1EEEiv
// CHECK: br i1 %[[call]], label {{.*}}, label {{.*}}

  if (auto [a, b] = S(1, 2)) {
    __builtin_assume(a == 1);
    __builtin_assume(b == 2);
  }
// CHECK: %[[call2:.+]] = call {{.*}} i1 @_ZN5Case11ScvbEv
// CHECK: %{{.*}} = call {{.*}} i32 @_ZN5Case11S3getILi0EEEiv
// CHECK: %{{.*}} = call {{.*}} i32 @_ZN5Case11S3getILi1EEEiv
// CHECK: br i1 %[[call2]], label {{.*}}, label {{.*}}

  if (S s(3, 4); auto& [a, b] = s) {
    __builtin_assume(a == 3);
    __builtin_assume(b == 4);
  }
// CHECK: %[[call3:.+]] = call {{.*}} i1 @_ZN5Case11ScvbEv
// CHECK: %{{.*}} = call {{.*}} i32 @_ZN5Case11S3getILi0EEEiv
// CHECK: %{{.*}} = call {{.*}} i32 @_ZN5Case11S3getILi1EEEiv
// CHECK: br i1 %[[call3]], label {{.*}}, label {{.*}}

  while (auto [i, j] = S(5, 6))
    break;

// CHECK: while.cond{{.*}}:
// CHECK: %[[call4:.+]] = call {{.*}} i1 @_ZN5Case11ScvbEv
// CHECK: %{{.*}} = call {{.*}} i32 @_ZN5Case11S3getILi0EEEiv
// CHECK: %{{.*}} = call {{.*}} i32 @_ZN5Case11S3getILi1EEEiv
// CHECK: br i1 %[[call4]], label {{.*}}, label {{.*}}

  S s(7, 8);
  while (auto& [i, j] = s)
    break;

// CHECK: while.cond{{.*}}:
// CHECK: %[[call5:.+]] = call {{.*}} i1 @_ZN5Case11ScvbEv
// CHECK: %{{.*}} = call {{.*}} i32 @_ZN5Case11S3getILi0EEEiv
// CHECK: %{{.*}} = call {{.*}} i32 @_ZN5Case11S3getILi1EEEiv
// CHECK: br i1 %[[call5]], label {{.*}}, label {{.*}}

  for (int k = 0; auto [i, j] = S(24, 42); ++k)
    break;

// CHECK: for.cond{{.*}}:
// CHECK: %[[call6:.+]] = call {{.*}} i1 @_ZN5Case11ScvbEv
// CHECK: %{{.*}} = call {{.*}} i32 @_ZN5Case11S3getILi0EEEiv
// CHECK: %{{.*}} = call {{.*}} i32 @_ZN5Case11S3getILi1EEEiv
// CHECK: br i1 %[[call6]], label {{.*}}, label {{.*}}

  for (S s(114, 514); auto& [i, j] = s; ++i)
    break;

// CHECK: for.cond{{.*}}:
// CHECK: %[[call7:.+]] = call {{.*}} i1 @_ZN5Case11ScvbEv
// CHECK: %{{.*}} = call {{.*}} i32 @_ZN5Case11S3getILi0EEEiv
// CHECK: %{{.*}} = call {{.*}} i32 @_ZN5Case11S3getILi1EEEiv
// CHECK: br i1 %[[call7]], label {{.*}}, label {{.*}}
}

} // namespace Case1
