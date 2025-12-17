// RUN: %clang_cc1 -std=c++2c -verify -emit-llvm -triple=x86_64-pc-linux-gnu %s -o -                                         | FileCheck %s
// RUN: %clang_cc1 -std=c++2c -verify -emit-llvm -triple=x86_64-pc-linux-gnu %s -o - -fexperimental-new-constant-interpreter | FileCheck %s
// expected-no-diagnostics

namespace std {

template <typename T> struct tuple_size;

template <int, typename> struct tuple_element;

} // namespace std

namespace Case1 {

struct S {
  int a, b;
  bool flag = false;

  constexpr explicit operator bool() {
    flag = true;
    return a != b;
  }

  constexpr operator int() {
    flag = true;
    return a * b;
  }

  constexpr bool operator==(S rhs) const {
    return a == rhs.a && b == rhs.b;
  }

  template <int I>
  constexpr int& get() {
    if (!flag)
      return a = a + b;
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
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi0EEERiv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi1EEERiv
// CHECK: br i1 %[[call]], label {{.*}}, label {{.*}}

  if (auto [a, b] = S(1, 2)) {
    __builtin_assume(a == 1);
    __builtin_assume(b == 2);
  }
// CHECK: %[[call2:.+]] = call {{.*}} i1 @_ZN5Case11ScvbEv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi0EEERiv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi1EEERiv
// CHECK: br i1 %[[call2]], label {{.*}}, label {{.*}}

  if (S s(3, 4); auto& [a, b] = s) {
    __builtin_assume(a == 3);
    __builtin_assume(b == 4);
  }
// CHECK: %[[call3:.+]] = call {{.*}} i1 @_ZN5Case11ScvbEv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi0EEERiv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi1EEERiv
// CHECK: br i1 %[[call3]], label {{.*}}, label {{.*}}

  while (auto [i, j] = S(5, 6))
    break;

// CHECK: while.cond{{.*}}:
// CHECK: %[[call4:.+]] = call {{.*}} i1 @_ZN5Case11ScvbEv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi0EEERiv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi1EEERiv
// CHECK: br i1 %[[call4]], label {{.*}}, label {{.*}}

  S s(7, 8);
  while (auto& [i, j] = s)
    break;

// CHECK: while.cond{{.*}}:
// CHECK: %[[call5:.+]] = call {{.*}} i1 @_ZN5Case11ScvbEv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi0EEERiv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi1EEERiv
// CHECK: br i1 %[[call5]], label {{.*}}, label {{.*}}

  for (int k = 0; auto [i, j] = S(24, 42); ++k)
    break;

// CHECK: for.cond{{.*}}:
// CHECK: %[[call6:.+]] = call {{.*}} i1 @_ZN5Case11ScvbEv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi0EEERiv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi1EEERiv
// CHECK: br i1 %[[call6]], label {{.*}}, label {{.*}}

  for (S s(114, 514); auto& [i, j] = s; ++i)
    break;

// CHECK: for.cond{{.*}}:
// CHECK: %[[call7:.+]] = call {{.*}} i1 @_ZN5Case11ScvbEv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi0EEERiv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi1EEERiv
// CHECK: br i1 %[[call7]], label {{.*}}, label {{.*}}

  switch (S s(10, 11); auto& [i, j] = s) {
    case 10 * 11:
      __builtin_assume(i == 10);
      __builtin_assume(j == 11);
      break;
    default:
      break;
  }

// CHECK: %[[call8:.+]] = call {{.*}} i32 @_ZN5Case11ScviEv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi0EEERiv
// CHECK: %{{.*}} = call {{.*}} ptr @_ZN5Case11S3getILi1EEERiv
// CHECK: switch i32 %[[call8]], label {{.*}}

}

constexpr int bar(auto) {
  constexpr auto value = [] {
    if (S s(1, 2); auto [i, j] = s)
      return S(i, j);
    return S(0, 0);
  }();
  static_assert(value == S(1, 2));

  constexpr auto value2 = [] {
    if (auto [a, b] = S(1, 2))
      return S(a, b);
    return S(0, 0);
  }();
  static_assert(value2 == S(1, 2));

  constexpr auto value3 = [] {
    if (auto&& [a, b] = S(3, 4))
      return S(a, b);
    return S(0, 0);
  }();
  static_assert(value3 == S(3, 4));

  constexpr auto value4 = [] {
    S s(7, 8);
    int cnt = 0;
    while (auto& [i, j] = s) {
      s.flag = false;
      ++i, ++j;
      if (++cnt == 10)
        break;
    }
    return s;
  }();
  static_assert(value4 == S(17, 18));

  constexpr auto value5 = [] {
    S s(3, 4);
    for (int cnt = 0; auto& [x, y] = s; s.flag = false, ++cnt) {
      if (cnt == 3)
        break;
      ++x, ++y;
    }
    return s;
  }();
  static_assert(value5 == S(6, 7));

  constexpr auto value6 = [] {
    switch (auto [x, y] = S(3, 4)) {
      case 3 * 4:
        return S(x, y);
      default:
        return S(y, x);
    }
  }();
  static_assert(value6 == S(3, 4));

  return 42;
}

constexpr int value = bar(1);

#if 0

// FIXME: This causes clang to ICE, though this is not a regression.
constexpr int ice(auto) {
  if constexpr (S s(1, 2); auto [i, j] = s) {
    static_assert(i == 1);
  }
  return 42;
}

constexpr int value2 = ice(1);

#endif

} // namespace Case1
