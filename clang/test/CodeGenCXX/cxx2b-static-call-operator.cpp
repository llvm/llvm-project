// RUN: %clang_cc1 -std=c++23 %s -emit-llvm -triple x86_64-linux -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++23 %s -emit-llvm -triple x86_64-windows-msvc -o - | FileCheck %s

struct Functor {
  static int operator()(int x, int y) {
    return x + y;
  }
};

auto GetALambda() {
  return [](int x, int y) static {
    return x + y;
  };
}

void CallsTheLambda() {
  GetALambda()(1, 2);
}

// CHECK:      define {{.*}}CallsTheLambda{{.*}}
// CHECK-NEXT: entry:
// CHECK:        {{.*}}call {{.*}}GetALambda{{.*}}()
// CHECK-NEXT:   {{.*}} = call noundef i32 {{.*}}(i32 noundef 1, i32 noundef 2)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

Functor GetAFunctor() {
  return {};
}

void call_static_call_operator() {
  Functor f;
  f(101, 102);
  f.operator()(201, 202);
  Functor{}(301, 302);
  Functor::operator()(401, 402);
  GetAFunctor()(501, 502);
}

// CHECK:      define {{.*}}call_static_call_operator{{.*}}
// CHECK-NEXT: entry:
// CHECK:        {{.*}} = call noundef i32 {{.*}}Functor{{.*}}(i32 noundef 101, i32 noundef 102)
// CHECK-NEXT:   {{.*}} = call noundef i32 {{.*}}Functor{{.*}}(i32 noundef 201, i32 noundef 202)
// CHECK-NEXT:   {{.*}} = call noundef i32 {{.*}}Functor{{.*}}(i32 noundef 301, i32 noundef 302)
// CHECK-NEXT:   {{.*}} = call noundef i32 {{.*}}Functor{{.*}}(i32 noundef 401, i32 noundef 402)
// CHECK:        {{.*}}call {{.*}}GetAFunctor{{.*}}()
// CHECK-NEXT:   {{.*}} = call noundef i32 {{.*}}Functor{{.*}}(i32 noundef 501, i32 noundef 502)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

struct FunctorConsteval {
  consteval static int operator()(int x, int y) {
      return x + y;
  }
};

struct FunctorConstexpr {
  constexpr static int operator()(int x, int y) {
      return x + y;
  }
};

constexpr auto my_lambda = []() constexpr {
  return 3;
};

void test_consteval_constexpr() {
  int x = 0;
  int y = FunctorConstexpr{}(x, 2);
  constexpr int z1 = FunctorConsteval{}(2, 2);
  constexpr int z2 = FunctorConstexpr{}(2, 2);
  
  static_assert(z1 == 4);
  static_assert(z2 == 4);

  constexpr auto my_lambda = []() constexpr static {
      return 3;
  };
  constexpr int (*f)(void) = my_lambda;
  constexpr int k = f();
  static_assert(k == 3);
}

template <class T>
struct DepFunctor {
  static int operator()(T t) {
    return int(t);
  }
};

template<class T>
auto dep_lambda1() {
  return [](T t) static -> int {
    return t;
  };
}

auto dep_lambda2() {
  return [](auto t) static -> int {
    return t;
  };
}

void test_dep_functors() {
  int x = DepFunctor<float>{}(1.0f);
  int y = DepFunctor<bool>{}(true);

  int a = dep_lambda1<float>()(1.0f);
  int b = dep_lambda1<bool>()(true);

  int h = dep_lambda2()(1.0f);
  int i = dep_lambda2()(true);
}

// CHECK:      define {{.*}}test_dep_functors{{.*}}
// CHECK-NEXT: entry:
// CHECK:        {{.*}} = call noundef i32 {{.*}}DepFunctor{{.*}}(float noundef 1.000000e+00)
// CHECK:        {{.*}} = call noundef i32 {{.*}}DepFunctor{{.*}}(i1 noundef zeroext true)
// CHECK:        {{.*}}call {{.*}}dep_lambda1{{.*}}()
// CHECK:        {{.*}} = call noundef i32 {{.*}}dep_lambda1{{.*}}(float noundef 1.000000e+00)
// CHECK:        {{.*}}call {{.*}}dep_lambda1{{.*}}()
// CHECK:        {{.*}} = call noundef i32 {{.*}}dep_lambda1{{.*}}(i1 noundef zeroext true)
// CHECK:        {{.*}}call {{.*}}dep_lambda2{{.*}}()
// CHECK:        {{.*}} = call noundef i32 {{.*}}dep_lambda2{{.*}}(float noundef 1.000000e+00)
// CHECK:        {{.*}}call {{.*}}dep_lambda2{{.*}}()
// CHECK:        {{.*}} = call noundef i32 {{.*}}dep_lambda2{{.*}}(i1 noundef zeroext true)
// CHECK:        ret void
// CHECK-NEXT: }


struct __unique {
    static constexpr auto operator()() { return 4; };

    using P = int();
    constexpr operator P*() { return operator(); }
};

__unique four{};

int test_four() {
  // Checks that overload resolution works.
  return four();
}
