// RUN: %clang_cc1 -std=c++2b %s -emit-llvm -triple x86_64-linux -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++2b %s -emit-llvm -triple x86_64-windows-msvc -o - | FileCheck %s

struct Functor {
  static int operator[](int x, int y) {
    return x + y;
  }
};

void call_static_subscript_operator() {
  Functor f;
  f[101, 102];
  f.operator[](201, 202);
  Functor{}[301, 302];
  Functor::operator[](401, 402);
}

// CHECK:      define {{.*}}call_static_subscript_operator{{.*}}
// CHECK-NEXT: entry:
// CHECK:        {{.*}} = call noundef i32 {{.*}}Functor{{.*}}(i32 noundef 101, i32 noundef 102)
// CHECK-NEXT:   {{.*}} = call noundef i32 {{.*}}Functor{{.*}}(i32 noundef 201, i32 noundef 202)
// CHECK-NEXT:   {{.*}} = call noundef i32 {{.*}}Functor{{.*}}(i32 noundef 301, i32 noundef 302)
// CHECK-NEXT:   {{.*}} = call noundef i32 {{.*}}Functor{{.*}}(i32 noundef 401, i32 noundef 402)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

struct FunctorConsteval {
  consteval static int operator[](int x, int y) {
      return x + y;
  }
};

struct FunctorConstexpr {
  constexpr static int operator[](int x, int y) {
      return x + y;
  }
};

void test_consteval_constexpr() {
  int x = 0;
  int y = FunctorConstexpr{}[x, 2];
  constexpr int z1 = FunctorConsteval{}[2, 2];
  constexpr int z2 = FunctorConstexpr{}[2, 2];
  
  static_assert(z1 == 4);
  static_assert(z2 == 4);
}

template <class T>
struct DepFunctor {
  static int operator[](T t) {
    return int(t);
  }
};

void test_dep_functors() {
  int x = DepFunctor<float>{}[1.0f];
  int y = DepFunctor<bool>{}[true];
}

// CHECK:      define {{.*}}test_dep_functors{{.*}}
// CHECK-NEXT: entry:
// CHECK:        %call = call noundef i32 {{.*}}DepFunctor{{.*}}(float noundef 1.000000e+00)
// CHECK:        %call1 = call noundef i32 {{.*}}DepFunctor{{.*}}(i1 noundef zeroext true)
// CHECK:        ret void
// CHECK-NEXT: }
