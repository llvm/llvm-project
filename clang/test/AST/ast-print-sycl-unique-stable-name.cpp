// RUN: %clang_cc1 -ast-print -fsycl-is-device %s -o - -triple spir64-sycldevice | FileCheck %s

template <typename T>
void WrappedInTemplate(T t) {
  (void)__builtin_sycl_unique_stable_name(T);
  (void)__builtin_sycl_unique_stable_name(typename T::type);
  (void)__builtin_sycl_unique_stable_name(decltype(t.foo()));
}

struct Type {
  using type = int;

  double foo();
};

void use() {
  WrappedInTemplate(Type{});
}

// CHECK: template <typename T> void WrappedInTemplate(T t)
// CHECK: __builtin_sycl_unique_stable_name(T);
// CHECK: __builtin_sycl_unique_stable_name(typename T::type);
// CHECK: __builtin_sycl_unique_stable_name(decltype(t.foo()));

// CHECK: template<> void WrappedInTemplate<Type>(Type t)
// CHECK: __builtin_sycl_unique_stable_name(Type);
// CHECK: __builtin_sycl_unique_stable_name(typename Type::type);
// CHECK: __builtin_sycl_unique_stable_name(decltype(t.foo()));
