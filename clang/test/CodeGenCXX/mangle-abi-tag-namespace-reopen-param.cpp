// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fcxx-exceptions \
// RUN:   -fexceptions -emit-llvm -o - %s | FileCheck %s

// The tags of an inline namespace do not only matter through the return type
// of a function. Here "A" is an implicit tag of early_param until the
// namespace of its parameter type gets the same tag. The names derived from
// early_param must not change when that happens.

template <class T> void thrower() { throw T(); }

struct __attribute__((abi_tag("A"))) Tagged { Tagged(); };
inline namespace NP { struct SP {}; }

Tagged early_param(SP) {
  struct L {};
  try {
    thrower<L>();
  } catch (L &) {
  }
  return {};
}

template <class T> struct Holder { static Tagged m; };
template <class T> Tagged Holder<T>::m;
Tagged *use_m() { return &Holder<SP>::m; }

inline namespace NP __attribute__((abi_tag("A"))) {}

Tagged late_param(SP) { return {}; }

// CHECK: @_ZTIZ11early_paramB1AN2NP2SPEE1L = internal constant
// CHECK: @_ZN6HolderIN2NP2SPEE1mB1AE =
// CHECK: @_ZGVN6HolderIN2NP2SPEE1mB1AE =
// CHECK-LABEL: define {{.*}} @_Z11early_paramB1AN2NP2SPE(
// CHECK: catch ptr @_ZTIZ11early_paramB1AN2NP2SPEE1L
// CHECK-LABEL: define {{.*}} @_Z7throwerIZ11early_paramB1AN2NP2SPEE1LEvv(
// CHECK: call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIZ11early_paramB1AN2NP2SPEE1L, ptr null)
// CHECK-LABEL: define {{.*}} @_Z10late_paramN2NP2SPE(
