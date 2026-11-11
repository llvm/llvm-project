// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -disable-llvm-passes -verify=clean -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -o /dev/null -verify -DERRORS

// An always_inline function that has no target attribute and that only direct
// calls in this translation unit can reach is compiled with the target
// features that its callers have in common.

// clean-no-diagnostics

typedef float v8f __attribute__((vector_size(32)));

template <class> struct Zero;
template <> struct Zero<v8f> {
  [[gnu::target("avx"), gnu::always_inline]] inline v8f operator()() const {
    return v8f{};
  }
};

template <class T>
[[gnu::always_inline]] static inline auto direct(T gen) { return gen(); }

// Unlike direct, this is not instantiated until the end of the translation
// unit, after its caller has been emitted.
template <class T>
[[gnu::always_inline]] static inline v8f indirect(T gen) { return direct(gen); }

// Passing a vector on must not be diagnosed as a change of the ABI either.
[[gnu::always_inline]] static inline v8f pass(v8f x) { return x; }

struct Vec {
  v8f v;
  [[gnu::always_inline]] Vec(v8f v) : v(v) {}
};

[[gnu::target("avx")]] v8f avx_caller() {
  return Vec(pass(indirect(Zero<v8f>()))).v;
}

[[gnu::target("avx2"), gnu::always_inline]] static inline int leaf() {
  return 1;
}
[[gnu::always_inline]] static inline int mid() { return leaf(); }
[[gnu::always_inline]] static inline int top() { return mid(); }

[[gnu::target("avx2")]] int avx2_caller() {
  return top() + []() __attribute__((always_inline)) { return leaf(); }();
}

[[gnu::always_inline]] static inline int common() { return 0; }
[[gnu::target("avx")]] int common_avx() { return common(); }
[[gnu::target("avx2")]] int common_avx2() { return common(); }

// A call that is never emitted requires nothing of the callers.
[[gnu::always_inline]] static inline int folded() {
  if (0)
    return leaf();
  return 0;
}
int default_caller() { return folded(); }

// Other translation units may give their definitions other features, so the
// linker must not merge them with this one.
[[gnu::always_inline]] inline int odr() { return leaf(); }
[[gnu::target("avx2")]] int odr_caller() { return odr(); }

[[gnu::always_inline]] inline int odr_default() { return 0; }
int odr_default_caller() { return odr_default(); }

// CHECK-DAG: define dso_local {{.*}} @_Z10avx_callerv() #[[AVX_CALLER:[0-9]+]]
// CHECK-DAG: define internal {{.*}} @_ZL4passDv8_f({{.*}}) #[[AVX_VECTOR:[0-9]+]]
// CHECK-DAG: define internal {{.*}} @_ZL8indirectI4ZeroIDv8_fEES1_T_() #[[AVX_VECTOR]]
// CHECK-DAG: define internal {{.*}} @_ZL6directI4ZeroIDv8_fEEDaT_() #[[AVX_VECTOR]]
// CHECK-DAG: define internal void @_ZN3VecC1EDv8_f({{.*}}) unnamed_addr #[[AVX:[0-9]+]]
// CHECK-DAG: define internal void @_ZN3VecC2EDv8_f({{.*}}) unnamed_addr #[[AVX]]
// CHECK-DAG: define dso_local {{.*}} @_Z11avx2_callerv() #[[AVX2_CALLER:[0-9]+]]
// CHECK-DAG: define internal {{.*}} @_ZL3topv() #[[AVX2:[0-9]+]]
// CHECK-DAG: define internal {{.*}} @_ZL3midv() #[[AVX2]]
// CHECK-DAG: define internal {{.*}} @"_ZZ11avx2_callervENK3$_0clEv"({{.*}}) #[[AVX2]]
// CHECK-DAG: define internal {{.*}} @_ZL6commonv() #[[AVX]]
// CHECK-DAG: define dso_local {{.*}} @_Z14default_callerv() #[[DEFAULT_CALLER:[0-9]+]]
// CHECK-DAG: define internal {{.*}} @_ZL6foldedv() #[[DEFAULT:[0-9]+]]
// CHECK-DAG: define internal {{.*}} @_Z3odrv() #[[AVX2]]
// CHECK-DAG: define linkonce_odr {{.*}} @_Z11odr_defaultv() #[[DEFAULT]] comdat
// CHECK-DAG: attributes #[[AVX_CALLER]] = { {{.*}}"target-features"="[[AVX_FEATURES:[^"]+]]"
// CHECK-DAG: attributes #[[AVX_VECTOR]] = { alwaysinline {{.*}}"target-features"="[[AVX_FEATURES]]"
// CHECK-DAG: attributes #[[AVX]] = { alwaysinline {{.*}}"target-features"="[[AVX_FEATURES]]"
// CHECK-DAG: attributes #[[AVX2_CALLER]] = { {{.*}}"target-features"="[[AVX2_FEATURES:[^"]+]]"
// CHECK-DAG: attributes #[[AVX2]] = { alwaysinline {{.*}}"target-features"="[[AVX2_FEATURES]]"
// CHECK-DAG: attributes #[[DEFAULT_CALLER]] = { {{.*}}"target-features"="[[DEFAULT_FEATURES:[^"]+]]"
// CHECK-DAG: attributes #[[DEFAULT]] = { alwaysinline {{.*}}"target-features"="[[DEFAULT_FEATURES]]"

#ifdef ERRORS
int disabled() {
  return top(); // expected-error {{always_inline function 'top' requires target feature 'avx2', but would be inlined into function 'disabled' that is compiled without support for 'avx2'}}
}

// Functions that can run without having been inlined into a function that was
// checked keep the features they were declared with.
[[gnu::always_inline]] static inline int escaping() {
  return leaf(); // expected-error {{always_inline function 'leaf' requires target feature 'avx2', but would be inlined into function 'escaping' that is compiled without support for 'avx2'}}
}
int (*pointer())() { return escaping; }

[[gnu::always_inline]] static inline int initializer() {
  return leaf(); // expected-error {{always_inline function 'leaf' requires target feature 'avx2', but would be inlined into function 'initializer' that is compiled without support for 'avx2'}}
}
int global = initializer();

[[gnu::always_inline]] int external() {
  return leaf(); // expected-error {{always_inline function 'leaf' requires target feature 'avx2', but would be inlined into function 'external' that is compiled without support for 'avx2'}}
}
[[gnu::target("avx2")]] int external_caller() { return external(); }
#endif
