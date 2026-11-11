// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -o - -verify=ok | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -o - -verify=ok -disable-llvm-passes | FileCheck %s --check-prefix=ATTRS
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -o /dev/null -verify -DLACKS_FEATURE
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -o /dev/null -verify=external -DEXTERNAL
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -o /dev/null -verify=abi -DABI

// ok-no-diagnostics

typedef long long avx_vector __attribute__((vector_size(32)));

template <class T> [[gnu::always_inline, gnu::target("avx")]] T needs_avx() { return T(); }

// A template cannot name the target features of each of its instantiations, so
// an always_inline function takes on those that its always_inline callees need.
template <class T> [[gnu::always_inline]] T wrapper() { return needs_avx<T>(); } // #wrapper
template <class T> [[gnu::always_inline]] T outer_wrapper() { return wrapper<T>(); }
[[gnu::always_inline]] inline int inline_wrapper() { return needs_avx<int>(); } // #inline-wrapper

// CHECK-LABEL: define {{.*}} @_Z7has_avxv(
// CHECK-NOT: call {{.*}} @_Z
// CHECK: ret
// ATTRS: define linkonce_odr {{.*}} @_Z13outer_wrapperI{{.*}}#[[OUTER:[0-9]+]]
// ATTRS: attributes #[[OUTER]] = { {{.*}}"target-features"="{{[^"]*}}+avx{{[,"]}}
[[gnu::target("avx")]] avx_vector has_avx() { return outer_wrapper<avx_vector>(); }
[[gnu::target("avx")]] int has_avx_too() { return inline_wrapper(); }

#ifdef LACKS_FEATURE
// The wrappers are emitted after this caller.
// expected-note@#wrapper {{always_inline function 'wrapper' calls 'needs_avx' here}}
int lacks_avx() { return outer_wrapper<int>(); } // expected-error {{always_inline function 'needs_avx' requires target feature 'avx', but would be inlined into function 'lacks_avx' that is compiled without support for 'avx'}}

// The wrapper is emitted before this caller.
// expected-note@#inline-wrapper {{always_inline function 'inline_wrapper' calls 'needs_avx' here}}
inline int inline_lacks_avx() { return inline_wrapper(); } // expected-error {{always_inline function 'needs_avx' requires target feature 'avx', but would be inlined into function 'inline_lacks_avx' that is compiled without support for 'avx'}}
int use_inline_lacks_avx() { return inline_lacks_avx(); }
#endif

#ifdef EXTERNAL
// This is emitted even when unused, for callers that cannot be checked here.
[[gnu::always_inline]] int external_wrapper() { return needs_avx<int>(); } // external-error {{always_inline function 'needs_avx' requires target feature 'avx', but would be inlined into function 'external_wrapper' that is compiled without support for 'avx'}}
#endif

#ifdef ABI
// An inherited target feature changes how vectors are passed, just like a
// declared one.
avx_vector lacks_avx_abi(avx_vector);
// abi-error@+1 {{AVX vector argument of type 'avx_vector' (vector of 4 'long long' values) without 'avx' enabled changes the ABI}}
template <class T> [[gnu::always_inline]] T abi_wrapper() { return lacks_avx_abi(needs_avx<T>()); }
[[gnu::target("avx")]] avx_vector has_avx_abi() { return abi_wrapper<avx_vector>(); }

// Nothing is inherited here, so this is diagnosed as it would be otherwise.
// abi-warning@+2 {{AVX vector return of type 'avx_vector' (vector of 4 'long long' values) without 'avx' enabled changes the ABI}}
// abi-warning@+1 {{AVX vector argument of type 'avx_vector' (vector of 4 'long long' values) without 'avx' enabled changes the ABI}}
[[gnu::always_inline]] inline avx_vector identity(avx_vector v) { return v; }
// abi-error@+1 {{AVX vector argument of type 'avx_vector' (vector of 4 'long long' values) without 'avx' enabled changes the ABI}}
[[gnu::target("avx")]] avx_vector calls_identity(avx_vector v) { return identity(v); }
#endif
