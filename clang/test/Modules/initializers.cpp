// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -emit-llvm -DIMPORT=1 -fmodules %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-IMPORT,CHECK-NO-NS,CHECK-IMPORT-NO-NS --implicit-check-not=unused
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -emit-llvm -DIMPORT=1 -DNS -fmodules %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-IMPORT,CHECK-NS,CHECK-IMPORT-NS --implicit-check-not=unused
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -emit-llvm -DIMPORT=2 -fmodules %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NO-NS --implicit-check-not=unused
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -emit-llvm -DIMPORT=2 -DNS -fmodules %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NS --implicit-check-not=unused
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -emit-llvm -fmodules %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NO-NS --implicit-check-not=unused
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -emit-llvm -DNS -fmodules %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NS --implicit-check-not=unused

// Check that we behave sensibly when importing a header containing strong and
// weak, ordered and unordered global initializers.
//
// Our behavior is as follows:
//
//  -- for variables with one or more specific points of initialization
//     (non-template variables, whether or not they are inline or thread_local),
//     emit them if (and only if) a header containing a point of initialization
//     is transitively #included / imported.
//
//  -- for variables with unordered initialization (any kind of templated
//     variable -- excluding explicit specializations), emit them if any part
//     of any module that triggers an instantiation is imported.
//
// The intent is to:
//
// 1) preserve order of initialization guarantees
// 2) preserve the behavior of globals with ctors in headers, and specifically
//    of std::ios_base::Init (do not run the iostreams initializer nor force
//    linking in the iostreams portion of the static library unless <iostream>
//    is included)
// 3) behave conservatively-correctly with regard to unordered initializers: we
//    might run them in cases where a traditional compilation would not, but
//    will never fail to run them in cases where a traditional compilation
//    would do so
//
// Perfect handling of unordered initializers would require tracking all
// submodules containing points of instantiation, which is very hard when those
// points of instantiation are within definitions that we skip because we
// already have a (non-visible) definition for the entity:
//
// // a.h
// template<typename> int v = f();
// inline int get() { return v<int>; }
//
// // b.h
// template<typename> int v = f();
// inline int get() { return v<int>; }
//
// If a.h and b.h are built as a module, we will only have a point of
// instantiation for v<int> in one of the two headers, because we will only
// parse one of the two get() functions.

#pragma clang module build m
module m {
  module a {
    header "foo.h" { size 123 mtime 456789 }
  }
  module b {}
}

#pragma clang module contents
#pragma clang module begin m.a
inline int non_trivial() { return 3; }

#ifdef NS
namespace ns {
#endif

int a = non_trivial();
inline int b = non_trivial();
thread_local int c = non_trivial();
inline thread_local int d = non_trivial();

template<typename U> int e = non_trivial();
template<typename U> inline int f = non_trivial();
template<typename U> thread_local int g = non_trivial();
template<typename U> inline thread_local int h = non_trivial();

inline int unused = 123; // should not be emitted

template<typename T> struct X {
  static int a;
  static inline int b = non_trivial();
  static thread_local int c;
  static inline thread_local int d = non_trivial();

  template<typename U> static int e;
  template<typename U> static inline int f = non_trivial();
  template<typename U> static thread_local int g;
  template<typename U> static inline thread_local int h = non_trivial();

  static inline int unused = 123; // should not be emitted
};

template<typename T> int X<T>::a = non_trivial();
template<typename T> thread_local int X<T>::c = non_trivial();
template<typename T> template<typename U> int X<T>::e = non_trivial();
template<typename T> template<typename U> thread_local int X<T>::g = non_trivial();

inline void use(bool b, ...) {
  if (b) return;
  use(true, e<int>, f<int>, g<int>, h<int>,
      X<int>::a, X<int>::b, X<int>::c, X<int>::d,
      X<int>::e<int>, X<int>::f<int>, X<int>::g<int>, X<int>::h<int>);
}

#ifdef NS
}
#endif

#pragma clang module end
#pragma clang module endbuild

#if IMPORT == 1
// Import the module and the m.a submodule; runs the ordered initializers and
// the unordered initializers.
#pragma clang module import m.a
#elif IMPORT == 2
// Import the module but not the m.a submodule; runs only the unordered
// initializers.
#pragma clang module import m.b
#else
// Load the module but do not import any submodules; runs only the unordered
// initializers. FIXME: Should this skip all of them?
#pragma clang module load m
#endif

// CHECK-IMPORT-NO-NS-DAG: @[[A:a]] ={{.*}} global i32 0, align 4
// CHECK-IMPORT-NO-NS-DAG: @[[B:b]] = linkonce_odr global i32 0, comdat, align 4
// CHECK-IMPORT-NO-NS-DAG: @[[C:c]] ={{.*}} thread_local global i32 0, align 4
// CHECK-IMPORT-NO-NS-DAG: @[[D:d]] = linkonce_odr thread_local global i32 0, comdat, align 4
// CHECK-NO-NS-DAG: @[[E:_Z1eIiE]] = linkonce_odr global i32 0, comdat, align 4
// CHECK-NO-NS-DAG: @[[F:_Z1fIiE]] = linkonce_odr global i32 0, comdat, align 4
// CHECK-NO-NS-DAG: @[[G:_Z1gIiE]] = linkonce_odr thread_local global i32 0, comdat, align 4
// CHECK-NO-NS-DAG: @[[H:_Z1hIiE]] = linkonce_odr thread_local global i32 0, comdat, align 4

// CHECK-IMPORT-NS-DAG: @[[A:_ZN2ns1aE]] ={{.*}} global i32 0, align 4
// CHECK-IMPORT-NS-DAG: @[[B:_ZN2ns1bE]] = linkonce_odr global i32 0, comdat, align 4
// CHECK-IMPORT-NS-DAG: @[[BG:_ZGVN2ns1bE]] = linkonce_odr global i64 0, comdat($[[B]]), align 8
// CHECK-IMPORT-NS-DAG: @[[C:_ZN2ns1cE]] ={{.*}} thread_local global i32 0, align 4
// CHECK-IMPORT-NS-DAG: @[[D:_ZN2ns1dE]] = linkonce_odr thread_local global i32 0, comdat, align 4
// CHECK-IMPORT-NS-DAG: @[[DG:_ZGVN2ns1dE]] = linkonce_odr thread_local global i64 0, comdat($[[D]]), align 8
// CHECK-NS-DAG: @[[E:_ZN2ns1eIiEE]] = linkonce_odr global i32 0, comdat, align 4
// CHECK-NS-DAG: @[[F:_ZN2ns1fIiEE]] = linkonce_odr global i32 0, comdat, align 4
// CHECK-NS-DAG: @[[G:_ZN2ns1gIiEE]] = linkonce_odr thread_local global i32 0, comdat, align 4
// CHECK-NS-DAG: @[[H:_ZN2ns1hIiEE]] = linkonce_odr thread_local global i32 0, comdat, align 4

// CHECK-DAG: @[[XA:_ZN(2ns)?1XIiE1aE]] = linkonce_odr global i32 0, comdat, align 4
// CHECK-DAG: @[[XB:_ZN(2ns)?1XIiE1bE]] = linkonce_odr global i32 0, comdat, align 4
// CHECK-DAG: @[[XC:_ZN(2ns)?1XIiE1cE]] = linkonce_odr thread_local global i32 0, comdat, align 4
// CHECK-DAG: @[[XD:_ZN(2ns)?1XIiE1dE]] = linkonce_odr thread_local global i32 0, comdat, align 4
// CHECK-DAG: @[[XE:_ZN(2ns)?1XIiE1eIiEE]] = linkonce_odr global i32 0, comdat, align 4
// CHECK-DAG: @[[XF:_ZN(2ns)?1XIiE1fIiEE]] = linkonce_odr global i32 0, comdat, align 4
// CHECK-DAG: @[[XG:_ZN(2ns)?1XIiE1gIiEE]] = linkonce_odr thread_local global i32 0, comdat, align 4
// CHECK-DAG: @[[XH:_ZN(2ns)?1XIiE1hIiEE]] = linkonce_odr thread_local global i32 0, comdat, align 4

// It's OK if the order of the first 6 of these changes.
// CHECK: @llvm.global_ctors = appending global
// CHECK-SAME: @[[E_INIT:[^,]*]], {{[^@]*}} @[[E]]
// CHECK-SAME: @[[F_INIT:[^,]*]], {{[^@]*}} @[[F]]
// CHECK-SAME: @[[XA_INIT:[^,]*]], {{[^@]*}} @[[XA]]
// CHECK-SAME: @[[XE_INIT:[^,]*]], {{[^@]*}} @[[XE]]
// CHECK-SAME: @[[XF_INIT:[^,]*]], {{[^@]*}} @[[XF]]
// CHECK-SAME: @[[XB_INIT:[^,]*]], {{[^@]*}} @[[XB]]
// CHECK-IMPORT-SAME: @[[TU_INIT:[^,]*]], i8* null }]

// FIXME: Should this use __cxa_guard_acquire?
// CHECK: define {{.*}} @[[E_INIT]]()
// CHECK: load {{.*}} (i64* @_ZGV
// CHECK: store {{.*}}, i32* @[[E]],

// FIXME: Should this use __cxa_guard_acquire?
// CHECK: define {{.*}} @[[F_INIT]]()
// CHECK: load {{.*}} (i64* @_ZGV
// CHECK: store {{.*}}, i32* @[[F]],

// CHECK: define {{.*}} @[[G_INIT:__cxx_global.*]]()
// CHECK: load {{.*}} (i64* @_ZGV
// CHECK: store {{.*}}, i32* @[[G]],

// CHECK: define {{.*}} @[[H_INIT:__cxx_global.*]]()
// CHECK: load {{.*}} (i64* @_ZGV
// CHECK: store {{.*}}, i32* @[[H]],

// FIXME: Should this use __cxa_guard_acquire?
// CHECK: define {{.*}} @[[XA_INIT]]()
// CHECK: load {{.*}} (i64* @_ZGV
// CHECK: store {{.*}}, i32* @[[XA]],

// CHECK: define {{.*}} @[[XC_INIT:__cxx_global.*]]()
// CHECK: load {{.*}} (i64* @_ZGV
// CHECK: store {{.*}}, i32* @[[XC]],

// FIXME: Should this use __cxa_guard_acquire?
// CHECK: define {{.*}} @[[XE_INIT]]()
// CHECK: load {{.*}} (i64* @_ZGV
// CHECK: store {{.*}}, i32* @[[XE]],

// CHECK: define {{.*}} @[[XG_INIT:__cxx_global.*]]()
// CHECK: load {{.*}} (i64* @_ZGV
// CHECK: store {{.*}}, i32* @[[XG]],

// CHECK: define {{.*}} @[[XH_INIT:__cxx_global.*]]()
// CHECK: load {{.*}} (i64* @_ZGV
// CHECK: store {{.*}}, i32* @[[XH]],

// FIXME: Should this use __cxa_guard_acquire?
// CHECK: define {{.*}} @[[XF_INIT]]()
// CHECK: load {{.*}} (i64* @_ZGV
// CHECK: store {{.*}}, i32* @[[XF]],

// CHECK: define {{.*}} @[[XD_INIT:__cxx_global.*]]()
// CHECK: load {{.*}} (i64* @_ZGV
// CHECK: store {{.*}}, i32* @[[XD]],

// FIXME: Should this use __cxa_guard_acquire?
// CHECK: define {{.*}} @[[XB_INIT]]()
// CHECK: load {{.*}} (i64* @_ZGV
// CHECK: store {{.*}}, i32* @[[XB]],

// CHECK-IMPORT: define {{.*}} @[[A_INIT:__cxx_global.*]]()
// CHECK-IMPORT: call i32 @_Z11non_trivialv(
// CHECK-IMPORT: store {{.*}}, i32* @[[A]],

// CHECK-IMPORT: define {{.*}} @[[B_INIT:__cxx_global.*]]()
// CHECK-IMPORT: call i32 @__cxa_guard_acquire(i64* @_ZGV
// CHECK-IMPORT: store {{.*}}, i32* @[[B]],

// CHECK-IMPORT: define {{.*}} @[[C_INIT:__cxx_global.*]]()
// CHECK-IMPORT: call i32 @_Z11non_trivialv(
// CHECK-IMPORT: store {{.*}}, i32* @[[C]],

// CHECK-IMPORT: define {{.*}} @[[D_INIT:__cxx_global.*]]()
// CHECK-IMPORT: load {{.*}} (i64* @_ZGV
// CHECK-IMPORT: store {{.*}}, i32* @[[D]],


// CHECK-IMPORT: define {{.*}} @[[TU_INIT]]()
// CHECK-IMPORT: call void @[[A_INIT]]()

// CHECK-IMPORT: define {{.*}} @__tls_init()
// CHECK-IMPORT: call void @[[C_INIT]]()
// CHECK-IMPORT: call void @[[D_INIT]]()
