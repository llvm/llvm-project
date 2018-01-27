// RUN: c-index-test core -print-source-symbols -- %s -std=c++14 | FileCheck %s
// RUN: %clang_cc1 -emit-pch %s -std=c++14 -o %t.pch
// RUN: c-index-test core -print-source-symbols -module-file %t.pch | FileCheck %s

// CHECK: [[@LINE+2]]:8 | struct(Gen)/C++ | DETECTOR | [[DETECTOR_USR:.*]] | {{.*}} | Def | rel: 0
template <class _Default, class _AlwaysVoid, template <class...> class _Op, class... _Args>
struct DETECTOR {
 using value_t = int;
};

struct nonesuch {};

// CHECK: [[@LINE+4]]:9 | type-alias/C++ | is_detected
// CHECK: [[@LINE+3]]:32 | struct(Gen)/C++ | DETECTOR | [[DETECTOR_USR]] | {{.*}} | Ref,RelCont | rel: 1
// CHECK-NEXT:	RelCont | is_detected
template <template<class...> class _Op, class... _Args>
  using is_detected = typename DETECTOR<nonesuch, void, _Op, _Args...>::value_t;
