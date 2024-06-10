// RUN: %clang_cc1 -ast-print %s -std=c++11 | FileCheck %s

// Make sure that for template value arguments that are unscoped enumerators,
// no qualified enum information is included in their name, as their visibility
// is global. In the case of scoped enumerators, they must include information
// about their enum enclosing scope.

enum E1 { e1 };
template<E1 v> struct tmpl_1 {};
// CHECK: template<> struct tmpl_1<e1>
tmpl_1<E1::e1> TMPL_1;                      // Name must be 'e1'.

namespace nsp_1 { enum E2 { e2 }; }
template<nsp_1::E2 v> struct tmpl_2 {};
// CHECK: template<> struct tmpl_2<nsp_1::e2>
tmpl_2<nsp_1::E2::e2> TMPL_2;               // Name must be 'nsp_1::e2'.

enum class E3 { e3 };
template<E3 v> struct tmpl_3 {};
// CHECK: template<> struct tmpl_3<E3::e3>
tmpl_3<E3::e3> TMPL_3;                      // Name must be 'E3::e3'.

namespace nsp_2 { enum class E4 { e4 }; }
template<nsp_2::E4 v> struct tmpl_4 {};
// CHECK: template<> struct tmpl_4<nsp_2::E4::e4>
tmpl_4<nsp_2::E4::e4> TMPL_4;               // Name must be 'nsp_2::E4::e4'.
