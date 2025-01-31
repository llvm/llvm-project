// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++20 mod.cppm -emit-module-interface -o mod.pcm -fallow-pcm-with-compiler-errors -verify
// RUN: %clang_cc1 -std=c++20 main.cpp -fmodule-file=mod=mod.pcm -verify -fallow-pcm-with-compiler-errors -fsyntax-only -ast-dump-all | FileCheck %s

// RUN: %clang_cc1 -std=c++20 mod.cppm -emit-reduced-module-interface -o mod.pcm -fallow-pcm-with-compiler-errors -verify
// RUN: %clang_cc1 -std=c++20 main.cpp -fmodule-file=mod=mod.pcm -verify -fallow-pcm-with-compiler-errors -fsyntax-only -ast-dump-all | FileCheck %s

//--- mod.cppm
export module mod;

template <typename T, auto Q> // expected-note 2{{template parameter is declared here}}
concept ReferenceOf = Q;

// expected-error@+2 {{unknown type name 'AngleIsInvalidNow'}}
// expected-error@+1 {{constexpr variable 'angle' must be initialized by a constant expression}}
constexpr struct angle {AngleIsInvalidNow e;} angle;

// expected-error@+1 {{non-type template argument is not a constant expression}}
template<ReferenceOf<angle> auto R, typename Rep> requires requires(Rep v) {cos(v);}
auto cos(const Rep& q);

// expected-error@+1 {{non-type template argument is not a constant expression}}
template<ReferenceOf<angle> auto R, typename Rep> requires requires(Rep v) {tan(v);}
auto tan(const Rep& q);

//--- main.cpp
// expected-no-diagnostics
import mod;

// CHECK:      |-FunctionTemplateDecl {{.*}} <line:11:1, line:12:22> col:6 imported in mod hidden invalid cos
// CHECK-NEXT: | |-NonTypeTemplateParmDecl {{.*}} <line:11:10, col:34> col:34 imported in mod hidden referenced invalid 'ReferenceOf<angle> auto' depth 0 index 0 R
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} <col:37, col:46> col:46 imported in mod hidden referenced typename depth 0 index 1 Rep
// CHECK-NEXT: | |-RequiresExpr {{.*}} <col:60, col:84> 'bool'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} <col:69, col:73> col:73 imported in mod hidden referenced v 'Rep'
// CHECK-NEXT: | | `-SimpleRequirement {{.*}} dependent
// CHECK-NEXT: | |   `-CallExpr {{.*}} <col:77, col:82> '<dependent type>'
// CHECK-NEXT: | |     |-UnresolvedLookupExpr {{.*}} <col:77> '<overloaded function type>' lvalue (ADL) = 'cos' empty
// CHECK-NEXT: | |     `-DeclRefExpr {{.*}} <col:81> 'Rep' lvalue ParmVar {{.*}} 'v' 'Rep' non_odr_use_unevaluated
// CHECK-NEXT: | `-FunctionDecl {{.*}} <line:12:1, col:22> col:6 imported in mod hidden cos 'auto (const Rep &)'
// CHECK-NEXT: |   `-ParmVarDecl {{.*}} <col:10, col:21> col:21 imported in mod hidden q 'const Rep &'

// CHECK:      |-FunctionTemplateDecl {{.*}} <line:15:1, line:16:22> col:6 imported in mod hidden invalid tan
// CHECK-NEXT: | |-NonTypeTemplateParmDecl {{.*}} <line:15:10, col:34> col:34 imported in mod hidden referenced invalid 'ReferenceOf<angle> auto' depth 0 index 0 R
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} <col:37, col:46> col:46 imported in mod hidden referenced typename depth 0 index 1 Rep
// CHECK-NEXT: | |-RequiresExpr {{.*}} <col:60, col:84> 'bool'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} <col:69, col:73> col:73 imported in mod hidden referenced v 'Rep'
// CHECK-NEXT: | | `-SimpleRequirement {{.*}} dependent
// CHECK-NEXT: | |   `-CallExpr {{.*}} <col:77, col:82> '<dependent type>'
// CHECK-NEXT: | |     |-UnresolvedLookupExpr {{.*}} <col:77> '<overloaded function type>' lvalue (ADL) = 'tan' empty
// CHECK-NEXT: | |     `-DeclRefExpr {{.*}} <col:81> 'Rep' lvalue ParmVar {{.*}} 'v' 'Rep' non_odr_use_unevaluated
// CHECK-NEXT: | `-FunctionDecl {{.*}} <line:16:1, col:22> col:6 imported in mod hidden tan 'auto (const Rep &)'
// CHECK-NEXT: |   `-ParmVarDecl {{.*}} <col:10, col:21> col:21 imported in mod hidden q 'const Rep &'
