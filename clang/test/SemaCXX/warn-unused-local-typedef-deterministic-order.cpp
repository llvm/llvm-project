// Verify that -Wunused-local-typedef diagnostics are emitted in a deterministic
// (source) order even when a scope contains many unused local typedefs. The
// candidates are collected while iterating a Scope's DeclsInScope, which is a
// SmallPtrSet, so without sorting the order would depend on pointer values and
// vary across runs.
//
// RUN: %clang_cc1 %s -fsyntax-only -Wunused-local-typedef 2>&1 | FileCheck %s

inline void f() {
  // Enough typedefs to exceed the small storage of Scope::DeclSetTy.
  typedef int t01;
  typedef int t02;
  typedef int t03;
  typedef int t04;
  typedef int t05;
  typedef int t06;
  typedef int t07;
  typedef int t08;
  typedef int t09;
  typedef int t10;
  typedef int t11;
  typedef int t12;
  typedef int t13;
  typedef int t14;
  typedef int t15;
  typedef int t16;
  typedef int t17;
  typedef int t18;
  typedef int t19;
  typedef int t20;
  typedef int t21;
  typedef int t22;
  typedef int t23;
  typedef int t24;
  typedef int t25;
  typedef int t26;
  typedef int t27;
  typedef int t28;
  typedef int t29;
  typedef int t30;
  typedef int t31;
  typedef int t32;
  typedef int t33;
  typedef int t34;
  typedef int t35;
  typedef int t36;
  typedef int t37;
  typedef int t38;
  typedef int t39;
  typedef int t40;
}

// CHECK: warning: unused typedef 't01'
// CHECK: warning: unused typedef 't02'
// CHECK: warning: unused typedef 't03'
// CHECK: warning: unused typedef 't04'
// CHECK: warning: unused typedef 't05'
// CHECK: warning: unused typedef 't06'
// CHECK: warning: unused typedef 't07'
// CHECK: warning: unused typedef 't08'
// CHECK: warning: unused typedef 't09'
// CHECK: warning: unused typedef 't10'
// CHECK: warning: unused typedef 't11'
// CHECK: warning: unused typedef 't12'
// CHECK: warning: unused typedef 't13'
// CHECK: warning: unused typedef 't14'
// CHECK: warning: unused typedef 't15'
// CHECK: warning: unused typedef 't16'
// CHECK: warning: unused typedef 't17'
// CHECK: warning: unused typedef 't18'
// CHECK: warning: unused typedef 't19'
// CHECK: warning: unused typedef 't20'
// CHECK: warning: unused typedef 't21'
// CHECK: warning: unused typedef 't22'
// CHECK: warning: unused typedef 't23'
// CHECK: warning: unused typedef 't24'
// CHECK: warning: unused typedef 't25'
// CHECK: warning: unused typedef 't26'
// CHECK: warning: unused typedef 't27'
// CHECK: warning: unused typedef 't28'
// CHECK: warning: unused typedef 't29'
// CHECK: warning: unused typedef 't30'
// CHECK: warning: unused typedef 't31'
// CHECK: warning: unused typedef 't32'
// CHECK: warning: unused typedef 't33'
// CHECK: warning: unused typedef 't34'
// CHECK: warning: unused typedef 't35'
// CHECK: warning: unused typedef 't36'
// CHECK: warning: unused typedef 't37'
// CHECK: warning: unused typedef 't38'
// CHECK: warning: unused typedef 't39'
// CHECK: warning: unused typedef 't40'
