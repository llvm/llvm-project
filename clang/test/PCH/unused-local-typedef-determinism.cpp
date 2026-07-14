// Check that emitting a PCH is deterministic even when a scope contains many
// unused local typedefs. These are collected into
// Sema::UnusedLocalTypedefNameCandidates while iterating a Scope's SmallPtrSet
// (a pointer-order, run-to-run unstable container) and are then serialized into
// the AST file, so without a stable order the two PCHs below would differ.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %clang_cc1 -x c++-header %s -emit-pch -o %t/a.pch
// RUN: %clang_cc1 -x c++-header %s -emit-pch -o %t/b.pch
// RUN: cmp %t/a.pch %t/b.pch

inline void f() {
  typedef int t00; typedef int t01; typedef int t02; typedef int t03;
  typedef int t04; typedef int t05; typedef int t06; typedef int t07;
  typedef int t08; typedef int t09; typedef int t10; typedef int t11;
  typedef int t12; typedef int t13; typedef int t14; typedef int t15;
  typedef int t16; typedef int t17; typedef int t18; typedef int t19;
  typedef int t20; typedef int t21; typedef int t22; typedef int t23;
  typedef int t24; typedef int t25; typedef int t26; typedef int t27;
  typedef int t28; typedef int t29; typedef int t30; typedef int t31;
  typedef int t32; typedef int t33; typedef int t34; typedef int t35;
  typedef int t36; typedef int t37; typedef int t38; typedef int t39;
  typedef int t40; typedef int t41; typedef int t42; typedef int t43;
  typedef int t44; typedef int t45; typedef int t46; typedef int t47;
  typedef int t48; typedef int t49;
}
