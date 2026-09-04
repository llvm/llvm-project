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
  // Enough typedefs to exceed the small storage of Scope::DeclSetTy.
  typedef int t000;
  typedef int t001;
  typedef int t002;
  typedef int t003;
  typedef int t004;
  typedef int t005;
  typedef int t006;
  typedef int t007;
  typedef int t008;
  typedef int t009;
  typedef int t010;
  typedef int t011;
  typedef int t012;
  typedef int t013;
  typedef int t014;
  typedef int t015;
  typedef int t016;
  typedef int t017;
  typedef int t018;
  typedef int t019;
  typedef int t020;
  typedef int t021;
  typedef int t022;
  typedef int t023;
  typedef int t024;
  typedef int t025;
  typedef int t026;
  typedef int t027;
  typedef int t028;
  typedef int t029;
  typedef int t030;
  typedef int t031;
  typedef int t032;
  typedef int t033;
  typedef int t034;
  typedef int t035;
  typedef int t036;
  typedef int t037;
  typedef int t038;
  typedef int t039;
  typedef int t040;
  typedef int t041;
  typedef int t042;
  typedef int t043;
  typedef int t044;
  typedef int t045;
  typedef int t046;
  typedef int t047;
  typedef int t048;
  typedef int t049;
  typedef int t050;
  typedef int t051;
  typedef int t052;
  typedef int t053;
  typedef int t054;
  typedef int t055;
  typedef int t056;
  typedef int t057;
  typedef int t058;
  typedef int t059;
  typedef int t060;
  typedef int t061;
  typedef int t062;
  typedef int t063;
  typedef int t064;
  typedef int t065;
  typedef int t066;
  typedef int t067;
  typedef int t068;
  typedef int t069;
  typedef int t070;
  typedef int t071;
  typedef int t072;
  typedef int t073;
  typedef int t074;
  typedef int t075;
  typedef int t076;
  typedef int t077;
  typedef int t078;
  typedef int t079;
  typedef int t080;
  typedef int t081;
  typedef int t082;
  typedef int t083;
  typedef int t084;
  typedef int t085;
  typedef int t086;
  typedef int t087;
  typedef int t088;
  typedef int t089;
  typedef int t090;
  typedef int t091;
  typedef int t092;
  typedef int t093;
  typedef int t094;
  typedef int t095;
  typedef int t096;
  typedef int t097;
  typedef int t098;
  typedef int t099;
}
