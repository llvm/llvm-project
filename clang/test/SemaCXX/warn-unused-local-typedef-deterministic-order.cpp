// Verify that -Wunused-local-typedef diagnostics are emitted in a deterministic
// (source) order even when a scope contains many unused local typedefs. The
// candidates are collected while iterating a Scope's DeclsInScope, which is a
// SmallPtrSet, so without sorting the order would depend on pointer values and
// vary across runs.
//
// RUN: %clang_cc1 %s -fsyntax-only -Wunused-local-typedef 2>&1 | FileCheck %s

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

// CHECK: warning: unused typedef 't000'
// CHECK: warning: unused typedef 't001'
// CHECK: warning: unused typedef 't002'
// CHECK: warning: unused typedef 't003'
// CHECK: warning: unused typedef 't004'
// CHECK: warning: unused typedef 't005'
// CHECK: warning: unused typedef 't006'
// CHECK: warning: unused typedef 't007'
// CHECK: warning: unused typedef 't008'
// CHECK: warning: unused typedef 't009'
// CHECK: warning: unused typedef 't010'
// CHECK: warning: unused typedef 't011'
// CHECK: warning: unused typedef 't012'
// CHECK: warning: unused typedef 't013'
// CHECK: warning: unused typedef 't014'
// CHECK: warning: unused typedef 't015'
// CHECK: warning: unused typedef 't016'
// CHECK: warning: unused typedef 't017'
// CHECK: warning: unused typedef 't018'
// CHECK: warning: unused typedef 't019'
// CHECK: warning: unused typedef 't020'
// CHECK: warning: unused typedef 't021'
// CHECK: warning: unused typedef 't022'
// CHECK: warning: unused typedef 't023'
// CHECK: warning: unused typedef 't024'
// CHECK: warning: unused typedef 't025'
// CHECK: warning: unused typedef 't026'
// CHECK: warning: unused typedef 't027'
// CHECK: warning: unused typedef 't028'
// CHECK: warning: unused typedef 't029'
// CHECK: warning: unused typedef 't030'
// CHECK: warning: unused typedef 't031'
// CHECK: warning: unused typedef 't032'
// CHECK: warning: unused typedef 't033'
// CHECK: warning: unused typedef 't034'
// CHECK: warning: unused typedef 't035'
// CHECK: warning: unused typedef 't036'
// CHECK: warning: unused typedef 't037'
// CHECK: warning: unused typedef 't038'
// CHECK: warning: unused typedef 't039'
// CHECK: warning: unused typedef 't040'
// CHECK: warning: unused typedef 't041'
// CHECK: warning: unused typedef 't042'
// CHECK: warning: unused typedef 't043'
// CHECK: warning: unused typedef 't044'
// CHECK: warning: unused typedef 't045'
// CHECK: warning: unused typedef 't046'
// CHECK: warning: unused typedef 't047'
// CHECK: warning: unused typedef 't048'
// CHECK: warning: unused typedef 't049'
// CHECK: warning: unused typedef 't050'
// CHECK: warning: unused typedef 't051'
// CHECK: warning: unused typedef 't052'
// CHECK: warning: unused typedef 't053'
// CHECK: warning: unused typedef 't054'
// CHECK: warning: unused typedef 't055'
// CHECK: warning: unused typedef 't056'
// CHECK: warning: unused typedef 't057'
// CHECK: warning: unused typedef 't058'
// CHECK: warning: unused typedef 't059'
// CHECK: warning: unused typedef 't060'
// CHECK: warning: unused typedef 't061'
// CHECK: warning: unused typedef 't062'
// CHECK: warning: unused typedef 't063'
// CHECK: warning: unused typedef 't064'
// CHECK: warning: unused typedef 't065'
// CHECK: warning: unused typedef 't066'
// CHECK: warning: unused typedef 't067'
// CHECK: warning: unused typedef 't068'
// CHECK: warning: unused typedef 't069'
// CHECK: warning: unused typedef 't070'
// CHECK: warning: unused typedef 't071'
// CHECK: warning: unused typedef 't072'
// CHECK: warning: unused typedef 't073'
// CHECK: warning: unused typedef 't074'
// CHECK: warning: unused typedef 't075'
// CHECK: warning: unused typedef 't076'
// CHECK: warning: unused typedef 't077'
// CHECK: warning: unused typedef 't078'
// CHECK: warning: unused typedef 't079'
// CHECK: warning: unused typedef 't080'
// CHECK: warning: unused typedef 't081'
// CHECK: warning: unused typedef 't082'
// CHECK: warning: unused typedef 't083'
// CHECK: warning: unused typedef 't084'
// CHECK: warning: unused typedef 't085'
// CHECK: warning: unused typedef 't086'
// CHECK: warning: unused typedef 't087'
// CHECK: warning: unused typedef 't088'
// CHECK: warning: unused typedef 't089'
// CHECK: warning: unused typedef 't090'
// CHECK: warning: unused typedef 't091'
// CHECK: warning: unused typedef 't092'
// CHECK: warning: unused typedef 't093'
// CHECK: warning: unused typedef 't094'
// CHECK: warning: unused typedef 't095'
// CHECK: warning: unused typedef 't096'
// CHECK: warning: unused typedef 't097'
// CHECK: warning: unused typedef 't098'
// CHECK: warning: unused typedef 't099'
