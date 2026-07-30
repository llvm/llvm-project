// Test -mloadtime-comment-vars= across a C++20 named-module boundary. Five
// scenarios, each named by its FileCheck or -verify prefix:
//
//   MOD       — the module unit is built to a BMI with the option and then
//               compiled to IR from the BMI (the two-phase flow build systems
//               use). Exported, module-linkage, and internal-linkage variables
//               are all preserved: the implicit attribute is serialized, and
//               the internal variable reaches CodeGen via the
//               module-initializer list.
//   NOOPT +   — the same module unit built to a BMI without the option and
//   NOOPTNOT    compiled to IR with it: nothing is preserved. The option
//               applies to the compilation of the module unit itself, where
//               semantic analysis runs.
//   IMPORT +  — an importing TU naming the module-owned variable: the
//   IMPORTNOT   variable is defined in the module unit, not here, so it is
//               neither re-emitted nor preserved here, and the module-internal
//               variable does not leak into the importer.
//   verify    — specializations instantiated here from the imported template
//               definitions are diagnosed in this TU, at the pattern location
//               in the module interface, with a note at the instantiation
//               point.
//
//   Source      IR symbol        Expected treatment
//   ------      ---------        ------------------
//   ver         _ZW1M3ver        exported: preserved when the module unit is
//                                compiled with the option
//   build       _ZW1M5build      module linkage: preserved likewise
//   priv        _ZL4priv         internal linkage: preserved likewise; never
//                                emitted by an importing TU
//   vt<int>     _ZW1M2vtIiE      instantiated in the importer: diagnosed there
//   S<int>::m   _ZNW1M1SIiE1mE   instantiated in the importer: diagnosed there

// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=_ZW1M3ver,_ZW1M5build,_ZL4priv \
// RUN:   -emit-module-interface %t/m.cppm -o %t/m.pcm
// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=_ZW1M3ver,_ZW1M5build,_ZL4priv \
// RUN:   -emit-llvm %t/m.pcm -o - | FileCheck %s --check-prefix=MOD

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -emit-module-interface %t/m.cppm -o %t/m-noopt.pcm
// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=_ZW1M3ver,_ZW1M5build,_ZL4priv \
// RUN:   -emit-llvm %t/m-noopt.pcm -o %t/m-noopt.ll
// RUN: FileCheck %s --check-prefix=NOOPT < %t/m-noopt.ll
// RUN: FileCheck %s --check-prefix=NOOPTNOT < %t/m-noopt.ll

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -fmodule-file=M=%t/m.pcm -mloadtime-comment-vars=_ZW1M3ver \
// RUN:   -emit-llvm %t/use.cpp -o %t/use.ll
// RUN: FileCheck %s --check-prefix=IMPORT < %t/use.ll
// RUN: FileCheck %s --check-prefix=IMPORTNOT < %t/use.ll

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -fmodule-file=M=%t/m.pcm \
// RUN:   -mloadtime-comment-vars=_ZW1M2vtIiE,_ZNW1M1SIiE1mE \
// RUN:   -fsyntax-only -verify %t/use.cpp

// All three variables carry the metadata and are kept in llvm.compiler.used
// when the module unit is compiled from its BMI with the option.
// MOD-DAG: @_ZW1M3ver = global [16 x i8] c"@(#) module ver\00", align 1, !loadtime_comment ![[MD:[0-9]+]]
// MOD-DAG: @_ZW1M5build = global [18 x i8] c"@(#) module build\00", align 1, !loadtime_comment ![[MD]]
// MOD-DAG: @_ZL4priv = internal global [17 x i8] c"@(#) module priv\00", align 1, !loadtime_comment ![[MD]]
// MOD-DAG: @llvm.compiler.used = appending global [3 x ptr]

// Without the option at BMI-build time nothing is preserved: the exported
// variable is an ordinary global (the {{$}} anchor proves no metadata), and
// the unreferenced internal variable is not emitted at all.
// NOOPT: @_ZW1M3ver = global [16 x i8] c"@(#) module ver\00", align 1{{$}}
// NOOPTNOT-NOT: !loadtime_comment
// NOOPTNOT-NOT: @llvm.compiler.used
// NOOPTNOT-NOT: @_ZL4priv

// The importer instantiates the templates (ordinary linkonce_odr definitions,
// no metadata — the {{$}} anchor proves it) but emits neither the module-owned
// variable it names nor the module-internal one.
// IMPORT: @_ZW1M2vtIiE = linkonce_odr global ptr @{{.*}}, align 8{{$}}
// IMPORTNOT-NOT: @_ZW1M3ver
// IMPORTNOT-NOT: @_ZW1M5build
// IMPORTNOT-NOT: @_ZL4priv
// IMPORTNOT-NOT: @llvm.compiler.used
// IMPORTNOT-NOT: !loadtime_comment

//--- m.cppm
export module M;
export char ver[] = "@(#) module ver";
char build[] = "@(#) module build";
static char priv[] = "@(#) module priv";
export template <class T> const char *vt = "@(#) vt";
export template <class T> struct S { static const char *m; };
template <class T> const char *S<T>::m = "@(#) sdm";
export inline void touch() {}

//--- use.cpp
import M;
const char *u1 = vt<int>;   // expected-note {{in instantiation of variable template specialization 'vt<int>' requested here}}
const char *u2 = S<int>::m; // expected-note {{in instantiation of static data member 'S<int>::m' requested here}}
// expected-warning@m.cppm:5 {{'vt<int>' named in '-mloadtime-comment-vars=' is a variable template specialization and will not be preserved}}
// expected-warning@m.cppm:7 {{'m' named in '-mloadtime-comment-vars=' is a static data member and will not be preserved}}
