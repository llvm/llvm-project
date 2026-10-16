// Test -mloadtime-comment-vars= IR output across C++20 module boundaries. The
// diagnostics for the variables that are not preserved are covered by
// clang/test/Sema/loadtime-comment-vars-modules.cpp and are silenced here
// with -Wno-loadtime-comment-var. The scenarios are named by their FileCheck
// prefixes:
//
//   MOD       — the module unit is built to a BMI with the option and then
//               compiled to IR from the BMI (the two-phase flow build systems
//               use). Exported, module-linkage, and internal-linkage
//               variables are all preserved: the implicit attribute is
//               serialized, and the internal variable reaches CodeGen via
//               the module-initializer list. The same IR is produced whether
//               or not the option is repeated on the codegen step: the BMI
//               already records the result. The name-matched inline variable
//               is not preserved and, being unreferenced, is not emitted.
//   NOOPT +   — the same module unit built to a BMI without the option and
//   NOOPTNOT    compiled to IR with it: nothing is preserved. The option
//               applies to the compilation of the module unit itself, where
//               semantic analysis runs.
//   IMPORT +  — an importing TU that names the module-owned variable in the
//   IMPORTNOT   option and references it: the variable is defined in the
//               module unit, not here, so it is only declared here and is not
//               preserved, and the module-internal variable does not leak
//               into the importer. The inline variable it references is
//               re-emitted here as usual for inline variables, and that copy
//               is not preserved either. The
//               name-matched specializations it instantiates from the
//               imported templates (a variable template and a static data
//               member of a class template) are emitted but not preserved.
//   GMF +     — a module unit whose global module fragment includes a header
//   GMFIMPORT   defining internal-linkage variables, built to a BMI with the
//               option and then compiled to IR from the BMI: the header's
//               variables are preserved in the module unit's object file.
//               An importing TU does not emit them.
//   HUIMPORT  — the same header built as a header unit: a header unit has no
//               object file of its own, so the name-matched variables are
//               not preserved, and an importing TU that emits one does not
//               preserve it either.
//
//   Source      IR symbol        Expected treatment
//   ------      ---------        ------------------
//   ver         _ZW1M3ver        exported: preserved when the module unit is
//                                compiled with the option
//   build       _ZW1M5build      module linkage: preserved likewise
//   priv        _ZL4priv         internal linkage: preserved likewise; never
//                                emitted by an importing TU
//   iv          _ZW1M2iv         exported inline: preserved neither in the
//                                module unit nor in an importer that
//                                references it
//   vt<int>     _ZW1M2vtIiE      instantiated in the importer: not preserved
//   S<int>::m   _ZNW1M1SIiE1mE   instantiated in the importer: not preserved
//   hdrid       _ZL5hdrid        defined in ident.h: preserved by a module unit
//   hdrptr      _ZL6hdrptr       that includes the header in its global module
//                                fragment; not preserved in a header unit

// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix -Wno-loadtime-comment-var \
// RUN:   -mloadtime-comment-vars=_ZW1M3ver,_ZW1M5build,_ZL4priv,_ZW1M2iv \
// RUN:   -emit-module-interface %t/m.cppm -o %t/m.pcm
// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=_ZW1M3ver,_ZW1M5build,_ZL4priv,_ZW1M2iv \
// RUN:   -emit-llvm %t/m.pcm -o - | FileCheck %s --check-prefix=MOD \
// RUN:   --implicit-check-not=@_ZW1M2iv
// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -emit-llvm %t/m.pcm -o - | FileCheck %s --check-prefix=MOD \
// RUN:   --implicit-check-not=@_ZW1M2iv

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -emit-module-interface %t/m.cppm -o %t/m-noopt.pcm
// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=_ZW1M3ver,_ZW1M5build,_ZL4priv,_ZW1M2iv \
// RUN:   -emit-llvm %t/m-noopt.pcm -o %t/m-noopt.ll
// RUN: FileCheck %s --check-prefix=NOOPT < %t/m-noopt.ll
// RUN: FileCheck %s --check-prefix=NOOPTNOT < %t/m-noopt.ll

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix -Wno-loadtime-comment-var \
// RUN:   -fmodule-file=M=%t/m.pcm \
// RUN:   -mloadtime-comment-vars=_ZW1M3ver,_ZW1M2vtIiE,_ZNW1M1SIiE1mE \
// RUN:   -emit-llvm %t/use.cpp -o %t/use.ll
// RUN: FileCheck %s --check-prefix=IMPORT < %t/use.ll
// RUN: FileCheck %s --check-prefix=IMPORTNOT < %t/use.ll

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=_ZL5hdrid,_ZL6hdrptr \
// RUN:   -emit-module-interface %t/gmf.cppm -o %t/gmf.pcm
// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -emit-llvm %t/gmf.pcm -o - | FileCheck %s --check-prefix=GMF
// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -fmodule-file=G=%t/gmf.pcm -mloadtime-comment-vars=_ZL5hdrid,_ZL6hdrptr \
// RUN:   -emit-llvm %t/use-gmf.cpp -o - | FileCheck %s --check-prefix=GMFIMPORT

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix -Wno-loadtime-comment-var \
// RUN:   -mloadtime-comment-vars=_ZL5hdrid,_ZL6hdrptr \
// RUN:   -emit-header-unit -xc++-user-header %t/ident.h -o %t/ident.pcm
// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -Wno-experimental-header-units -fmodule-file=%t/ident.pcm \
// RUN:   -mloadtime-comment-vars=_ZL5hdrid,_ZL6hdrptr \
// RUN:   -emit-llvm %t/use-hu.cpp -o - | FileCheck %s --check-prefix=HUIMPORT

// The three non-inline variables carry the metadata and are kept in
// llvm.compiler.used when the module unit is compiled from a BMI that was
// built with the option. The inline variable is not among them.
// MOD-DAG: @_ZW1M3ver = global [16 x i8] c"@(#) module ver\00", align 1, !loadtime_comment ![[MD:[0-9]+]]
// MOD-DAG: @_ZW1M5build = global [18 x i8] c"@(#) module build\00", align 1, !loadtime_comment ![[MD]]
// MOD-DAG: @_ZL4priv = internal global [17 x i8] c"@(#) module priv\00", align 1, !loadtime_comment ![[MD]]
// MOD-DAG: @llvm.compiler.used = appending global [3 x ptr]

// Without the option at BMI-build time nothing is preserved: the exported
// variable and the non-exported module-linkage variable are defined as
// ordinary globals (the {{$}} anchor proves no metadata), and the
// unreferenced internal and inline variables are not emitted at all.
// NOOPT-DAG: @_ZW1M3ver = global [16 x i8] c"@(#) module ver\00", align 1{{$}}
// NOOPT-DAG: @_ZW1M5build = global [18 x i8] c"@(#) module build\00", align 1{{$}}
// NOOPTNOT-NOT: !loadtime_comment
// NOOPTNOT-NOT: @llvm.compiler.used
// NOOPTNOT-NOT: @_ZL4priv
// NOOPTNOT-NOT: @_ZW1M2iv

// The importer instantiates the variable template and the static data member
// of the class template, both of which it names, and re-emits the inline
// variable it references (ordinary linkonce_odr definitions, no metadata —
// the {{$}} anchor proves it), so nothing is preserved here. The module-owned
// non-inline variable it names and references is only declared, without
// metadata; the other module-owned variable and the module-internal one are
// not emitted.
// IMPORT-DAG: @_ZW1M3ver = external global [16 x i8], align 1{{$}}
// IMPORT-DAG: @_ZW1M2vtIiE = linkonce_odr global ptr @{{.*}}, align 8{{$}}
// IMPORT-DAG: @_ZNW1M1SIiE1mE = linkonce_odr global ptr @{{.*}}, align 8{{$}}
// IMPORT-DAG: @_ZW1M2iv = linkonce_odr global ptr @{{.*}}, align 8{{$}}
// IMPORTNOT-NOT: !loadtime_comment
// IMPORTNOT-NOT: @llvm.compiler.used
// IMPORTNOT-NOT: @_ZW1M5build
// IMPORTNOT-NOT: @_ZL4priv

// The internal-linkage variables from the header included in the global
// module fragment carry the metadata and are kept in llvm.compiler.used when
// the module unit is compiled from its BMI.
// GMF-DAG: @_ZL5hdrid = internal global [15 x i8] c"@(#) header id\00", align 1, !loadtime_comment ![[GMD:[0-9]+]]
// GMF-DAG: @_ZL6hdrptr = internal global ptr @{{.*}}, align 8, !loadtime_comment ![[GMD]]
// GMF-DAG: @llvm.compiler.used = appending global [2 x ptr]

// An importer of that module emits neither variable.
// GMFIMPORT-NOT: @_ZL5hdrid
// GMFIMPORT-NOT: @_ZL6hdrptr
// GMFIMPORT-NOT: !loadtime_comment

// An importer of the header unit emits the variable it references as an
// ordinary definition (the {{$}} anchor proves no metadata) and preserves
// nothing.
// HUIMPORT: @_ZL6hdrptr = internal global ptr @{{.*}}, align 8{{$}}
// HUIMPORT-NOT: !loadtime_comment
// HUIMPORT-NOT: @llvm.compiler.used

//--- m.cppm
export module M;
export char ver[] = "@(#) module ver";
char build[] = "@(#) module build";
static char priv[] = "@(#) module priv";
export inline const char *iv = "@(#) module inline";
export template <class T> const char *vt = "@(#) vt";
export template <class T> struct S { static const char *m; };
template <class T> const char *S<T>::m = "@(#) sdm";

//--- use.cpp
import M;
const char *u0 = ver;
const char *u1 = vt<int>;
const char *u2 = S<int>::m;
const char *u3 = iv;

//--- ident.h
#ifndef IDENT_H
#define IDENT_H
static char hdrid[] = "@(#) header id";
static const char *hdrptr = "@(#) header ptr";
#endif

//--- gmf.cppm
module;
#include "ident.h"
export module G;

//--- use-gmf.cpp
import G;
void use_g() {}

//--- use-hu.cpp
import "ident.h";
const char *use_hu() { return hdrptr; }
