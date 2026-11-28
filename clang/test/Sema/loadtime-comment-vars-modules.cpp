// Verify that -mloadtime-comment-vars= diagnoses the variables it cannot
// preserve across C++20 module boundaries. The IR produced for these cases is
// covered by clang/test/CodeGen/PowerPC/loadtime-comment-vars-modules.cpp.
//
// Two scenarios are covered, each with its own -verify prefix:
//
//   import — specializations instantiated in an importing TU from the
//            imported template definitions are diagnosed in that TU, at the
//            pattern location in the module interface, with a note at the
//            instantiation point.
//   hu     — name-matched variables defined in a header unit are diagnosed
//            when the header unit is built.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -emit-module-interface %t/m.cppm -o %t/m.pcm
// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -fmodule-file=M=%t/m.pcm \
// RUN:   -mloadtime-comment-vars=_ZW1M2vtIiE,_ZNW1M1SIiE1mE \
// RUN:   -fsyntax-only -verify=import %t/use.cpp

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=_ZL5hdrid,_ZL6hdrptr \
// RUN:   -emit-header-unit -xc++-user-header %t/ident.h -o %t/ident.pcm \
// RUN:   -verify=hu

//--- m.cppm
export module M;
export template <class T> const char *vt = "@(#) vt";
export template <class T> struct S { static const char *m; };
template <class T> const char *S<T>::m = "@(#) sdm";

//--- use.cpp
import M;
const char *u1 = vt<int>;   // import-note {{in instantiation of variable template specialization 'vt<int>' requested here}}
const char *u2 = S<int>::m; // import-note {{in instantiation of static data member 'S<int>::m' requested here}}
// import-warning@m.cppm:2 {{'vt<int>' named in '-mloadtime-comment-vars=' is a variable template specialization and will not be preserved}}
// import-warning@m.cppm:4 {{'m' named in '-mloadtime-comment-vars=' is a static data member and will not be preserved}}

//--- ident.h
#ifndef IDENT_H
#define IDENT_H
static char hdrid[] = "@(#) header id"; // hu-warning {{'hdrid' named in '-mloadtime-comment-vars=' is defined in a header unit and will not be preserved}}
static const char *hdrptr = "@(#) header ptr"; // hu-warning {{'hdrptr' named in '-mloadtime-comment-vars=' is defined in a header unit and will not be preserved}}
#endif
