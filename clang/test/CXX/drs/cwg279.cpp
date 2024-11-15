// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file --leading-lines %s %t
// RUN: %clang_cc1 -std=c++20 -pedantic-errors -fexceptions -fcxx-exceptions %t/cwg279_A.cppm -triple x86_64-unknown-unknown -emit-module-interface -o %t/cwg279_A.pcm
// RUN: %clang_cc1 -std=c++20 -verify=since-cxx20 -pedantic-errors -fexceptions -fcxx-exceptions -triple x86_64-unknown-unknown %t/cwg279.cpp -fmodule-file=cwg279_A=%t/cwg279_A.pcm
// RUN: %clang_cc1 -std=c++23 -pedantic-errors -fexceptions -fcxx-exceptions %t/cwg279_A.cppm -triple x86_64-unknown-unknown -emit-module-interface -o %t/cwg279_A.pcm
// RUN: %clang_cc1 -std=c++23 -verify=since-cxx20 -pedantic-errors -fexceptions -fcxx-exceptions -triple x86_64-unknown-unknown %t/cwg279.cpp -fmodule-file=cwg279_A=%t/cwg279_A.pcm
// RUN: %clang_cc1 -std=c++2c -pedantic-errors -fexceptions -fcxx-exceptions %t/cwg279_A.cppm -triple x86_64-unknown-unknown -emit-module-interface -o %t/cwg279_A.pcm
// RUN: %clang_cc1 -std=c++2c -verify=since-cxx20 -pedantic-errors -fexceptions -fcxx-exceptions -triple x86_64-unknown-unknown %t/cwg279.cpp -fmodule-file=cwg279_A=%t/cwg279_A.pcm

// cwg279: no

//--- cwg279_A.cppm
export module cwg279_A;

export {
struct S; // #cwg279-S
extern S *q; // #cwg279-q

struct S2 {}; // #cwg279-S2
extern S2 *q2; // #cwg279-q2

struct S3 {}; // #cwg279-S3
extern S3 *q3; // #cwg279-q3
} // export

//--- cwg279.cpp
import cwg279_A;

// FIXME: We should use markers instead. They are less fragile,
//        but -verify doesn't support them across modules yet.
// FIXME: This is well-formed. Previous "definition" is actually just a declaration.
typedef struct {} S;
// since-cxx20-error@-1 {{typedef redefinition with different types ('struct S' vs 'S')}}
//   since-cxx20-note@cwg279_A.cppm:17 {{previous definition is here}}
extern S *q;
// since-cxx20-error@-1 {{declaration of 'q' in the global module follows declaration in module cwg279_A}}
//   since-cxx20-note@cwg279_A.cppm:18 {{previous declaration is here}}

typedef struct {} S2;
// since-cxx20-error@-1 {{typedef redefinition with different types ('struct S2' vs 'S2')}}
//   since-cxx20-note@cwg279_A.cppm:20 {{previous definition is here}}
extern S2 *q2;
// since-cxx20-error@-1 {{declaration of 'q2' in the global module follows declaration in module cwg279_A}}
//   since-cxx20-note@cwg279_A.cppm:21 {{previous declaration is here}}

// FIXME: This is well-formed, because [basic.def.odr]/15 is satisfied.
struct S3 {};
// since-cxx20-error@-1 {{redefinition of 'S3'}}
//   since-cxx20-note@cwg279_A.cppm:23 {{previous definition is here}}
extern S3 *q3;
// since-cxx20-error@-1 {{declaration of 'q3' in the global module follows declaration in module cwg279_A}}
//   since-cxx20-note@cwg279_A.cppm:24 {{previous declaration is here}}
