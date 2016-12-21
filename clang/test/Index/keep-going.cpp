#include "missing1.h"

template<class T>
class A { T a; };

class B : public A<int> { };

#include "missing2.h"

class C : public A<float> { };

// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_KEEP_GOING=1 c-index-test -test-print-type %s -std=c++03 2> %t.stderr.txt  | FileCheck %s
// RUN: FileCheck -check-prefix CHECK-DIAG %s < %t.stderr.txt

// CHECK: inclusion directive=missing1.h ((null)) [type=] [typekind=Invalid] [isPOD=0]
// CHECK: inclusion directive=missing2.h ((null)) [type=] [typekind=Invalid] [isPOD=0]
// CHECK: ClassTemplate=A:4:7 (Definition) [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TemplateTypeParameter=T:3:16 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: FieldDecl=a:4:13 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: TypeRef=T:3:16 [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: ClassDecl=B:6:7 (Definition) [type=B] [typekind=Record] [isPOD=0]
// CHECK: C++ base class specifier=A<int>:4:7 [access=public isVirtual=false] [type=A<int>] [typekind=Unexposed] [templateargs/1= [type=int] [typekind=Int]] [canonicaltype=A<int>] [canonicaltypekind=Record] [canonicaltemplateargs/1= [type=int] [typekind=Int]] [isPOD=0] [nbFields=1]
// CHECK: TemplateRef=A:4:7 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: ClassDecl=C:10:7 (Definition) [type=C] [typekind=Record] [isPOD=0]
// CHECK: C++ base class specifier=A<float>:4:7 [access=public isVirtual=false] [type=A<float>] [typekind=Unexposed] [templateargs/1= [type=float] [typekind=Float]] [canonicaltype=A<float>] [canonicaltypekind=Record] [canonicaltemplateargs/1= [type=float] [typekind=Float]] [isPOD=0] [nbFields=1]
// CHECK: TemplateRef=A:4:7 [type=] [typekind=Invalid] [isPOD=0]

// CHECK-DIAG: keep-going.cpp:1:10: error: 'missing1.h' file not found
// CHECK-DIAG: keep-going.cpp:8:10: error: 'missing2.h' file not found
