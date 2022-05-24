// RUN: c-index-test -index-file %s -std=gnu++20 | FileCheck %s
// UNSUPPORTED: aix
template <typename T>
concept LargeType = sizeof(T) > 8;
// CHECK: [indexDeclaration]: kind: concept | name: LargeType | USR: c:@CT@LargeType | lang: C | cursor: ConceptDecl=LargeType:[[@LINE-1]]:9 (Definition) | loc: [[@LINE-1]]:9 | semantic-container: [TU] | lexical-container: [TU] | isRedecl: 0 | isDef: 1 | isContainer: 0 | isImplicit: 0

template <LargeType T>
// CHECK: [indexEntityReference]: kind: concept | name: LargeType | USR: c:@CT@LargeType | lang: C | cursor: TemplateRef=LargeType:4:9 | loc: [[@LINE-1]]:11 | <parent>:: kind: function-template | name: f | USR: c:@FT@>1#Tf#v# | lang: C++ | container: [<<NULL>>] | refkind: direct | role: ref
void f();
