#include "redeclarations.h"

class A
{
};

// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 2 all -I%S/Inputs -fno-delayed-template-parsing -fno-ms-compatibility -fno-ms-extensions %s | FileCheck %s

// CHECK: redeclarations.h:1:7: ClassDecl=X:1:7 (Definition) Extent=[1:1 - 4:2]
// CHECK: redeclarations.h:8:7: ClassTemplate=B:8:7 (Definition) Extent=[7:1 - 10:2]
// CHECK: redeclarations.h:7:20: TemplateTypeParameter=T1:7:20 (Definition) Extent=[7:11 - 7:22]
// CHECK: redeclarations.h:7:33: TemplateTypeParameter=T2:7:33 (Definition) Extent=[7:24 - 7:35]
// CHECK: redeclarations.h:13:8: ClassTemplate=C:13:8 (Definition) Extent=[12:1 - 15:2]
// CHECK: redeclarations.h:12:17: TemplateTypeParameter=T:12:17 (Definition) Extent=[12:11 - 12:18]
// CHECK: redeclarations.h:17:7: ClassDecl=D:17:7 (Definition) Extent=[17:1 - 21:2]
// CHECK: redeclarations.h:19:16: ClassDecl=A:19:16 Extent=[19:10 - 19:17]
// CHECK: redeclarations.h:19:19: FieldDecl=x:19:19 (Definition) Extent=[19:5 - 19:20]
// CHECK: redeclarations.h:19:5: TemplateRef=B:8:7 Extent=[19:5 - 19:6]
// CHECK: redeclarations.h:19:7: TypeRef=class D:17:7 Extent=[19:7 - 19:8]
// CHECK: redeclarations.h:19:16: TypeRef=class A:3:7 Extent=[19:16 - 19:17]
// CHECK: redeclarations.cpp:3:7: ClassDecl=A:3:7 (Definition) Extent=[3:1 - 5:2]
