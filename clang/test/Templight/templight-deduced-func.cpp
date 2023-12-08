// RUN: %clang_cc1 -templight-dump %s 2>&1 | FileCheck %s

template <class T>
int foo(T){return 0;}

// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+foo$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-deduced-func.cpp:4:5'}}
// CHECK: {{^poi:[ ]+'.*templight-deduced-func.cpp:44:12'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+foo$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-deduced-func.cpp:4:5'}}
// CHECK: {{^poi:[ ]+'.*templight-deduced-func.cpp:44:12'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'foo<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-deduced-func.cpp:4:5'}}
// CHECK: {{^poi:[ ]+'.*templight-deduced-func.cpp:44:12'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'foo<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-deduced-func.cpp:4:5'}}
// CHECK: {{^poi:[ ]+'.*templight-deduced-func.cpp:44:12'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'foo<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-deduced-func.cpp:4:5'}}
// CHECK: {{^poi:[ ]+'.*templight-deduced-func.cpp:44:12'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'foo<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-deduced-func.cpp:4:5'}}
// CHECK: {{^poi:[ ]+'.*templight-deduced-func.cpp:44:12'$}}
int gvar = foo(0);
