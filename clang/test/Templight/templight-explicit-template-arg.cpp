// RUN: %clang_cc1 -templight-dump %s 2>&1 | FileCheck %s
template <class T>
void f(){}

int main()
{
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+f$}}
// CHECK: {{^kind:[ ]+ExplicitTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-explicit-template-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-explicit-template-arg.cpp:58:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+f$}}
// CHECK: {{^kind:[ ]+ExplicitTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-explicit-template-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-explicit-template-arg.cpp:58:3'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+f$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-explicit-template-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-explicit-template-arg.cpp:58:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+f$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-explicit-template-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-explicit-template-arg.cpp:58:3'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'f<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-explicit-template-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-explicit-template-arg.cpp:58:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'f<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-explicit-template-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-explicit-template-arg.cpp:58:3'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'f<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-explicit-template-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-explicit-template-arg.cpp:58:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'f<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-explicit-template-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-explicit-template-arg.cpp:58:3'$}}
  f<int>();
}
