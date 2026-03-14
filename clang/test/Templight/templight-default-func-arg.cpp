// RUN: %clang_cc1 -std=c++14 -templight-dump %s 2>&1 | FileCheck %s
template <class T>
void foo(T b = 0) {};

int main()
{

// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+foo$}}
// CHECK: {{^kind:[ ]+ExplicitTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-func-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-default-func-arg.cpp:72:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+foo$}}
// CHECK: {{^kind:[ ]+ExplicitTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-func-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-default-func-arg.cpp:72:3'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+foo$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-func-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-default-func-arg.cpp:72:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+foo$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-func-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-default-func-arg.cpp:72:3'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'foo<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-func-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-default-func-arg.cpp:72:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'foo<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-func-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-default-func-arg.cpp:72:3'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+b$}}
// CHECK: {{^kind:[ ]+DefaultFunctionArgumentInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-func-arg.cpp:3:12'}}
// CHECK: {{^poi:[ ]+'.*templight-default-func-arg.cpp:72:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+b$}}
// CHECK: {{^kind:[ ]+DefaultFunctionArgumentInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-func-arg.cpp:3:12'}}
// CHECK: {{^poi:[ ]+'.*templight-default-func-arg.cpp:72:3'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'foo<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-func-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-default-func-arg.cpp:72:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'foo<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-func-arg.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-default-func-arg.cpp:72:3'$}}
  foo<int>();
}
