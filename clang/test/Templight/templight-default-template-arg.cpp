// RUN: %clang_cc1 -templight-dump %s 2>&1 | FileCheck %s
template <class T = int>
class A {};

// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A::T'$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentChecking$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-template-arg.cpp:2:17'}}
// CHECK: {{^poi:[ ]+'.*templight-default-template-arg.cpp:69:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A::T'$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentChecking$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-template-arg.cpp:2:17'}}
// CHECK: {{^poi:[ ]+'.*templight-default-template-arg.cpp:69:3'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-template-arg.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-template-arg.cpp:69:5'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-template-arg.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-template-arg.cpp:69:5'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-template-arg.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-template-arg.cpp:69:5'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-template-arg.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-template-arg.cpp:69:5'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-template-arg.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-template-arg.cpp:69:5'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-template-arg.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-template-arg.cpp:69:5'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-template-arg.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-template-arg.cpp:69:5'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-template-arg.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-template-arg.cpp:69:5'$}}
A<> a;
