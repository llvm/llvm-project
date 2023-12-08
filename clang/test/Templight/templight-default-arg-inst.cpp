// RUN: %clang_cc1 -templight-dump %s 2>&1 | FileCheck %s
template<class T, class U = T>
class A {};

// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A::U'$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-arg-inst.cpp:2:25'}}
// CHECK: {{^poi:[ ]+'.*templight-default-arg-inst.cpp:82:1'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A::U'$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-arg-inst.cpp:2:25'}}
// CHECK: {{^poi:[ ]+'.*templight-default-arg-inst.cpp:82:1'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A::U'$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentChecking$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-arg-inst.cpp:2:25'}}
// CHECK: {{^poi:[ ]+'.*templight-default-arg-inst.cpp:82:6'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A::U'$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentChecking$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-arg-inst.cpp:2:25'}}
// CHECK: {{^poi:[ ]+'.*templight-default-arg-inst.cpp:82:6'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int, int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-arg-inst.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-arg-inst.cpp:82:8'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int, int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-arg-inst.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-arg-inst.cpp:82:8'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int, int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-arg-inst.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-arg-inst.cpp:82:8'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int, int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-arg-inst.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-arg-inst.cpp:82:8'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int, int>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-arg-inst.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-arg-inst.cpp:82:8'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int, int>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-arg-inst.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-arg-inst.cpp:82:8'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int, int>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-default-arg-inst.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-arg-inst.cpp:82:8'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'A<int, int>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-default-arg-inst.cpp:3:7'}}
// CHECK: {{^poi:[ ]+'.*templight-default-arg-inst.cpp:82:8'$}}
A<int> a;
