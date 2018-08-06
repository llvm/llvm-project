// RUN: %clang_cc1 -templight-dump %s 2>&1 | FileCheck %s

template <class T>
struct foo {};

// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'foo<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-one-instantiation.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-one-instantiation.cpp:18:10'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'foo<int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-one-instantiation.cpp:4:8'}}
// CHECK: {{^poi:[ ]+'.*templight-one-instantiation.cpp:18:10'$}}
foo<int> x;
