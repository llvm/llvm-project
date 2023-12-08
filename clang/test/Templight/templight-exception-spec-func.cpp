// RUN: %clang_cc1 -templight-dump -std=c++14 %s 2>&1 | FileCheck %s
template <bool B>
void f() noexcept(B) {}

int main()
{

// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+f$}}
// CHECK: {{^kind:[ ]+ExplicitTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-exception-spec-func.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-exception-spec-func.cpp:72:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+f$}}
// CHECK: {{^kind:[ ]+ExplicitTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-exception-spec-func.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-exception-spec-func.cpp:72:3'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+f$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-exception-spec-func.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-exception-spec-func.cpp:72:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+f$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-exception-spec-func.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-exception-spec-func.cpp:72:3'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'f<false>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-exception-spec-func.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-exception-spec-func.cpp:72:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'f<false>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-exception-spec-func.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-exception-spec-func.cpp:72:3'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'f<false>'$}}
// CHECK: {{^kind:[ ]+ExceptionSpecInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-exception-spec-func.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-exception-spec-func.cpp:72:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'f<false>'$}}
// CHECK: {{^kind:[ ]+ExceptionSpecInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-exception-spec-func.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-exception-spec-func.cpp:72:3'$}}
//
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'f<false>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-exception-spec-func.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-exception-spec-func.cpp:72:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'f<false>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-exception-spec-func.cpp:3:6'}}
// CHECK: {{^poi:[ ]+'.*templight-exception-spec-func.cpp:72:3'$}}
  f<false>();
}
