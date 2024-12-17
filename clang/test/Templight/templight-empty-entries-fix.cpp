// RUN: %clang_cc1 -templight-dump -Wno-unused-value %s 2>&1 | FileCheck %s

void a() {
  [] {};
}

// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'\(lambda at .*templight-empty-entries-fix.cpp:4:3\)'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:4:3'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:4:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'\(lambda at .*templight-empty-entries-fix.cpp:4:3\)'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:4:3'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:4:3'$}}

template <int = 0> void a() { a(); }

// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+a$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:20:25'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:20:31'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+unnamed template non-type parameter 0 of a$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:20:15'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:20:25'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+unnamed template non-type parameter 0 of a$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:20:15'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:20:25'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+a$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:20:25'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:20:31'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'a<0>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:20:25'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:20:31'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'a<0>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:20:25'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:20:31'$}}

template <int> struct b { typedef int c; };
template <bool d = true, class = typename b<d>::c> void a() { a(); }

// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+a$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:60:57'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:63'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+d$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:60:16'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:57'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+d$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:60:16'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:57'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+unnamed template type parameter 1 of a$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:60:32'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:57'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'b<1>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:59:23'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:43'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'b<1>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:59:23'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:43'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'b<1>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:59:23'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:43'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'b<1>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:59:23'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:43'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'b<1>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:59:23'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:43'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'b<1>'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:59:23'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:43'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+unnamed template type parameter 1 of a$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:60:32'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:57'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+a$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:60:57'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:63'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'a<true, int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:60:57'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:63'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'a<true, int>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:60:57'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:63'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+a$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:20:25'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:63'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+unnamed template non-type parameter 0 of a$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:20:15'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:20:25'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+unnamed template non-type parameter 0 of a$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:20:15'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:20:25'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+a$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:20:25'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:63'$}}

template <bool = true> void d(int = 0) { d(); }

// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+a$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:60:57'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:63'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+a$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:60:57'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:63'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+a$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:20:25'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:63'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+a$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:20:25'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:60:63'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+d$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:171:29'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:171:42'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+unnamed template non-type parameter 0 of d$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:171:16'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:171:29'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+unnamed template non-type parameter 0 of d$}}
// CHECK: {{^kind:[ ]+DefaultTemplateArgumentInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:171:16'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:171:29'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+d$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:171:29'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:171:42'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'d<true>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:171:29'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:171:42'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'d<true>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:171:29'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:171:42'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'unnamed function parameter 0 of d<true>'$}}
// CHECK: {{^kind:[ ]+DefaultFunctionArgumentInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:171:35'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:171:42'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'unnamed function parameter 0 of d<true>'$}}
// CHECK: {{^kind:[ ]+DefaultFunctionArgumentInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:171:35'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:171:42'$}}

void e() {
  struct {
  } f;
}

// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'\(unnamed struct at .*templight-empty-entries-fix.cpp:247:3\)'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:247:3'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:248:5'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'\(unnamed struct at .*templight-empty-entries-fix.cpp:247:3\)'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:247:3'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:248:5'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'\(unnamed struct at .*templight-empty-entries-fix.cpp:247:3\)'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:247:3'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:248:5'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'\(unnamed struct at .*templight-empty-entries-fix.cpp:247:3\)'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:247:3'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:248:5'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'\(unnamed struct at .*templight-empty-entries-fix.cpp:247:3\)'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:247:3'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:247:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'\(unnamed struct at .*templight-empty-entries-fix.cpp:247:3\)'$}}
// CHECK: {{^kind:[ ]+Memoization$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:247:3'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:247:3'$}}


template <template<typename> class>
void d();

template <typename T> struct C;

void foo() {
  d<C>();
}

// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+d$}}
// CHECK: {{^kind:[ ]+ExplicitTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:290:6'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:295:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+unnamed template template parameter 0 of d$}}
// CHECK: {{^kind:[ ]+PriorTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:289:35'$}}
// CHECK: {{^poi:[ ]+''$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+unnamed template template parameter 0 of d$}}
// CHECK: {{^kind:[ ]+PriorTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:289:35'$}}
// CHECK: {{^poi:[ ]+''$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+d$}}
// CHECK: {{^kind:[ ]+ExplicitTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:290:6'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:295:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+d$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:290:6'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:295:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+d$}}
// CHECK: {{^kind:[ ]+DeducedTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:290:6'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:295:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'d<C>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:290:6'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:295:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+'d<C>'$}}
// CHECK: {{^kind:[ ]+TemplateInstantiation$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:290:6'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:295:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+d$}}
// CHECK: {{^kind:[ ]+ExplicitTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+Begin$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:171:29'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:295:3'$}}
// CHECK-LABEL: {{^---$}}
// CHECK: {{^name:[ ]+d$}}
// CHECK: {{^kind:[ ]+ExplicitTemplateArgumentSubstitution$}}
// CHECK: {{^event:[ ]+End$}}
// CHECK: {{^orig:[ ]+'.*templight-empty-entries-fix.cpp:171:29'$}}
// CHECK: {{^poi:[ ]+'.*templight-empty-entries-fix.cpp:295:3'$}}
