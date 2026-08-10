@interface Foo
-(instancetype)init;
@end

// RUN: env CINDEXTEST_VISIT_IMPLICIT_ATTRIBUTES=1 c-index-test -test-print-decl-attributes %s -fobjc-runtime=macosx-10.7 -fobjc-arc | FileCheck %s
// CHECK: ObjCInstanceMethodDecl=init:2:16 attribute(ns_consumes_self)= attribute(ns_returns_retained)=
