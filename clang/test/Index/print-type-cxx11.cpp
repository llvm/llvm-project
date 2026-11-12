struct RefQualifierTest {
  void f() & {};
  void f() && {};
};

// RUN: c-index-test -test-print-type -std=c++11 %s | FileCheck %s
// CHECK: CXXMethod=f:2:8 (Definition) [type=void () {{.*}}&] [typekind=FunctionProto] lvalue-ref-qualifier [resulttype=void] [resulttypekind=Void] [isPOD=0]
// CHECK: CXXMethod=f:3:8 (Definition) [type=void () {{.*}}&&] [typekind=FunctionProto] rvalue-ref-qualifier [resulttype=void] [resulttypekind=Void] [isPOD=0]
