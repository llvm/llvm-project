template<typename ...Args>
int f(Args ...args) {
  return sizeof...(args) + sizeof...(Args);
}

void test() {
  int a;
  decltype(a) b;

  typedef int Integer;
  typedef float Float;
  typedef bool Bool;
  bool b2 = __is_trivially_constructible(Integer, Float, Bool);
}

// RUN: c-index-test -test-annotate-tokens=%s:1:1:5:1 -fno-delayed-template-parsing -std=c++11 %s | FileCheck %s
// CHECK: Identifier: "args" [3:20 - 3:24] SizeOfPackExpr=args:2:15
// CHECK: Identifier: "Args" [3:38 - 3:42] TypeRef=Args:1:22

// RUN: c-index-test -test-annotate-tokens=%s:8:1:9:1 -std=c++11 %s | FileCheck -check-prefix=CHECK-DECLTYPE %s
// CHECK-DECLTYPE: Identifier: "a" [8:12 - 8:13] DeclRefExpr=a:7:7

// RUN: c-index-test -test-annotate-tokens=%s:13:1:14:1 -std=c++11 %s | FileCheck -check-prefix=CHECK-TRAIT %s
// CHECK-TRAIT: Identifier: "Integer" [13:42 - 13:49] TypeRef=Integer:10:15
// CHECK-TRAIT: Identifier: "Float" [13:51 - 13:56] TypeRef=Float:11:17
// CHECK-TRAIT: Identifier: "Bool" [13:58 - 13:62] TypeRef=Bool:12:16

