// RUN: env CINDEXTEST_SKIP_FUNCTION_BODIES=1 c-index-test -test-load-source all %s | FileCheck %s

class A {
  class B {};

public:
  A() {
    struct C {
      void d() {}
    };
  }

  typedef B E;
};

@interface F
- (void) G;
@end
@implementation F
- (void) G {
  typedef A H;
  class I {};
}
@end

void J() {
  class K {};
}

// CHECK: skip-function-bodies.mm:3:7: ClassDecl=A:3:7 (Definition) Extent=[3:1 - 14:2]
// CHECK: skip-function-bodies.mm:4:9: ClassDecl=B:4:9 (Definition) Extent=[4:3 - 4:13]
// CHECK: skip-function-bodies.mm:6:1: CXXAccessSpecifier=:6:1 (Definition) Extent=[6:1 - 6:8]
// CHECK: skip-function-bodies.mm:7:3: CXXConstructor=A:7:3 (default constructor) Extent=[7:3 - 7:6]
// CHECK-NOT: skip-function-bodies.mm:8:12: StructDecl=C:8:12 (Definition) Extent=[8:5 - 10:6]
// CHECK-NOT: skip-function-bodies.mm:9:12: CXXMethod=d:9:12 (Definition) Extent=[9:7 - 9:18]
// CHECK: skip-function-bodies.mm:13:13: TypedefDecl=E:13:13 (Definition) Extent=[13:3 - 13:14]
// CHECK: skip-function-bodies.mm:13:11: TypeRef=class A::B:4:9 Extent=[13:11 - 13:12]
// CHECK: skip-function-bodies.mm:16:12: ObjCInterfaceDecl=F:16:12 Extent=[16:1 - 18:5]
// CHECK: skip-function-bodies.mm:17:10: ObjCInstanceMethodDecl=G:17:10 Extent=[17:1 - 17:12]
// CHECK: skip-function-bodies.mm:19:17: ObjCImplementationDecl=F:19:17 (Definition) Extent=[19:1 - 24:2]
// CHECK: skip-function-bodies.mm:20:10: ObjCInstanceMethodDecl=G:20:10 Extent=[20:1 - 20:13]
// CHECK-NOT: skip-function-bodies.mm:21:13: TypedefDecl=H:21:13 (Definition) Extent=[21:3 - 21:14]
// CHECK-NOT: skip-function-bodies.mm:21:11: TypeRef=class A:3:7 Extent=[21:11 - 21:12]
// CHECK: skip-function-bodies.mm:26:6: FunctionDecl=J:26:6 Extent=[26:1 - 26:9]
// CHECK-NOT: skip-function-bodies.mm:27:9: ClassDecl=K:27:9 (Definition) Extent=[27:3 - 27:13]
