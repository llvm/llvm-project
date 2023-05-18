__attribute__((availability(macosx, introduced = 8.0)))
@interface C {
  int i0;
  int i1 __attribute__((availability(macosx, introduced = 9.0)));
}
@property int p0;
@property int p1 __attribute__((availability(macosx, introduced=9.0)));
- (void)m0;
- (void)m1 __attribute__((availability(macosx, introduced = 9.0)));
@end

@implementation C
- (void)m0 {
}
- (void)m1 {
}
@end

__attribute__((availability(macosx, introduced = 10.0)))
@interface C(Cat)
@property int p2;
@property int p3 __attribute__((availability(macosx, introduced=11.0)));
- (void)m2;
- (void)m3 __attribute__((availability(macosx, introduced = 11.0)));
@end

@implementation C(Cat)
- (void)m2 {
}
- (void)m3 {
}
@end

__attribute__((availability(macosx, introduced = 10.0)))
@protocol P
@property int p4;
@property int p5 __attribute__((availability(macosx, introduced=11.0)));
- (void)m4;
- (void)m5 __attribute__((availability(macosx, introduced = 11.0)));
@end

@interface C(Cat2)
@end

@implementation C(Cat2)
@end

// RUN: c-index-test -test-print-type --std=c++11 %s | FileCheck %s

// CHECK: ObjCInterfaceDecl=C:2:12  (macos, introduced=8.0)
// CHECK: ObjCIvarDecl=i0:3:7 (Definition)  (macos, introduced=8.0)
// CHECK: ObjCIvarDecl=i1:4:7 (Definition)  (macos, introduced=9.0)
// CHECK: ObjCPropertyDecl=p0:6:15  (macos, introduced=8.0)
// CHECK: ObjCPropertyDecl=p1:7:15  (macos, introduced=9.0)
// CHECK: ObjCInstanceMethodDecl=m0:8:9  (macos, introduced=8.0)
// CHECK: ObjCInstanceMethodDecl=m1:9:9  (macos, introduced=9.0)

// CHECK: ObjCImplementationDecl=C:12:17 (Definition)  (macos, introduced=8.0)
// CHECK: ObjCInstanceMethodDecl=m0:13:9 (Definition)  (macos, introduced=8.0)
// CHECK: ObjCInstanceMethodDecl=m1:15:9 (Definition)  (macos, introduced=9.0)

// CHECK: ObjCCategoryDecl=Cat:20:12  (macos, introduced=10.0)
// CHECK: ObjCPropertyDecl=p2:21:15  (macos, introduced=10.0)
// CHECK: ObjCPropertyDecl=p3:22:15  (macos, introduced=11.0)
// CHECK: ObjCInstanceMethodDecl=m2:23:9  (macos, introduced=10.0)
// CHECK: ObjCInstanceMethodDecl=m3:24:9  (macos, introduced=11.0)

// CHECK: ObjCCategoryImplDecl=Cat:27:17 (Definition)  (macos, introduced=10.0)
// CHECK: ObjCInstanceMethodDecl=m2:28:9 (Definition)  (macos, introduced=10.0)
// CHECK: ObjCInstanceMethodDecl=m3:30:9 (Definition)  (macos, introduced=11.0)

// CHECK: ObjCProtocolDecl=P:35:11 (Definition)  (macos, introduced=10.0)
// CHECK: ObjCPropertyDecl=p4:36:15  (macos, introduced=10.0)
// CHECK: ObjCPropertyDecl=p5:37:15  (macos, introduced=11.0)
// CHECK: ObjCInstanceMethodDecl=m4:38:9  (macos, introduced=10.0)
// CHECK: ObjCInstanceMethodDecl=m5:39:9  (macos, introduced=11.0)

// CHECK: ObjCCategoryDecl=Cat2:42:12  (macos, introduced=8.0)
// CHECK: ObjCCategoryImplDecl=Cat2:45:17 (Definition)  (macos, introduced=8.0)
