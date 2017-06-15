@protocol Proto1

- (void)requiredInstanceMethod:(int)y;

+ (void)aRequiredInstanceMethod:(int (*)(void))function with:(Proto *)p;

@optional;
- (void)anOptionalMethod;

@end

@protocol Proto2

- (void)a;

+ (void)b;

@end

@protocol Proto3

- (void)otherProtocolMethod:(int (^)(id<Proto2>))takesBlock;

@end

@interface Base
@end

@interface I1 : Base<Proto3>

@end
// CHECK1: "- (void)otherProtocolMethod:(int (^)(id<Proto2>))takesBlock;\n\n" [[@LINE-1]]:1 -> [[@LINE-1]]:1
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:29:1 %s | FileCheck --check-prefix=CHECK1 %s

@implementation I1

@end
// CHECK2: "- (void)otherProtocolMethod:(int (^)(id<Proto2>))takesBlock { \n  <#code#>\n}\n\n" [[@LINE-1]]:1 -> [[@LINE-1]]:1
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:35:1 %s | FileCheck --check-prefix=CHECK2 %s

@interface I2 : I1<Proto1, Proto2>

@end
// CHECK3: "+ (void)aRequiredInstanceMethod:(int (*)(void))function with:(id)p;\n\n- (void)requiredInstanceMethod:(int)y;\n\n- (void)a;\n\n+ (void)b;\n\n- (void)otherProtocolMethod:(int (^)(id<Proto2>))takesBlock;\n\n" [[@LINE-1]]:1 -> [[@LINE-1]]:1
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:41:1 %s | FileCheck --check-prefix=CHECK3 %s

@implementation I2

@end
// CHECK4: "+ (void)aRequiredInstanceMethod:(int (*)(void))function with:(id)p { \n  <#code#>\n}\n\n- (void)requiredInstanceMethod:(int)y { \n  <#code#>\n}\n\n- (void)a { \n  <#code#>\n}\n\n+ (void)b { \n  <#code#>\n}\n\n- (void)otherProtocolMethod:(int (^)(id<Proto2>))takesBlock { \n  <#code#>\n}\n\n" [[@LINE-1]]:1 -> [[@LINE-1]]:1
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:47:1 %s | FileCheck --check-prefix=CHECK4 %s

@interface I1(Category) <Proto2>
@end
// CHECK5: "- (void)a;\n\n+ (void)b;\n\n" [[@LINE-1]]:1 -> [[@LINE-1]]:1

@implementation I1(Category)
@end
// CHECK5-NEXT: "- (void)a { \n  <#code#>\n}\n\n+ (void)b { \n  <#code#>\n}\n\n" [[@LINE-1]]:1 -> [[@LINE-1]]:1
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:53:1 -at=%s:57:1 %s | FileCheck --check-prefix=CHECK5 %s

@interface I3 : I1<Proto1, Proto2>

- (void)requiredInstanceMethod:(int)y;

+ (void)b;

#ifdef HAS_OTHER
- (void) otherProtocolMethod:(int (^)(id<Proto2>))takesBlock;
#endif

@end
// CHECK6: "\n\n+ (void)aRequiredInstanceMethod:(int (*)(void))function with:(id)p;\n" [[@LINE-9]]:39 -> [[@LINE-9]]:39
// CHECK6-NEXT: "\n\n- (void)a;\n" [[@LINE-8]]:11 -> [[@LINE-8]]:11
// CHECK6-NEXT: "- (void)otherProtocolMethod:(int (^)(id<Proto2>))takesBlock;\n\n" [[@LINE-3]]:1 -> [[@LINE-3]]:1
// CHECK7: "\n\n+ (void)aRequiredInstanceMethod:(int (*)(void))function with:(id)p;\n" [[@LINE-12]]:39 -> [[@LINE-12]]:39
// CHECK7-NEXT: "\n\n- (void)a;\n" [[@LINE-11]]:11 -> [[@LINE-11]]:11

@implementation I3

- (void)requiredInstanceMethod:(int)y {
}

+ (void)b {
}

#ifdef HAS_OTHER
- (void) otherProtocolMethod:(int (^)(id<Proto2>))takesBlock { }
#endif

@end
// CHECK6: "\n\n+ (void)aRequiredInstanceMethod:(int (*)(void))function with:(id)p { \n  <#code#>\n}\n" [[@LINE-10]]:2 -> [[@LINE-10]]:2
// CHECK6-NEXT: "\n\n- (void)a { \n  <#code#>\n}\n" [[@LINE-8]]:2 -> [[@LINE-8]]:2
// CHECK6-NEXT: "- (void)otherProtocolMethod:(int (^)(id<Proto2>))takesBlock { \n  <#code#>\n}\n\n" [[@LINE-3]]:1 -> [[@LINE-3]]:1
// CHECK7: "\n\n+ (void)aRequiredInstanceMethod:(int (*)(void))function with:(id)p { \n  <#code#>\n}\n" [[@LINE-13]]:2 -> [[@LINE-13]]:2
// CHECK7-NEXT: "\n\n- (void)a { \n  <#code#>\n}\n" [[@LINE-11]]:2 -> [[@LINE-11]]:2

// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:62:1 -at=%s:79:1 %s | FileCheck --check-prefix=CHECK6 %s
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:62:1 -at=%s:79:1 %s -DHAS_OTHER | FileCheck --check-prefix=CHECK7 %s

@protocol ProtoWith3Methods

- (void)a;
- (void)b;
- (void)c;

@end

@interface I4 : Base<ProtoWith3Methods>

#ifndef USE_MACRO
- (void)b;

- (void)a; // comment

#else

#define METHOD(name) -(void)name;

METHOD(b)
METHOD(c) - (void)d;

#endif

@end
// CHECK8: "\n\n- (void)c;\n" [[@LINE-12]]:22 ->  [[@LINE-12]]:22
// CHECK9: "\n\n- (void)a;\n" [[@LINE-6]]:10 -> [[@LINE-6]]:10

// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:109:1 %s | FileCheck --check-prefix=CHECK8 %s
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:109:1 %s -D USE_MACRO | FileCheck --check-prefix=CHECK9 %s

@protocol NSObject

- (void)nsObjectMethod;

@end

@protocol SubProto

- (void)sub1;
- (id<SubProto>)sub2;

@end

@protocol SubProto2 <NSObject>

- (void)sub11;

@end

@protocol SuperProto <SubProto, NSObject, SubProto2>

@optional
- (void)mySub;

@end

@interface HasSubProtocolMethods: Base <SuperProto, NSObject>

@end
// CHECK10: "- (void)sub1;\n\n- (id<SubProto>)sub2;\n\n- (void)sub11;\n\n" [[@LINE-1]]:1 -> [[@LINE-1]]:1
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:158:1 %s | FileCheck --check-prefix=CHECK10 %s



@interface SuperClassWithSomeDecls : Base<SuperProto>

- (void)sub1;

@end

@interface SubClassOfSuperClassWithSomeDecls : SuperClassWithSomeDecls

#ifdef HAS_SUB1_OVERRIDE
- (void)sub1;
#endif
#ifdef HAS_SUB11
- (void)sub11;
#endif

@end
// CHECK11: "- (id<SubProto>)sub2;\n\n- (void)sub11;\n\n" [[@LINE-1]]:1
// CHECK12: "\n\n- (id<SubProto>)sub2;\n" [[@LINE-8]]:14
// CHECK12-NEXT: "- (void)sub11;\n\n" [[@LINE-3]]:1
// CHECK13: "- (id<SubProto>)sub2;\n\n" [[@LINE-4]]:1
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:172:1 %s | FileCheck --check-prefix=CHECK11 %s
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:172:1 %s -DHAS_SUB1_OVERRIDE | FileCheck --check-prefix=CHECK12 %s
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:172:1 %s -DHAS_SUB11 | FileCheck --check-prefix=CHECK13 %s

@implementation SubClassOfSuperClassWithSomeDecls

@end
// CHECK14: "- (id<SubProto>)sub2 { \n  <#code#>\n}\n\n- (void)sub11 { \n  <#code#>\n}\n\n" [[@LINE-1]]:1
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:190:1 %s | FileCheck --check-prefix=CHECK14 %s
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:190:1 %s -DHAS_SUB1_OVERRIDE | FileCheck --check-prefix=CHECK14 %s
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:190:1 %s -DHAS_SUB11 | FileCheck --check-prefix=CHECK14 %s
