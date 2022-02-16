// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin %s -o - | FileCheck %s
// rdar://16462586

__attribute__((objc_runtime_name("MySecretNamespace.Protocol")))
@protocol Protocol
- (void) MethodP;
+ (void) ClsMethodP;
@end

__attribute__((objc_runtime_name("MySecretNamespace.Protocol2")))
@protocol Protocol2
- (void) MethodP2;
+ (void) ClsMethodP2;
@end

__attribute__((objc_runtime_name("MySecretNamespace.Protocol3")))
@protocol Protocol3
@end

__attribute__((objc_runtime_name("MySecretNamespace.Message")))
@interface Message <Protocol, Protocol2> {
  id MyIVAR;
}

@property(retain) Message *msgProp;
@property(retain) Message<Protocol3> *msgProtoProp;
@property(retain) id<Protocol3> idProtoProp;

@end

@implementation Message
- (id) MyMethod {
  return MyIVAR;
}

+ (id) MyClsMethod {
  return 0;
}

- (void) MethodP{}
- (void) MethodP2{}

+ (void) ClsMethodP {}
+ (void) ClsMethodP2 {}
@end

// rdar://16877359
__attribute__((objc_runtime_name("foo")))
@interface SLREarth
- (instancetype)init;
+ (instancetype)alloc;
@end

id Test16877359(void) {
    return [SLREarth alloc];
}

// CHECK: @"OBJC_IVAR_$_MySecretNamespace.Message.MyIVAR" ={{.*}} global i64 0
// CHECK: @"OBJC_CLASS_$_MySecretNamespace.Message" ={{.*}} global %struct._class_t
// CHECK: @"OBJC_METACLASS_$_MySecretNamespace.Message" ={{.*}} global %struct._class_t

// CHECK: private unnamed_addr constant [42 x i8] c"T@\22MySecretNamespace.Message\22,&,V_msgProp\00"
// CHECK: private unnamed_addr constant [76 x i8] c"T@\22MySecretNamespace.Message<MySecretNamespace.Protocol3>\22,&,V_msgProtoProp\00"
// CHECK: private unnamed_addr constant [50 x i8] c"T@\22<MySecretNamespace.Protocol3>\22,&,V_idProtoProp\00"

// CHECK: @"OBJC_CLASS_$_foo" = external global %struct._class_t
// CHECK: define internal i8* @"\01-[Message MyMethod]"
// CHECK: [[IVAR:%.*]] = load i64, i64* @"OBJC_IVAR_$_MySecretNamespace.Message.MyIVAR"
