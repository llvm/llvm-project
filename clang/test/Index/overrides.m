
@protocol P1
- (void)protoMethod;
- (void)protoMethodWithParam:(int)param;
@end

@protocol P3
- (void)protoMethod;
@end

@protocol P2 <P1>
- (void)protoMethod;
@end

@interface A
- (void)method;
- (void)protoMethod;
+ (void)methodWithParam:(int)param;
@end

@interface B : A <P2, P3>
- (void)method;
- (void)protoMethod;
@end

@implementation B
- (void)method { }
+ (void)methodWithParam:(int)param { }
@end

@protocol P4 <P3>
- (void)protoMethod;
@end

@interface B(cat) <P4>
- (void)protoMethod;
@end

@interface B2
@end

@interface B2(cat)
-(void)meth;
@end

@interface I2 : B2
@end

@implementation I2
-(void)meth { }
@end

@protocol P5
-(void)kol;
-(void)kol;
@end

@protocol P6
@property (readonly) id prop1;
@property (readonly) id prop2;
-(void)meth;
@end

@interface I3 <P6>
@property (readwrite) id prop1;
@property (readonly) id bar;
@end

@interface I3()
@property (readwrite) id prop2;
@property (readwrite) id bar;
-(void)meth;
@end

@interface B4
-(id)prop;
-(void)setProp:(id)prop;
@end

@interface I4 : B4
@property (assign) id prop;
@end

@interface B5
@end

@interface I5 : B5
-(void)meth;
@end

@interface B5(cat)
-(void)meth;
@end

@implementation I5
-(void)meth{}
@end

// RUN: c-index-test -test-load-source local %s | FileCheck %s
// CHECK: overrides.m:12:9: ObjCInstanceMethodDecl=protoMethod:12:9 [Overrides @3:9]
// CHECK: overrides.m:22:9: ObjCInstanceMethodDecl=method:22:9 [Overrides @16:9]
// CHECK: overrides.m:23:9: ObjCInstanceMethodDecl=protoMethod:23:9 [Overrides @8:9, @12:9, @17:9, @32:9]
// CHECK: overrides.m:27:9: ObjCInstanceMethodDecl=method:27:9 (Definition) [Overrides @16:9]
// CHECK: overrides.m:28:9: ObjCClassMethodDecl=methodWithParam::28:9 (Definition) [Overrides @18:9]
// CHECK: overrides.m:32:9: ObjCInstanceMethodDecl=protoMethod:32:9 [Overrides @8:9]
// CHECK: overrides.m:36:9: ObjCInstanceMethodDecl=protoMethod:36:9 [Overrides @8:9, @12:9, @17:9, @32:9]
// CHECK: overrides.m:50:8: ObjCInstanceMethodDecl=meth:50:8 (Definition) [Overrides @43:8]
// CHECK: overrides.m:55:8: ObjCInstanceMethodDecl=kol:55:8 Extent=[55:1 - 55:12]
// CHECK: overrides.m:65:26: ObjCInstanceMethodDecl=prop1:65:26 [Overrides @59:25] Extent=[65:26 - 65:31]
// CHECK: overrides.m:65:26: ObjCInstanceMethodDecl=setProp1::65:26 Extent=[65:26 - 65:31]
// CHECK: overrides.m:70:26: ObjCInstanceMethodDecl=prop2:70:26 [Overrides @60:25] Extent=[70:26 - 70:31]
// CHECK: overrides.m:70:26: ObjCInstanceMethodDecl=setProp2::70:26 Extent=[70:26 - 70:31]
// CHECK: overrides.m:71:26: ObjCInstanceMethodDecl=setBar::71:26 Extent=[71:26 - 71:29]
// CHECK: overrides.m:72:8: ObjCInstanceMethodDecl=meth:72:8 [Overrides @61:8] Extent=[72:1 - 72:13]
// CHECK: overrides.m:81:23: ObjCInstanceMethodDecl=prop:81:23 [Overrides @76:6] Extent=[81:23 - 81:27]
// CHECK: overrides.m:81:23: ObjCInstanceMethodDecl=setProp::81:23 [Overrides @77:8] Extent=[81:23 - 81:27]
// CHECK: overrides.m:92:8: ObjCInstanceMethodDecl=meth:92:8 Extent=[92:1 - 92:13]
// CHECK: overrides.m:95:17: ObjCImplementationDecl=I5:95:17 (Definition) Extent=[95:1 - 97:2]
// CHECK: overrides.m:96:8: ObjCInstanceMethodDecl=meth:96:8 (Definition) [Overrides @92:8] Extent=[96:1 - 96:14]
