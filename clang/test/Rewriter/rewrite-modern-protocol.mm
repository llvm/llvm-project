// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

@protocol ROOT @end

@protocol P1 @end

@protocol P2<ROOT> @end

@class NSObject;

@protocol PROTO <P1, P2>
- (id) INST_METHOD;
+ (id) CLASS_METHOD : (id)ARG;
@property id Prop_in_PROTO;
@optional
- (id) opt_instance_method;
+ (id) opt_class_method;
@property (readonly, retain) NSObject *AnotherProperty;
@required
- (id) req;
@optional
- (id) X_opt_instance_method;
+ (id) X_opt_class_method;
@end

@interface INTF <PROTO, ROOT>
@end

@implementation INTF
@end
