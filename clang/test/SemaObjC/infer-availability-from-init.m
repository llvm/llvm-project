// RUN: %clang_cc1 -triple x86_64-apple-macosx-10.9 -Wunguarded-availability -fblocks -fsyntax-only -verify %s

__attribute__((objc_root_class))
@interface NSObject
+(instancetype)new;
-(instancetype)init;
@end

@interface MyObject : NSObject
-(instancetype)init __attribute__((unavailable)); // expected-note{{'init' has been explicitly marked unavailable here}}
@end

void usemyobject(void) {
  [MyObject new]; // expected-error{{'new' is unavailable}}
}

@interface MyOtherObject : NSObject
+(instancetype)init __attribute__((unavailable));
+(instancetype)new;
@end

void usemyotherobject(void) {
  [MyOtherObject new]; // no error; new is overrideen.
}

@interface NotGoodOverride : NSObject
+(instancetype)init __attribute__((unavailable));
-(instancetype)new;
+(instancetype)new: (int)x;
@end

void usenotgoodoverride(void) {
  [NotGoodOverride new]; // no error
}

@interface NotNSObject
+(instancetype)new;
-(instancetype)init;
@end

@interface NotMyObject : NotNSObject
-(instancetype)init __attribute__((unavailable));
@end

void usenotmyobject(void) {
  [NotMyObject new]; // no error; this isn't NSObject
}

@interface FromSelf : NSObject
-(instancetype)init __attribute__((unavailable));
+(FromSelf*)another_one;
@end

@implementation FromSelf
+(FromSelf*)another_one {
  [self new];
}
@end

@interface NoInit : NSObject
-(instancetype)init __attribute__((unavailable)); // expected-note {{'init' has been explicitly marked unavailable here}}
@end

@interface NoInitSub : NoInit @end

@implementation NoInitSub
-(void)meth:(Class)c {
  [c new]; // No error; unknown interface.
  [NoInitSub new]; // expected-error {{'new' is unavailable}}
}
@end
