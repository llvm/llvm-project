// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -fblocks -Wno-address-of-temporary -Did="void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar:// 8558702

@class NSString;
@interface NSObject @end

@protocol P
@property (retain) NSString* test;
@end


@interface A : NSObject <P> {
	NSString* _test;
}
@end


@implementation A
@synthesize test=_test;
@end

