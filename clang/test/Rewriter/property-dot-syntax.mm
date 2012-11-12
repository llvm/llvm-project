// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar:// 8520727

void *sel_registerName(const char *);

@class NSString;

@protocol CoreDAVAccountInfoProvider
- (NSString *)userAgentHeader;
@end

@interface CoreDAVTask
{
  id<CoreDAVAccountInfoProvider> _accountInfoProvider;
}
- (void)METHOD;
@end

@implementation CoreDAVTask
- (void)METHOD {
  if ([_accountInfoProvider userAgentHeader]) {
  }
  if (_accountInfoProvider.userAgentHeader) {
  }
}
@end

//rdar: // 8541517
@interface A { }
@property (retain) NSString *scheme;
@end

@interface B : A {
	NSString* _schemeName;
}
@end


@implementation B
-(void) test {
 B *b;
 b.scheme = _schemeName; // error because of this line
}
@end

