// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -fblocks -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

void *sel_registerName(const char *);

@class NSString;
@interface CoreDAVDiscoveryAccountInfo  {
  NSString *_scheme;
}
@property (retain) NSString *scheme;
- (void) Meth ;
@end

@implementation CoreDAVDiscoveryAccountInfo
@synthesize scheme=_scheme;
- (void) Meth {
  CoreDAVDiscoveryAccountInfo *discoveryInfo;
  discoveryInfo.scheme = @"https";
}
@end
