// RUN: %clang_cc1 -x objective-c -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar://6969189

@class XX;
@class YY, ZZ, QQ;
@class ISyncClient, SMSession, ISyncManager, ISyncSession, SMDataclassInfo, SMClientInfo,
    DMCConfiguration, DMCStatusEntry;

@interface QQ

@end

@interface SMDataclassInfo : QQ
- (XX*) Meth;
- (DMCStatusEntry*)Meth2;
@end

@implementation SMDataclassInfo
- (XX*) Meth { return 0; }
- (DMCStatusEntry*)Meth2 { return 0; }
@end

@interface YY 
{
  ISyncClient *p1;
  ISyncSession *p2;
}
@property (copy) ISyncClient *p1;
@end

@implementation YY
@synthesize p1;
@end

