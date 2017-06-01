// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -fblocks -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

extern "C" {
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

extern "C" {
@class CCC;
@class Protocol, P , Q;
int I,J,K;
};

};


// rdar://15027032
@interface ISDPropertyChangeGroup
@end

@implementation ISDPropertyChangeGroup
@class ISDClientState;
- (id)lastModifiedGeneration : (ISDClientState *) obj
{
  return obj ;
}
@end
