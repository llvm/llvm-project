// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -fblocks -Wno-address-of-temporary -D"id=struct objc_object*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar:// 9039342

void *sel_registerName(const char *);
void objc_enumerationMutation(id);

@protocol CoreDAVLeafDataPayload @end

@class NSString;

@interface CoreDAVAction
- (id) context;
@end

@interface I
{
  id uuidsToAddActions;
}
@end

@implementation I
- (void) Meth {
  for (id<CoreDAVLeafDataPayload> uuid in uuidsToAddActions) {
    CoreDAVAction *action = 0;
    id <CoreDAVLeafDataPayload> payload = [action context];
  }
}
@end
