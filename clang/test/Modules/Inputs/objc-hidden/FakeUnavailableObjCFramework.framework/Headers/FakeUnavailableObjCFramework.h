#include <X.h>

__attribute__((availability(macosx,introduced=1066.0)))  __attribute__((availability(ios,introduced=1066.0)))
@interface UnavailableObjCClass : NSObject
- (void)someMethod;
@end

