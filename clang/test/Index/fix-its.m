// RUN: c-index-test -test-load-source all %s -Wno-objc-root-class > %t 2>&1
// RUN: FileCheck -input-file=%t %s

@class NSString;
void _rdar_12584554_A (volatile const void * object, volatile const void * selector, const char * functionName, const char * fileName, unsigned int lineNumber, NSString * msgFormat, ...);
#define _rdar_12584554_B(self,_format_and_args_...) \
    do{ _rdar_12584554_A(&self,&_cmd,__PRETTY_FUNCTION__,__FILE__,__LINE__, _format_and_args_); }while(0)
#define _rdar_12584554_C(_format_and_args_...) \
    _rdar_12584554_B(self, _format_and_args_)

@interface RDar12584554
@end

// This test case tests that the "@" is properly inserted before the '"', even in the
// presence of a nested macro chain.
@implementation RDar12584554
- (void) test:(int)result {
    _rdar_12584554_C("ted");
}
@end

// CHECK: FIX-IT: Insert "@" at 18:22
