// RUN: %clang_cc1 -Wselector -x objective-c %s -include %s -verify
// expected-no-diagnostics
// rdar://16600230

#ifndef INCLUDED
#define INCLUDED

#pragma clang system_header

@interface NSObject @end
@interface NSString @end

@interface NSString (NSStringExtensionMethods)
- (void)compare:(NSString *)string;
@end

@interface MyObject : NSObject
@end

#else
int main(void) {
    (void)@selector(compare:);
}

@implementation MyObject

@end
#endif
