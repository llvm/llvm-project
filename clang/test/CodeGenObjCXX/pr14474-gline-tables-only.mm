// PR 14474
// RUN: %clang_cc1 -triple i386-apple-macosx10.6.0 -emit-llvm \
// RUN:   -debug-info-kind=line-tables-only -x objective-c++ -o /dev/null %s
// RUN: %clang_cc1 -triple i386-apple-macosx10.6.0 -emit-llvm \
// RUN:   -debug-info-kind=line-directives-only -x objective-c++ -o /dev/null %s

typedef signed char BOOL;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject - (BOOL)isEqual:(id)object;
@end
@protocol NSCoding - (void)encodeWithCoder:(NSCoder *)aCoder;
@end 
@interface NSObject <NSObject> { }
@end    
@interface NSResponder : NSObject <NSCoding> { }
@end
@protocol NSValidatedUserInterfaceItem - (SEL)action;
@end
@protocol NSUserInterfaceValidations - (BOOL)validateUserInterfaceItem:(id
<NSValidatedUserInterfaceItem>)anItem;
@end
@interface NSRunningApplication : NSObject { }
@end
@interface NSApplication : NSResponder <NSUserInterfaceValidations> { }
@end
@implementation MockCrApp + (NSApplication*)sharedApplication { }
@end
