// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

typedef signed char BOOL;

@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;

@protocol NSObject
- (BOOL)isEqual:(id)object;
@end

@interface NSObject <NSObject> {}
@end

@interface XCDeviceWillExecuteInfoBaton : NSObject {}
  @property (retain) __attribute__((objc_gc(strong))) NSString *sdkPath;
@end

@implementation XCDeviceWillExecuteInfoBaton
  @synthesize sdkPath; 
@end

