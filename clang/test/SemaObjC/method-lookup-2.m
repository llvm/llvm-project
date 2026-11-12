// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
typedef signed char BOOL;

@protocol NSObject
+ alloc;
- init;
- (BOOL) isEqual:(id) object;
- (Class)class;
@end

@interface NSObject < NSObject > {} @end

@class NSString, NSPort;

@interface NSPortNameServer:NSObject
+ (NSPortNameServer *) systemDefaultPortNameServer;
@end

@interface NSMachBootstrapServer:NSPortNameServer + (id) sharedInstance; @end

enum {
  NSWindowsNTOperatingSystem = 1, NSWindows95OperatingSystem, NSSolarisOperatingSystem, NSHPUXOperatingSystem, NSMACHOperatingSystem, NSSunOSOperatingSystem, NSOSF1OperatingSystem
};

@interface NSRunLoop:NSObject {} @end

@interface NSRunLoop(NSRunLoopConveniences)
- (void) run;
@end

extern NSString *const NSWillBecomeMultiThreadedNotification;

@interface SenTestTool:NSObject {}
@end

@implementation SenTestTool
+ (void) initialize {}
+(SenTestTool *) sharedInstance { return 0; }
-(int) run { return 0; }
+(int) run {
  return[[self sharedInstance] run];
}
@end

@interface XX : NSObject

+ classMethod;

@end

@interface YY : NSObject
- whatever;
@end

@implementation YY 

- whatever {
  id obj = [[XX alloc] init];
  [[obj class] classMethod];
  return 0;
}

@end
