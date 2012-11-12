// RUN: %clang_cc1 -emit-llvm -o %t %s
// RUN: grep '@"OBJC_IVAR_$_XCOrganizerDeviceNodeInfo.viewController"' %t

// PR13820
// REQUIRES: LP64

@interface XCOrganizerNodeInfo
@property (readonly, retain) id viewController;
@end

@interface XCOrganizerDeviceNodeInfo : XCOrganizerNodeInfo
@end

@interface XCOrganizerDeviceNodeInfo()
@property (retain) id viewController;
@end

@implementation XCOrganizerDeviceNodeInfo
@synthesize viewController;
@end

