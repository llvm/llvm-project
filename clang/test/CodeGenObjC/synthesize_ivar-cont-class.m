// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

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
// CHECK: @"OBJC_IVAR_$_XCOrganizerDeviceNodeInfo.viewController"
@end

