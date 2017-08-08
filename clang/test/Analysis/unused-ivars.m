// RUN: %clang_analyze_cc1 -fblocks -analyzer-checker=osx.cocoa.UnusedIvars -verify -Wno-objc-root-class %s

//===--- BEGIN: Delta-debugging reduced headers. --------------------------===//

@protocol NSObject
- (id)retain;
- (oneway void)release;
@end
@interface NSObject <NSObject> {}
- (id)init;
+ (id)alloc;
@end

//===--- END: Delta-debugging reduced headers. ----------------------------===//

// This test case tests the basic functionality of the unused ivar test.
@interface TestA {
@private
  int x; // expected-warning {{Instance variable 'x' in class 'TestA' is never used}}
}
@end
@implementation TestA @end

// This test case tests whether the unused ivar check handles blocks that
// reference an instance variable. (<rdar://problem/7075531>)
@interface TestB : NSObject {
@private
  id _ivar; // no-warning
}
@property (readwrite,retain) id ivar;
@end

@implementation TestB
- (id)ivar {
  __attribute__((__blocks__(byref))) id value = ((void*)0);
  void (^b)() = ^{ value = _ivar; };
  b();
  return value;
}

- (void)setIvar:(id)newValue {
  void (^b)() = ^{ [_ivar release]; _ivar = [newValue retain]; };
  b();
}
@end

//===----------------------------------------------------------------------===//
// <rdar://problem/6260004> Detect that ivar is in use, if used in category 
//  in the same file as the implementation
//===----------------------------------------------------------------------===//

@protocol Protocol6260004
- (id) getId;
@end

@interface RDar6260004 {
@private
  id x; // no-warning
}
@end
@implementation RDar6260004 @end
@implementation RDar6260004 (Protocol6260004)
- (id) getId {
  return x;
}
@end

//===----------------------------------------------------------------------===//
// <rdar://problem/7254495> - ivars referenced by lexically nested functions
//  should not be flagged as unused
//===----------------------------------------------------------------------===//

@interface RDar7254495 {
@private
  int x; // no-warning
}
@end

@implementation RDar7254495
int radar_7254495(RDar7254495 *a) {
  return a->x;
}
@end

//===----------------------------------------------------------------------===//
// <rdar://problem/7353683> - consult attribute((unused)) to silence warnings
// about unused instance variables
//===----------------------------------------------------------------------===//

@interface RDar7353683 {
@private
  id x __attribute__((unused));
}
@end

@implementation RDar7353683
@end
//===----------------------------------------------------------------------===//
// <rdar://problem/8481311> Unused bitfield ivars trigger cause weird
// diagnostic: "Instance variable '' in class..."
//===----------------------------------------------------------------------===//

@interface RDar8481311 {
@private
    unsigned bitfield:1; // expected-warning {{Instance variable 'bitfield' in class 'RDar8481311' is never used}}
}
@end

@implementation RDar8481311
@end

@class NSString;
@interface Radar11059352_1 {
@private
    NSString *_pathString;
}
@property (readonly, strong) NSString *pathString;
@end

@interface Radar11059352 {
@private
Radar11059352_1 *_workspacePath;
}
@end

@implementation Radar11059352

- (void)useWorkspace {
    NSString *workspacePathString = _workspacePath.pathString;
}
@end