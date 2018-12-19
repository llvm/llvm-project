// RUN: %clang_cc1 -emit-llvm -fobjc-arc -debug-info-kind=limited -triple x86_64-apple-darwin10 %s -o - | FileCheck %s
// Ensure that the line info is making sense:
// ARC cleanups should be at the closing '}'.
@protocol NSObject
@end

@interface NSObject <NSObject> {}
@end

@protocol NSCopying
@end

@protocol NSCoding
@end

typedef double CGFloat;
struct CGRect {};
typedef struct CGRect CGRect;
typedef CGRect NSRect;
NSRect NSMakeRect(CGFloat x, CGFloat y, CGFloat w, CGFloat h);
@interface NSBezierPath : NSObject <NSCopying, NSCoding>
+ (NSBezierPath *)bezierPathWithRoundedRect:(NSRect)rect xRadius:(CGFloat)xRadius yRadius:(CGFloat)yRadius;
@end
@implementation AppDelegate : NSObject {}
- (NSBezierPath *)_createBezierPathWithWidth:(CGFloat)width height:(CGFloat)height radius:(CGFloat)radius lineWidth:(CGFloat)lineWidth
{
  NSRect rect = NSMakeRect(0, 0, width, height);
  NSBezierPath *path = [NSBezierPath bezierPathWithRoundedRect:rect xRadius:radius yRadius:radius];
  CGFloat pattern[2];
  // CHECK: define {{.*}}_createBezierPathWithWidth
  // CHECK: load {{.*}} %path, align {{.*}}, !dbg ![[RET:[0-9]+]]
  // CHECK: call void @llvm.objc.storeStrong{{.*}} !dbg ![[ARC:[0-9]+]]
  // CHECK: call {{.*}} @llvm.objc.autoreleaseReturnValue{{.*}} !dbg ![[ARC]]
  // CHECK: ret {{.*}} !dbg ![[ARC]]
  // CHECK: ![[RET]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
  return path;
  // CHECK: ![[ARC]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
}
@end
