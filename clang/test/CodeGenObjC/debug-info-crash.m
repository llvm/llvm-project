// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple i386-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -fblocks -g -S %s -o -

// rdar://7556129
@implementation test
- (void)wait {
  ^{};
}
@end

// PR4894
@interface I0 {
  I0 *_iv0;
}
@end
@protocol P0 @end

@interface I1 @end
@implementation I1
- (I0<P0> *) im0 {
  // CHECK: @"\01-[I1 im0]"
  // CHECK: llvm.dbg.func.start
  return 0;
}
@end

// PR4541
@class NSString;
@interface NSAttributedString 
- (NSString *)string;
@end 
@interface NSMutableAttributedString : NSAttributedString 
@end 
@class NSImage;
@implementation CYObjectsController 
+ (void)initialize {
}
+ (NSAttributedString *)attributedStringWithString:(id)string image:(NSImage *)image  {
  NSMutableAttributedString *attrStr;
}
@end
