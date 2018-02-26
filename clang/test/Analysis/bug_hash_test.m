// RUN: %clang_analyze_cc1 -fblocks -analyzer-checker=core,debug.ExprInspection %s -verify

void clang_analyzer_hashDump(int);

@protocol NSObject
+ (id)alloc;
- (id)init;
@end

@protocol NSCopying
@end

__attribute__((objc_root_class))
@interface NSObject <NSObject>
- (void)method:(int)arg param:(int)arg2;
@end

@implementation NSObject
+ (id)alloc {
  return 0;
}
- (id)init {
  return self;
}
- (void)method:(int)arg param:(int)arg2 {
  clang_analyzer_hashDump(5); // expected-warning {{debug.ExprInspection$NSObject::method:param:$27$clang_analyzer_hashDump(5);$Category}}
}
@end


void testBlocks() {
  int x = 5;
  ^{
    clang_analyzer_hashDump(x); // expected-warning {{debug.ExprInspection$$29$clang_analyzer_hashDump(x);$Category}}
  }();
}
