
int capturedBlock(void (^block)(int x, int (^)())) {
  return capturedBlock(block);
}

// CHECK1: extracted(void (^block)(int, int (^)()))

typedef void (^BlockTypedef)();

int capturedBlockTypedef(BlockTypedef fp) {
  return capturedBlockTypedef(fp);
}
// CHECK1: extracted(BlockTypedef fp)

// RUN: clang-refactor-test perform -action extract -selected=%s:3:10-3:29 -selected=%s:11:10-11:34 %s -fblocks | FileCheck --check-prefix=CHECK1 %s

@interface I
@end

@implementation I

- (void)method {
  void (^block)(int x, int (^)());
  block(2, ^ (void) { return 0; });
}
// CHECK2: - (void)extracted:(void (^)(int, int (^)()))block {

@end

// RUN: clang-refactor-test perform -action extract-method -selected=%s:24:3-24:35 %s -fblocks | FileCheck --check-prefix=CHECK2 %s
