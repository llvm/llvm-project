// RUN: %check_clang_tidy %s bugprone-infinite-loop %t -- -- -fblocks -fexceptions
// RUN: %check_clang_tidy %s bugprone-infinite-loop %t -- -- -fblocks -fobjc-arc -fexceptions

@interface I
+ (void)foo;
+ (void)bar;
+ (void)baz __attribute__((noreturn));
+ (instancetype)alloc;
- (instancetype)init;
@end

_Noreturn void term();

void plainCFunction() {
  int i = 0;
  int j = 0;
  int a[10];

  while (i < 10) {
    // no warning, function term has C noreturn attribute
    term();
  }
  while (i < 10) {
    // no warning, class method baz has noreturn attribute
    [I baz];
  }
  while (i + j < 10) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i, j) are updated in the loop body [bugprone-infinite-loop]
    [I foo];
  }
  while (i + j < 10) {
    [I foo];
    [I baz]; // no warning, class method baz has noreturn attribute
  }

  void (^block)() = ^{
  };
  void __attribute__((noreturn)) (^block_nr)(void) = ^void __attribute__((noreturn)) (void) { throw "err"; };

  while (i < 10) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i) are updated in the loop body [bugprone-infinite-loop]
    block();
  }
  while (i < 10) {
    // no warning, the block has "noreturn" arribute
    block_nr();
  }
}

@implementation I
+ (void)bar {
}

+ (void)foo {
  static int i = 0;

  while (i < 10) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i) are updated in the loop body [bugprone-infinite-loop]
    [I bar];
  }
}
@end
