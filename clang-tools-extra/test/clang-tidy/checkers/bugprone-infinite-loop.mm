// RUN: %check_clang_tidy %s bugprone-infinite-loop %t -- -- -fblocks

@interface I
-(void) instanceMethod;
+(void) classMethod;
@end

void plainCFunction() {
  int i = 0;
  int j = 0;
  while (i < 10) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i) are updated in the loop body [bugprone-infinite-loop]
    j++;
  }
}

@implementation I
- (void)instanceMethod {
  int i = 0;
  int j = 0;
  while (i < 10) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i) are updated in the loop body [bugprone-infinite-loop]
    j++;
  }
}

+ (void)classMethod {
  int i = 0;
  int j = 0;
  while (i < 10) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i) are updated in the loop body [bugprone-infinite-loop]
    j++;
  }
}
@end
