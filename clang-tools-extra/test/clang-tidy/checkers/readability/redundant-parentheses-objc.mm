// RUN: %check_clang_tidy %s readability-redundant-parentheses %t

@interface NSObject
@end

@interface NSNumber : NSObject
+ (NSNumber *)numberWithInt:(int)value;
@end

int width();

struct S {
  int m;
};

// The parentheses of a boxed expression are required syntax.
void boxedExpressions(int i, int j, int *a, struct S s) {
  (void)@(i);
  (void)@(width());
  (void)@(s.m);
  (void)@(a[0]);
  (void)@(42);

  // Redundant parentheses inside a boxed expression are still diagnosed.
  (void)@((j));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES: (void)@(j);
}
