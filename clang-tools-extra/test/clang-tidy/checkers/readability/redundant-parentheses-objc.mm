// RUN: %check_clang_tidy %s readability-redundant-parentheses %t

@interface NSObject
@end

@interface NSNumber : NSObject
+ (NSNumber *)numberWithInt:(int)value;
@end

@interface NSValue : NSObject
+ (NSValue *)valueWithBytes:(const void *)bytes objCType:(const char *)type;
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

struct __attribute__((objc_boxable)) BoxableStruct {
  int x;
};

// Boxing a struct copy-constructs it, so the parentheses are not the immediate
// child of the boxed expression.
void boxedStruct(BoxableStruct bs) {
  (void)@(bs);

  // Redundant parentheses inside a boxed expression are still diagnosed.
  (void)@(BoxableStruct((bs)));
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: redundant parentheses around expression [readability-redundant-parentheses]
  // CHECK-FIXES: (void)@(BoxableStruct(bs));
}
